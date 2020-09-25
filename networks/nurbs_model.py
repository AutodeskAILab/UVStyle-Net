import torch
import torch.nn as nn
import torch.nn.functional as F
import networks.nn_utils as nnu
from networks.wconv import WeightedConv


def get_nurbs_model(nurbs_model_type, mask_mode="channel", area_as_channel=False, output_dims=64,
                    input_channels='xyz_normals'):
    if nurbs_model_type == "cnn":
        return NURBSFeatureExtractor(input_channels=input_channels, output_dims=output_dims, mask_mode=mask_mode, weighted=False)
    if nurbs_model_type == "wcnn":
        return NURBSFeatureExtractor(input_channels=input_channels, output_dims=output_dims, mask_mode=mask_mode, weighted=True)
    raise ValueError("Invalid nurbs model type: {}, expected one of ('cnn', 'wcnn')".format(nurbs_model_type))


def get_nurbs_curve_model(nurbs_model_type, input_channels='xyz_normals', output_dims=64):
    if nurbs_model_type == "cnn":
        return NURBSCurveFeatureExtractor(input_channels=input_channels, output_dims=output_dims)
    if nurbs_model_type == "wcnn":
        raise NotImplementedError("")
    raise ValueError("Invalid nurbs curve model type: {}, expected one of ('cnn', 'wcnn')".format(nurbs_model_type))

class NURBSFeatureExtractor(nn.Module):
    def __init__(self, input_channels='xyz_normals', output_dims=64, mask_mode="channel", weighted=True):
        super(NURBSFeatureExtractor, self).__init__()
        assert mask_mode in ("multiply", "channel")
        self.mask_mode = mask_mode
        assert input_channels in ('xyz_only', 'xyz_normals')
        self.input_channels = input_channels
        num_inp_channels = 3 if input_channels == 'xyz_only' else 6
        if mask_mode == "channel":
            num_inp_channels += 1
        if weighted:
            self.conv1 = nnu.wconv(num_inp_channels, 64, 3, padding=1)
        else:
            self.conv1 = nnu.conv(num_inp_channels, 64, 3, padding=1)
        self.weighted = weighted
        self.conv2 = nnu.conv(64, 128, 3, padding=1, bias=False)
        self.conv3 = nnu.conv(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nnu.fc(256, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)
        self.activations = None

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, WeightedConv)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward_mask_channel(self, inp):
        if self.weighted:
            self.conv1[0].set_weights(inp[:, 7, :, :], inp[:, 8, :, :], inp[:, 9, :, :])
        if self.input_channels == 'xyz_only':
            x = inp[:, [0, 1, 2, 6], :, :]  # xyz, mask
        else:
            x = inp[:, :7, :, :]  # xyz, normals, mask
        batch_size = x.size(0)
        self.activations = {}
        self.activations['feats'] = x[:, :6, :, :]
        x = self.conv1(x)
        self.activations['conv1'] = x
        x = self.conv2(x)
        self.activations['conv2'] = x
        x = self.conv3(x)
        self.activations['conv3'] = x
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        self.activations['fc'] = x.unsqueeze(-1)
        return x

    def forward_mask_multiply(self, inp):
        if self.weighted:
            self.conv1[0].set_weights(inp[:, 7, :, :], inp[:, 8, :, :], inp[:, 9, :, :])
        mask = inp[:, 6, :, :].unsqueeze(1)
        if self.input_channels == 'xyz_only':
            x = inp[:, :3, :, :]  # xyz
        else:
            x = inp[:, :6, :, :]  # xyz, normals
        batch_size = x.size(0)
        x = self.conv1(x * mask)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

    def forward(self, inp):
        if self.mask_mode == "channel":
            return self.forward_mask_channel(inp)
        if self.mask_mode == "multiply":
            return self.forward_mask_multiply(inp)
        raise ValueError("Unknown mask mode {}, expected one of ('channel', 'multiply')".format(self.mask_mode))


class NURBSCurveFeatureExtractor(nn.Module):
    def __init__(self, input_channels='xyz_normals', output_dims=64):
        super(NURBSCurveFeatureExtractor, self).__init__()
        assert input_channels in ('xyz_only', 'xyz_normals')
        self.input_channels = input_channels
        num_inp_channels = 3 if input_channels == 'xyz_only' else 6
        # Convolution layer 1
        self.conv1 = nnu.conv1d(num_inp_channels, 64, kernel_size=3, padding=1, bias=False)
        self.pool1 = nn.MaxPool1d(2)
        # Convolution layer 2
        self.conv2 = nnu.conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.pool2 = nn.MaxPool1d(2)
        # Convolution layer 3
        self.conv3 = nnu.conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.pool3 = nn.MaxPool1d(2)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nnu.fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        batch_size = x.size(0)
        # Take only xyz, normals for now
        if self.input_channels == "xyz_only":
            x = x[:, :3, :]
        else:
            x = x[:, :6, :]
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x