import torch
import torch.nn as nn

import networks.nn_utils as nnu


def get_face_model(output_dims=64, input_channels='xyz_normals'):
    return UVNetFaceModel(input_channels=input_channels, output_dims=output_dims)


class UVNetFaceModel(nn.Module):
    def __init__(self, input_channels='xyz_normals', output_dims=64):
        super(UVNetFaceModel, self).__init__()
        assert input_channels in ('xyz_only', 'xyz_normals')
        self.input_channels = input_channels
        num_inp_channels = 3 if input_channels == 'xyz_only' else 6
        num_inp_channels += 1

        self.conv1 = nnu.conv(num_inp_channels, 64, 3, padding=1)
        self.conv2 = nnu.conv(64, 128, 3, padding=1, bias=False)
        self.conv3 = nnu.conv(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nnu.fc(256, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)
        self.activations = None

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward_mask_channel(self, inp):
        if self.input_channels == 'xyz_only':
            x = inp[:, [0, 1, 2, 6], :, :]  # xyz, mask
        else:
            x = inp[:, :7, :, :]  # xyz, normals, mask
        batch_size = x.size(0)
        self.activations = {}
        self.activations['feats'] = inp[:, :7, :, :]
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

    def forward(self, inp):
        return self.forward_mask_channel(inp)
