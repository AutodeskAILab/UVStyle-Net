import torch
import torch.nn as nn
import torch.nn.functional as F
import networks.nn_utils as nnu
from networks.layers import WeightedConv
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


class FeatureNetEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        """
        Voxel encoder based on FeatureNet
        Expects an input tensor of siez (64 x 64 x 64)

        Reference:
        FeatureNet: Machining feature recognition based on 3D Convolution Neural Network
        Zhibo Zhang, Prakhar Jaiswal, Rahul Rai
        Computer-Aided Design 101 (2018) 12–22
        """
        super(FeatureNetEncoder, self).__init__()
        self.conv1 = nnu.conv3d(1, 32, kernel_size=7, padding=3, stride=2)
        self.conv2 = nnu.conv3d(32, 32, kernel_size=5, padding=2, stride=1)
        self.conv3 = nnu.conv3d(32, 64, kernel_size=4, padding=2, stride=1)
        self.conv4 = nnu.conv3d(64, 64, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool3d(2, stride=2)
        self.fc = nn.Linear(262144, latent_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.fc(x.view(batch_size, -1))
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, emb_dims):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nnu.conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nnu.conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nnu.conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nnu.conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nnu.conv1d(128, emb_dims, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        return x


class UVNetCurveEncoder(nn.Module):
    def __init__(self, input_channels="xyz_normals", output_dims=64):
        super(UVNetCurveEncoder, self).__init__()
        assert input_channels in ("xyz_only", "xyz_normals")
        self.input_channels = input_channels
        num_inp_channels = 3 if input_channels == "xyz_only" else 6
        # Convolution layer 1
        self.conv1 = nnu.conv1d(
            num_inp_channels, 64, kernel_size=3, padding=1, bias=False
        )
        # Convolution layer 2
        self.conv2 = nnu.conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        # Convolution layer 3
        self.conv3 = nnu.conv1d(128, 256, kernel_size=3, padding=1, bias=False)
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
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class UVNetSurfaceEncoder(nn.Module):
    def __init__(
        self,
        input_channels="xyz_normals",
        output_dims=64,
        mask_mode="channel",
        weighted=False,
    ):
        super(UVNetSurfaceEncoder, self).__init__()
        assert mask_mode in ("multiply", "channel")
        self.mask_mode = mask_mode
        assert input_channels in ("xyz_only", "xyz_normals")
        self.input_channels = input_channels
        num_inp_channels = 3 if input_channels == "xyz_only" else 6
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

        self.activations = {}

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, WeightedConv)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward_mask_channel(self, inp):
        if self.weighted:
            self.conv1[0].set_weights(inp[:, 7, :, :], inp[:, 8, :, :], inp[:, 9, :, :])
        if self.input_channels == "xyz_only":
            x = inp[:, [0, 1, 2, 6], :, :]  # xyz, mask
        else:
            x = inp[:, :7, :, :]  # xyz, normals, mask
        self.activations = {'feats': x[:, :6, :, :]}
        batch_size = x.size(0)
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
        if self.input_channels == "xyz_only":
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
        raise ValueError(
            "Unknown mask mode {}, expected one of ('channel', 'multiply')".format(
                self.mask_mode
            )
        )


class _ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(_ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class _MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


def _get_graph_pooling_layer(graph_pooling_type):
    if graph_pooling_type == "sum":
        return SumPooling()
    elif graph_pooling_type == "mean":
        return AvgPooling()
    elif graph_pooling_type == "max":
        return MaxPooling()
    raise ValueError("Expected graph_pooling_type to be one of ('sum', 'mean', 'max')")


class UVNetGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        neighbor_pooling_type="sum",
        graph_pooling_type="max",
        learn_eps=True,
        num_layers=3,
        num_mlp_layers=2,
    ):
        super(UVNetGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = _MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = _MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(_ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.final_layer = nn.Linear(hidden_dim, output_dim)
        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(0.5)

        self.pool = _get_graph_pooling_layer(graph_pooling_type)

        self.activations = {}

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]
        self.activations = {}
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
            self.activations[f'GIN_{i + 1}'] = h.unsqueeze(-1)

        out = hidden_rep[-1]
        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return out, score_over_layer


class SVGVAEImageEncoder(nn.Module):
    def __init__(self):
        """
        SVGVAE Image Encoder, accepts 64 x 64 images.
        Reference:

        """
        super(SVGVAEImageEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2), nn.InstanceNorm2d(32), nn.ReLU()
        )  # 64 x 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.InstanceNorm2d(32), nn.ReLU()
        )  # 32 x 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1, padding=2), nn.InstanceNorm2d(64), nn.ReLU()
        )  # 32 x 32
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 5, stride=2, padding=2), nn.InstanceNorm2d(64), nn.ReLU()
        )  # 16 x 16

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.InstanceNorm2d(64), nn.ReLU()
        )  # 8 x 8
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.InstanceNorm2d(64), nn.ReLU()
        )  # 4 x 4

    def forward(self, imgs):
        batch_size = imgs.size(0)
        x = self.conv1(imgs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(batch_size, -1)
        return x  # batch_size x 1024


class MolGANDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(MolGANDiscriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = lay.GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout)
        self.agg_layer = lay.GraphAggregation(
            graph_conv_dim[-1], aux_dim, b_dim, dropout
        )

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim] + linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activatation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node
        h = self.gcn_layer(annotations, adj)
        annotations = torch.cat(
            (h, hidden, node) if hidden is not None else (h, node), -1
        )
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        # Need to implemente batch discriminator #
        ##########################################

        output = self.output_layer(h)
        output = activatation(output) if activatation is not None else output

        return output, h
