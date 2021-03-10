from torch import nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):
        # input : 16x9x9
        # adj : 16x4x9x9

        hidden = torch.stack([self.linear1(input) for _ in range(adj.size(1))], 1)
        hidden = torch.einsum("bijk,bikl->bijl", (adj, hidden))
        hidden = torch.sum(hidden, 1) + self.linear1(input)
        hidden = activation(hidden) if activation is not None else hidden
        hidden = self.dropout(hidden)

        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum("bijk,bikl->bijl", (adj, output))
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class GraphAggregation(Module):
    def __init__(self, in_features, out_features, b_dim, dropout):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(
            nn.Linear(in_features + b_dim, out_features), nn.Sigmoid()
        )
        self.tanh_linear = nn.Sequential(
            nn.Linear(in_features + b_dim, out_features), nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i, j), 1)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class WeightedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super(WeightedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.unfold = nn.Unfold(
            kernel_size=(kernel_size, kernel_size), padding=(padding, padding)
        )
        self.weight = nn.Parameter(
            torch.zeros((out_channels, in_channels, kernel_size, kernel_size)).float()
        )
        self.bias = (
            nn.Parameter(torch.zeros((1, out_channels, 1, 1)).float()) if bias else None
        )
        self.ff = None

        # UV params for weighting
        self.us = torch.ones((1, 9, 1)).float().cuda()
        self.vs = torch.ones((1, 9, 1)).float().cuda()
        # Row 1
        self.us[:, 0, :] = -0.1
        self.vs[:, 0, :] = -0.1
        self.us[:, 1, :] = 0.0
        self.vs[:, 1, :] = -0.1
        self.us[:, 2, :] = +0.1
        self.vs[:, 2, :] = -0.1
        # Row 2
        self.us[:, 3, :] = -0.1
        self.vs[:, 3, :] = 0.0
        self.us[:, 4, :] = 0.0
        self.vs[:, 4, :] = 0.0
        self.us[:, 5, :] = +0.1
        self.vs[:, 5, :] = 0.0
        # Row 3
        self.us[:, 6, :] = -0.1
        self.vs[:, 6, :] = +0.1
        self.us[:, 7, :] = 0.0
        self.vs[:, 7, :] = +0.1
        self.us[:, 8, :] = +0.1
        self.vs[:, 8, :] = +0.1

    def set_weights(self, ffE, ffF, ffG):
        with torch.no_grad():
            ffE_unf = F.unfold(ffE.unsqueeze(1), kernel_size=(1, 1))
            ffF_unf = F.unfold(ffF.unsqueeze(1), kernel_size=(1, 1))
            ffG_unf = F.unfold(ffG.unsqueeze(1), kernel_size=(1, 1))
            ff = (
                ffE_unf * self.us * self.us
                + 2.0 * ffF_unf * self.us * self.vs
                + ffG_unf * self.vs * self.vs
            )
            self.ff = (1.0 - torch.softmax(ff.view(-1), dim=0)).view(*ff.shape)

    def forward(self, x):
        batch_size = x.size(0)
        x_unf = self.unfold(x)
        w_flat = self.weight.view(self.out_channels, -1)
        if self.ff is not None:
            ff_unf = self.ff.repeat((1, x_unf.size(1) // self.ff.size(1), 1))
            x_unf = x_unf * ff_unf
        out_unf = (
            x_unf.transpose(1, 2)
            .matmul(self.weight.view(self.out_channels, -1).t())
            .transpose(1, 2)
        )
        out = out_unf.view(batch_size, self.out_channels, x.size(2), x.size(3))
        out = out + self.bias
        return out