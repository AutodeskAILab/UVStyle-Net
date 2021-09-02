import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networks.nn_utils as nnu


def get_decoder_model(decoder_type, num_points, latent_dim):
    if decoder_type == "point_decoder":
        return PointDecoder(num_points, latent_dim)
    if decoder_type == "graph_decoder":
        return PointDecoder(num_points, latent_dim)
    if decoder_type == "atlas_decoder":
        return AtlasV2_Decoder(num_points, latent_dim)
    raise ValueError("Decoder {} not found, expected one of ('PointDecoder',)".format(decoder_type))


class PointDecoder(nn.Module):
    def __init__(self, num_points, emb_dims):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = emb_dims
        self.fc1 = nnu.fc(emb_dims, 1024)
        self.fc2 = nnu.fc(1024, 2048)
        self.fc3 = nn.Linear(2048, num_points * 3 )
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(batchsize, 3, self.num_points).transpose(1,2).contiguous()
        return x ### B x Num_points X 3 
