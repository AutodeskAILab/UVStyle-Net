from networks import encoder
from networks import decoder
from networks import classifier
import torch
import torch.nn as nn
import networks.nn_utils as nnu


class UVNetSolid2PointsAutoEnc(nn.Module):
    def __init__(
            self,
            input_channels="xyz_only",
            surf_emb_dim=64,
            graph_emb_dim=128,
            ae_latent_dim=1024,
            num_points=1024,
            use_tanh=False,
    ):
        """
        Autoencoder: UVNet solid encoder + pointcloud decoder
        :param input_channels: Channels to consider in UV-grid. One of ("xyz_only", "xyz_normals")
        :param surf_emb_dim: Embedding dimension for each surface. This will be pooled element wise to get embedding for solid with same dimension.
        :param graph_emb_dim: Embedding dimension for the face-adj graph of the solid
        :param ae_latent_dim: Dimension of the autoencoder latent vector
        :num_points: Number of points to reconstruct
        """
        assert input_channels in ("xyz_only", "xyz_normals")
        super(UVNetSolid2PointsAutoEnc, self).__init__()
        self.surf_encoder = encoder.UVNetSurfaceEncoder(output_dims=surf_emb_dim)
        self.graph_encoder = encoder.UVNetGraphEncoder(surf_emb_dim, graph_emb_dim)
        self.project = nnu.fc(graph_emb_dim, ae_latent_dim)
        self.decoder = decoder.PointDecoder(input_dim=ae_latent_dim, num_points=1024)
        self.tanh = nn.Tanh()
        self.use_tanh = use_tanh

    def forward(self, bg, feat):
        out = self.surf_encoder(feat)
        bg = bg.to('cuda:0')
        node_emb, graph_emb = self.graph_encoder(bg, out)
        embedding = self.project(graph_emb)
        out = self.decoder(embedding)
        if self.use_tanh:
            out = self.tanh(out)
        return out, embedding


class Points2PointsAutoEnc(nn.Module):
    def __init__(self, ae_latent_dim=1024, num_out_points=1024, use_tanh=False):
        """
        Autoencoder: PointNet encoder + pointcloud decoder
        :param ae_latent_dim: Embedding dimension for the input pointcloud.
        :param num_out_points: Number of points in the output pointcloud
        """
        super(Points2PointsAutoEnc, self).__init__()
        self.encoder = encoder.PointNetEncoder(ae_latent_dim)
        self.decoder = decoder.PointDecoder(
            input_dim=ae_latent_dim, num_points=num_out_points
        )
        self.tanh = nn.Tanh()
        self.use_tanh = use_tanh

    def forward(self, pc):
        embedding = self.encoder(pc)
        out = self.decoder(embedding)
        if self.use_tanh:
            out = self.tanh(out)
        return out, embedding
