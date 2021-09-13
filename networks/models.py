from networks import encoder
from networks import decoder
from networks import classifier
import torch
import torch.nn as nn
import networks.nn_utils as nnu


class FeatureNetClassifier(nn.Module):
    def __init__(self, num_classes):
        """
        Voxel Grid Classifier based on FeatureNet.        
        Expects an input tensor of size (64 x 64 x 64)
        
        Reference:
        FeatureNet: Machining feature recognition based on 3D Convolution Neural Network
        Zhibo Zhang, Prakhar Jaiswal, Rahul Rai
        Computer-Aided Design 101 (2018) 12–22

        :param num_classes: Number of classes (dimension of output layer)
        """
        super(FeatureNetClassifier, self).__init__()
        self.encoder = encoder.FeatureNetEncoder(128)
        self.classifier = classifier.LinearClassifier(128, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PointNetClassifier, self).__init__()
        self.encoder = encoder.PointNetEncoder(emb_dims=1024)
        self.fc1 = nnu.fc(1024, 512)
        self.fc2 = nnu.fc(512, 256)
        self.dp1 = nn.Dropout(0.3)
        self.dp2 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        return x


class SVGVAEImageClassifier(nn.Module):
    def __init__(self, num_classes):
        """
        Classifier: Image classifier from SVGVAE for 64x64 images
        """
        super(SVGVAEImageClassifier, self).__init__()
        self.encoder = encoder.SVGVAEImageEncoder()
        self.clf = classifier.NonLinearClassifier(4 * 4 * 64, num_classes)

    def forward(self, imgs):
        batch_size = imgs.size(0)
        x = self.encoder(imgs)
        x = x.view(batch_size, -1)
        out = self.clf(x)
        return out


class UVNetCurveClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels="xyz_only",
        crv_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
    ):
        """
        Classifier: UVNet curve classifier
        """
        super(UVNetCurveClassifier, self).__init__()
        self.crv_encoder = encoder.UVNetCurveEncoder(
            input_channels=input_channels, output_dims=crv_emb_dim,
        )
        self.graph_encoder = encoder.UVNetGraphEncoder(crv_emb_dim, graph_emb_dim)
        self.clf = classifier.NonLinearClassifier(graph_emb_dim, num_classes, dropout)

    def forward(self, bg, feat):
        out = self.crv_encoder(feat)
        node_emb, graph_emb = self.graph_encoder(bg, out)
        out = self.clf(graph_emb)
        return out


class UVNetSolidClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels="xyz_only",
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
    ):
        """
        Classifier: UVNet solid classifier
        """
        super(UVNetSolidClassifier, self).__init__()
        self.surf_encoder = encoder.UVNetSurfaceEncoder(input_channels=input_channels,)
        self.graph_encoder = encoder.UVNetGraphEncoder(srf_emb_dim, graph_emb_dim)
        self.clf = classifier.NonLinearClassifier(graph_emb_dim, num_classes, dropout)

    def forward(self, bg, feat):
        out = self.surf_encoder(feat)
        node_emb, graph_emb = self.graph_encoder(bg, out)
        out = self.clf(graph_emb)
        return out


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


class UVNetCurve2ImageAutoEnc(nn.Module):
    def __init__(
        self,
        input_channels="xyz_only",
        crv_emb_dim=64,
        graph_emb_dim=128,
        ae_latent_dim=1024,
    ):
        """
        Autoencoder: UVNet curve encoder + image decoder
        :param input_channels: Channels to consider in UV-grid. One of ("xyz_only", "xyz_normals")
        :param crv_emb_dim: Embedding dimension for each curve. This will be pooled element wise to get embedding for curve-network with same dimension.
        :param graph_emb_dim: Embedding dimension for the edge-adj graph of the curve-network
        :param ae_latent_dim: Dimension of the autoencoder latent vector
        """
        super(UVNetCurve2ImageAutoEnc, self).__init__()
        self.crv_encoder = encoder.UVNetCurveEncoder(
            input_channels=input_channels, output_dims=crv_emb_dim
        )
        self.graph_encoder = encoder.UVNetGraphEncoder(
            input_dim=crv_emb_dim, output_dim=graph_emb_dim
        )
        self.project = nn.Linear(graph_emb_dim, ae_latent_dim)
        self.decoder = decoder.ImageDecoder(ae_latent_dim)

    def forward(self, bg, feat):
        batch_size = feat.size(0)
        out = self.crv_encoder(feat)
        node_emb, x = self.graph_encoder(bg, out)
        embedding = self.project(x)
        x = self.decoder(embedding)
        return x, embedding


class UVNetBezierCurve2ImageAutoEnc(nn.Module):
    def __init__(
        self,
        input_channels="xyz_only",
        crv_emb_dim=128,
        graph_emb_dim=128,
        ae_latent_dim=1024,
    ):
        """
        Autoencoder: UVNet Bezier curve encoder + image decoder
        :param input_channels: Channels to consider in UV-grid. One of ("xyz_only", "xyz_normals")
        :param crv_emb_dim: Embedding dimension for each curve. This will be pooled element wise to get embedding for curve-network with same dimension.
        :param graph_emb_dim: Embedding dimension for the edge-adj graph of the curve-network
        :param ae_latent_dim: Dimension of the autoencoder latent vector
        """
        super(UVNetBezierCurve2ImageAutoEnc, self).__init__()
        self.crv_encoder = encoder.UVNetCurveEncoder(
            input_channels=input_channels, output_dims=crv_emb_dim
        )
        self.graph_encoder = encoder._get_graph_pooling_layer("sum")
        self.project = nn.Linear(graph_emb_dim, ae_latent_dim)
        self.decoder = decoder.ImageDecoder(ae_latent_dim)

    def forward(self, bg, feat):
        batch_size = feat.size(0)
        out = self.crv_encoder(feat)
        x = self.graph_encoder(bg, out)
        embedding = self.project(x)
        x = self.decoder(embedding)
        return x, embedding


class SVGVAEImage2ImageAutoEnc(nn.Module):
    def __init__(
        self, ae_latent_dim=1024,
    ):
        """
        Autoencoder: Image encoder + image decoder based on SVGVAE
        :param ae_latent_dim: Dimension of the autoencoder latent vector
        """
        super(SVGVAEImage2ImageAutoEnc, self).__init__()
        self.img_encoder = encoder.SVGVAEImageEncoder()
        self.project = nn.Linear(1024, ae_latent_dim)
        self.decoder = decoder.ImageDecoder(ae_latent_dim)

    def forward(self, imgs):
        batch_size = imgs.size(0)
        x = self.image_encoder(imgs)
        embedding = self.project(x)
        x = self.decoder(embedding)
        return x, embedding
