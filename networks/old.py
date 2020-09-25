INPUT_CHANNELS = 6  # xyz for points, xyz for normals
NUM_CLASSES = 36 # 26 alphabets, 10 digits
EMBEDDING_DIM = 32


def conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels), nn.BatchNorm2d(out_channels), nn.LeakyReLU())


def weightedconv(in_channels, out_channels):
    # TODO: implement weighting using I-form
    pass


def fc(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.LeakyReLU())


class NURBSFeatureExtractor(nn.Module):
    def __init__(self, output_dims=64):
        self.conv1 = conv(INPUT_CHANNELS, 16)
        self.conv2 = conv(16, 32)
        self.conv3 = conv(32, 64)
        self.conv4 = conv(64, 32)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = fc(64, output_dims)

    def forward(self, data):
        x = data.x
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = self.adaptive_pool(x)
        x = x.flatten(1, -1)
        x = self.fc(x)
        return x


class BRepFeatureExtractor(nn.Module):
    def __init__(self):
        self.conv1 = GCNConv(64, 32)
        self.conv2 = GCNConv(32, EMBEDDING_DIM)
        

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        return x


class Classifier(nn.Module):
    def __init__(self):
        self.conv = GCNConv(EMBEDDING_DIM, NUM_CLASSES)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        return F.log_softmax(x, dim=1)
