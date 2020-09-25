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

class ImageDecoder(nn.Module):
    def __init__(self,  emb_dims):
        super(ImageDecoder, self).__init__()
        self.de_proj = nn.Linear(emb_dims, 1024)
        
        self.deconv1 = nn.Sequential(
                        nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 8 x 8
        
        self.deconv2 = nn.Sequential(
                        nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 16 x 16
        
        
        self.deconv3 = nn.Sequential(
                        nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 16 x 16
        
        self.deconv4 = nn.Sequential(
                        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 32 x 32
        
        self.deconv5 = nn.Sequential(
                        nn.ConvTranspose2d(64, 32, 5, stride=1, padding=2),
                        nn.InstanceNorm2d(32),
                        nn.ReLU()
                     ) # 32 x 32
        self.deconv6 = nn.Sequential(
                        nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1),
                        nn.InstanceNorm2d(32),
                        nn.ReLU()
                     ) # 64 x 64
        
        self.deconv7 = nn.Sequential(
                        nn.ConvTranspose2d(32, 32, 5, stride=1, padding=2),
                        nn.InstanceNorm2d(32),
                        nn.ReLU()
                     ) # 64 x 64
        self.deconv8 = nn.ConvTranspose2d(32, 1, 5, stride=1, padding=2)
        
    def forward(self, x):
        batchsize = x.size()[0]
        x = self.de_proj(x)
        x = x.view(batchsize, 64, 4, 4)
        
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        x = self.deconv8(x)
        return x  

class InnerProductDecoder(nn.Module):
    def __init__(self, latent_dim, max_num_nodes, dropout, act=torch.sigmoid, node_feat_dim=(3, 10, 10)):
        """
        Decoder that outputs an adjacency matrix by using inner product
        of features derived from the latent vectors.
        """
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.fc = nn_utils.fc(latent_dim, max_num_nodes)
        self.act = act

    def forward(self, z):
        z = self.fc(z)
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class mlpAdj(nn.Module):
    def __init__(self, nlatent = 1024):
        """Atlas decoder"""

        super(mlpAdj, self).__init__()
        self.nlatent = nlatent
        self.conv1 = torch.nn.Conv1d(self.nlatent, self.nlatent, 1)
        self.conv2 = torch.nn.Conv1d(self.nlatent, self.nlatent//2, 1)
        self.conv3 = torch.nn.Conv1d(self.nlatent//2, self.nlatent//4, 1)
        self.conv4 = torch.nn.Conv1d(self.nlatent//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.nlatent)
        self.bn2 = torch.nn.BatchNorm1d(self.nlatent//2)
        self.bn3 = torch.nn.BatchNorm1d(self.nlatent//4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class AtlasV2_Decoder(nn.Module):
    def __init__(self, num_points, emb_dims):
        super(AtlasV2_Decoder, self).__init__()
        self.num_points = num_points
        self.nlatent = emb_dims
        self.nb_primitives = 5
        self.decoder = nn.ModuleList([mlpAdj(nlatent = 2 +self.nlatent) for i in range(0, self.nb_primitives)])

        grain = int(np.sqrt(self.num_points/self.nb_primitives)) - 1
        grain = grain*1.0
        vertices = []
        
        for i in range(0,int(grain + 1 )):
            for j in range(0,int(grain + 1 )):
                vertices.append([i/grain,j/grain])

        grids = [vertices for i in range(0,self.nb_primitives)]

        grids = torch.Tensor(grids)
        grids = grids.transpose(2,1)
        self.register_buffer('grids', grids)

    def forward(self, x):
        device = x.device
        outs = []
        for i in range(0,self.nb_primitives):
            if self.training == True:
                rand_grid = torch.FloatTensor(x.size(0), 2, self.num_points//self.nb_primitives).to(device) 
                rand_grid.data.uniform_(0, 1)
                rand_grid[:,2:,:] = 0
            else:
                grid = self.grids[i]
                rand_grid = grid.unsqueeze(0).expand(x.size(0),grid.size(0), grid.size(1))
                
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()