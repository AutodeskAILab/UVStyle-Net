import torch.nn as nn

import networks.nn_utils as nnu


class PointDecoder(nn.Module):
    def __init__(self, input_dim, num_points):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nnu.fc(input_dim, 1024)
        self.fc2 = nnu.fc(1024, 2048)
        self.fc3 = nn.Linear(2048, num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(batchsize, 3, self.num_points).transpose(1, 2).contiguous()
        return x  ### B x Num_points X 3
