import torch.nn as nn


class Image_Encoder(nn.Module):
    def __init__(self, args):
   
        super(Image_Encoder, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 32, 5, stride=1, padding=2),
                        nn.InstanceNorm2d(32),
                        nn.ReLU()
                     ) # 64 x 64
        self.conv2 = nn.Sequential(
                        nn.Conv2d(32, 32, 5, stride=2, padding=2),
                        nn.InstanceNorm2d(32),
                        nn.ReLU()
                     ) # 32 x 32
        self.conv3 = nn.Sequential(
                        nn.Conv2d(32, 64, 5, stride=1, padding=2),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 32 x 32
        self.conv4 = nn.Sequential(
                        nn.Conv2d(64, 64, 5, stride=2, padding=2),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 16 x 16
        
        self.conv5 = nn.Sequential(
                        nn.Conv2d(64, 64, 4, stride=2, padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 8 x 8
        self.conv6 = nn.Sequential(
                        nn.Conv2d(64, 64, 4, stride=2, padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 4 x 4
        
      
    def forward(self, imgs):
        batch_size = imgs.size(0)
        x = self.conv1(imgs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(batch_size,-1)
        return x # returns b x 1024 --> same as david ha paper 

