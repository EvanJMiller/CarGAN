import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, d):
        super(Discriminator, self).__init__()

        self.num_freature_maps = 128
        self.num_channels = 1

        # # input is (self.num_channels) x 64 x 64
        # self.deconv1 = nn.Conv2d(1, d, 4, 2, 1, bias=False)
        # #nn.LeakyReLU(0.2, inplace=True),
        # # state size. (self.num_freature_maps) x 32 x 32
        # self.deconv2 = nn.Conv2d(d, d * 2, 4, 2, 1, bias=False)
        # self.norm1 = nn.BatchNorm2d(self.num_freature_maps * 2)
        # #nn.LeakyReLU(0.2, inplace=True),
        # # state size. (self.num_freature_maps*2) x 16 x 16
        # self.deconv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1, bias=False)
        # self.norm2 = nn.BatchNorm2d(self.num_freature_maps * 4)
        # #nn.LeakyReLU(0.2, inplace=True),
        # # state size. (self.num_freature_maps*4) x 8 x 8
        # self.deconv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1, bias=False)
        # self.norm3 = nn.BatchNorm2d(d * 8)
        # #nn.LeakyReLU(0.2, inplace=True),
        # # state size. (self.num_freature_maps*8) x 4 x 4
        # self.deconv5 = nn.Conv2d(d * 8, 1, 4, 1, 0, bias=False)

        self.deconv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.deconv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.norm1 = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.norm2 = nn.BatchNorm2d(d*4)
        self.deconv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.norm3 = nn.BatchNorm2d(d*8)
        self.deconv5 = nn.Conv2d(d*8, 1, 4, 1, 0)


    def forward(self, x):

        x = F.relu(self.deconv1(x))
        x = F.relu(self.norm1(self.deconv2(x)))
        x = F.relu(self.norm2(self.deconv3(x)))
        x = F.relu(self.norm3(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x