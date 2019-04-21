
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.num_freature_maps = 64
        self.num_channels = 3

        self.main = nn.Sequential(
            # input is (self.num_channels) x 64 x 64
            nn.Conv2d(self.num_channels, self.num_freature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.num_freature_maps) x 32 x 32
            nn.Conv2d(self.num_freature_maps, self.num_freature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_freature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.num_freature_maps*2) x 16 x 16
            nn.Conv2d(self.num_freature_maps * 2, self.num_freature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_freature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.num_freature_maps*4) x 8 x 8
            nn.Conv2d(self.num_freature_maps * 4, self.num_freature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_freature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.num_freature_maps*8) x 4 x 4
            nn.Conv2d(self.num_freature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)