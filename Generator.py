# Generator Code
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()

        self.latent_z_vector = 100
        self.num_feature_maps = 64
        self.num_channels = 3
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.latent_z_vector, self.num_feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.num_feature_maps * 8),
            nn.ReLU(True),
            # state size. (self.num_feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(self.num_feature_maps * 8, self.num_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_feature_maps * 4),
            nn.ReLU(True),
            # state size. (self.num_feature_maps*4) x 8 x 8
            nn.ConvTranspose2d( self.num_feature_maps * 4, self.num_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_feature_maps * 2),
            nn.ReLU(True),
            # state size. (self.num_feature_maps*2) x 16 x 16
            nn.ConvTranspose2d( self.num_feature_maps * 2, self.num_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_feature_maps),
            nn.ReLU(True),
            # state size. (self.num_feature_maps) x 32 x 32
            nn.ConvTranspose2d( self.num_feature_maps, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (self.num_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)