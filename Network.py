import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.features = nn.Sequential(

            # Convoltion 1
            nn.Conv2d(3, 128, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Convolution 2
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Convolution 3
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(p=0.5),
            nn.Linear(576, 4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 196)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = F.softmax(x)
        return x
