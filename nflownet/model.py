import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import pad_to_divisible_by_4, crop_to_target_size


class ResidualBlock(nn.Module):
    """
    Adapted from EVPropNet: https://prg.cs.umd.edu/EVPropNet
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)



class TransposedResidualBlock(nn.Module):
    """
    Adapted from EVPropNet: https://prg.cs.umd.edu/EVPropNet
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


 
class NFlowNet(nn.Module):
    """
    depth: χ, how many residual/transpose blocks to repeat
    in_channels: input image channels (6 by default)                        
    base_channels: the number of filters used for the first convolution layer
    expansion_rate: factor by which the number of neurons are increased after every block
    """

    def __init__(
        self,
        depth: int = 2,
        in_channels: int = 6,
        base_channels: int = 37,    
        expansion_rate: int = 2,     
    ):
        super().__init__()
        num_channels = base_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_channels, int(num_channels*expansion_rate), kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(int(num_channels * expansion_rate)),
            nn.ReLU(inplace=True)
        )
        num_channels = int(num_channels*expansion_rate)


        bottleneck_layers = []
        for _ in range(depth):
            bottleneck_layers.append(ResidualBlock(num_channels))
            bottleneck_layers.append(nn.Conv2d(num_channels, num_channels*expansion_rate, kernel_size=3, stride=2, padding=1))
            num_channels = int(num_channels*expansion_rate)
        for _ in range(depth):
            bottleneck_layers.append(TransposedResidualBlock(num_channels))
            bottleneck_layers.append(nn.ConvTranspose2d(num_channels, int(num_channels/expansion_rate), kernel_size=3, stride=2, padding=1, output_padding=1))
            num_channels = int(num_channels/expansion_rate)

        self.bottleneck = nn.Sequential(*bottleneck_layers)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_channels, int(num_channels/expansion_rate), kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(int(num_channels/expansion_rate)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(int(num_channels/expansion_rate), int(num_channels/(expansion_rate**2)), kernel_size=7, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(int(num_channels/(expansion_rate**2))),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(int(num_channels/(expansion_rate**2)), 2, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x):
        x = self.encoder(x)
        shape_temp = x.size()
        x = pad_to_divisible_by_4(x)
        x = self.bottleneck(x)
        x = crop_to_target_size(x, shape_temp[2], shape_temp[3])
        x = self.decoder(x)
        return x
