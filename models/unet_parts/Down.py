import torch.nn as nn
from . import DoubleConv
class Down(nn.Module):
    """Downscalin with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv   = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv.DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv (x)

