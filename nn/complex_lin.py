import torch
import torch.nn as nn

from utils.harmonic import magnitudes

class ComplexLin(nn.Module):
    r"""A linear layer applied to complex feature vectors
    The result is a linear combination of the complex input features

    Args:
        in_channels (int): number of input features
        out_channels (int): number of output features
    """
    def __init__(self, in_channels, out_channels):
        super(ComplexLin, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels)


    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        sh = x.size()
        x = self.lin(x.reshape(-1, self.in_channels))
        return x.reshape(sh[0], sh[1], sh[2], self.out_channels).permute(0, 1, 3, 2)