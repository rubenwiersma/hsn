import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.inits import zeros
from utils.harmonic import magnitudes

class ComplexNonLin(nn.Module):
    r"""Adds a learned bias and applies the nonlinearity
    given by fnc to the radial component of complex features

    Args:
        num_features (int): number of input features
        fnc (torch.nn.Module): non-linearity function
    """
    def __init__(self, in_channels, fnc=F.relu):
        super(ComplexNonLin, self).__init__()
        
        self.fnc = fnc

        self.bias = nn.Parameter(torch.Tensor(in_channels))
        zeros(self.bias)


    def forward(self, x):
        magnitude = magnitudes(x)
        rb = magnitude + self.bias.unsqueeze(-1)
        c = torch.div(self.fnc(rb), magnitude)
        return c*x