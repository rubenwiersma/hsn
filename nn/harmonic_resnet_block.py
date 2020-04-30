import torch
import torch.nn as nn
import torch.nn.functional as F

from nn import HarmonicConv, ComplexNonLin, ComplexLin
from utils.harmonic import c_batch_norm

class HarmonicResNetBlock(torch.nn.Module):
    r"""
    ResNet block with convolutions, linearities, and non-linearities
    as described in Harmonic Surface Networks

    Args:
        in_channels (int): number of input features
        out_channels (int): number of output features
        prev_order (int, optional): the maximum rotation order of the previous layer,
            should be set to 0 if this is the first layer (default: :obj:`1`)
        max_order (int, optionatl): the maximum rotation order of this convolution
            will convolve with every rotation order up to and including `max_order`,
            (default: :obj:`1`)
        n_rings (int, optional): the number of rings in the radial profile (default: :obj:`2`)
        offset (bool, optional): if set to :obj:`False`, does not learn an offset parameter,
            this is practical in the last layer of a network (default: :obj:`True`)
        separate_streams (bool, optional): if set to :obj:`True`, learns a radial profile
            for each convolution connecting streams, instead of only an m=0 and m=1 convolution
            (default: :obj:`True`)
        last_layer (bool, optional): if set to :obj:`True`, does not learn a phase offset
            for the last harmonic conv. (default :obj:`False`)
    """
    def __init__(self, in_channels, out_channels, max_order=1, n_rings=2, prev_order=1,
                offset=True, separate_streams=True, last_layer=False):
        super(HarmonicResNetBlock, self).__init__()
        self.prev_order = prev_order
        
        self.conv1 = HarmonicConv(in_channels, out_channels, max_order, n_rings, prev_order, offset, separate_streams)
        self.nonlin1 = ComplexNonLin(out_channels, F.relu)
        self.conv2 = HarmonicConv(out_channels, out_channels, max_order, n_rings, max_order, offset and not last_layer, separate_streams)
        
        self.project_residuals = in_channels != out_channels
        if self.project_residuals:
            self.lin = ComplexLin(in_channels, out_channels)
        self.nonlin2 = ComplexNonLin(out_channels, F.relu)


    def forward(self, x, edge_index, precomp, connection=None):
        # Apply convolutions
        if self.prev_order == 0:
            x_conv = self.nonlin1(self.conv1(x, edge_index, precomp))
        else:
            x_conv = self.nonlin1(self.conv1(x, edge_index, precomp, connection))
        x_conv = self.conv2(x_conv, edge_index, precomp, connection)
        
        # Add residual connection
        x = self.lin(x) if self.project_residuals else x
        x_conv = x_conv + x
        return self.nonlin2(x_conv)
        