import numpy as np

import torch
from torch.nn import Parameter, Module
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import glorot, zeros

from utils.harmonic import complex_product, mask_idx
from transforms import ScaleMask

class ParallelTransportUnpool(Module):
    """
    Unpool layer with parallel transport

    Args:
        from_lvl (int): 
    """
    def __init__(self, from_lvl):
        super(ParallelTransportUnpool, self).__init__()
        self.from_lvl = from_lvl


    def forward(self, x, data):
        # Remove edges used for pooling from stack
        unpool_nodes, unpool_edges, unpool_connection = data.unpool_nodes[-1], data.unpool_edges[-1], data.unpool_connection[-1]
        if len(data.unpool_nodes) > 1:
            data.unpool_nodes, data.unpool_edges, data.unpool_connection = data.unpool_nodes[:-1], data.unpool_edges[:-1], data.unpool_connection[:-1]

        # Create a mapping from edge indices to pooled node indices
        unpool_map = torch.zeros(data.num_nodes).long()
        unpool_map[unpool_nodes] = torch.arange(unpool_nodes.size(0))

        # Assign values of pooled nodes to nearest nodes
        x_sh = [data.num_nodes] + list(x.size()[1:])
        new_x = torch.zeros(x_sh).to(x.device)
        new_x[unpool_edges[1]] = x[unpool_map[unpool_edges[0]]]
        x = new_x

        # Apply parallel transport to correct rotation
        connection = unpool_connection[unpool_edges[1].argsort()]
        if x.size(1) > 1:
            x[:, 1, :, 0], x[:, 1, :, 1] = complex_product(x[:, 1, :, 0], x[:, 1, :, 1], connection[:, None, 0], -connection[:, None, 1])
        return x