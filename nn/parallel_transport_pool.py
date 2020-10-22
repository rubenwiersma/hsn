import numpy as np

import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add

from utils.harmonic import complex_product
from transforms import ScaleMask

class ParallelTransportPool(MessagePassing):
    r"""
    Pooling layer with parallel transport

    Args:
        lvl (int): scale to pool to
        transform (obj): transform to apply to the pooled data
    """
    def __init__(self, lvl, transform=None):
        super(ParallelTransportPool, self).__init__(aggr='mean', flow='target_to_source', node_dim=0)
        self.lvl = lvl
        self.scale_mask = ScaleMask(lvl, True)
        self.transform = transform


    def forward(self, x, data):
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        # Mask the edge indices and connection tensor for pooling
        pooling_data = self.scale_mask(data)
        edge_index, connection = pooling_data.edge_index, pooling_data.connection
        node_idx = torch.nonzero(data.node_mask >= self.lvl).view(-1)

        # Apply parallel transport and aggregate
        out = self.propagate(edge_index=edge_index, x=x, connection=connection)

        # Sum weights for later layers
        data.node_weight = scatter_add(pooling_data.weight, edge_index[0])[node_idx]

        # Apply transform to data, return as new tensor,
        # so original multiscale graph is still accessible
        data_pooled = self.transform(data)

        # Store pooling edges, nodes and parallel transport for unpooling
        if not hasattr(data, 'unpool_nodes'):
            data.unpool_edges, data.unpool_nodes, data.unpool_connection = [], [], []
        data.unpool_edges.append(edge_index)
        data.unpool_nodes.append(node_idx)
        data.unpool_connection.append(connection)

        return out[node_idx], data, data_pooled


    def message(self, x_j, connection):
        """
        Applies connection to each neighbour, before aggregating for pooling.
        """

        # Apply parallel transport to features from stream 1
        if (x_j.size(1) > 1):
            x_j[:, 1, :, 0], x_j[:, 1, :, 1] = complex_product(x_j[:, 1, :, 0], x_j[:, 1, :, 1], connection[:, None, 0], connection[:, None, 1])

        return x_j