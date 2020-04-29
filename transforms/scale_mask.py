import torch
from math import pi as PI
from utils.harmonic import mask_idx
from torch_geometric.data import Data

class ScaleMask(object):
    r"""Masks the nodes, edges and edge attributes for a given scale level.

    Args:
        lvl (int): the scale level, starting at 0.
        pooling (boolean): if set to :obj:`True`, masks edges for pooling to scale `lvl`.
            if :obj:`False`, masks edges to be used in scale `lvl`.
    """

    def __init__(self, lvl, pooling=False):
        self.node_lvl = lvl
        self.edge_lvl = (lvl * 2) + (not pooling)
        self.pooling = pooling


    def __call__(self, data):
        # Store a copy of the original node mask, used to map edge indices
        if not hasattr(data, 'node_mask_'):
            data.node_mask_ = data.node_mask

        # Compute mask for edges
        mask = mask_idx(self.edge_lvl, data.edge_mask) # mask for edges in original graph
        n_nodes = data.num_nodes # n_nodes in original graph

        # Mask nodes
        node_idx = torch.nonzero(data.node_mask >= self.node_lvl).view(-1)
        if (not self.pooling):
            # Also pool node_mask
            data.node_mask = data.node_mask[node_idx]
            data.batch = data.batch[node_idx]

        # Mask edges
        edge_index = data.edge_index[:, mask]

        # Map edges to node edges in new graph
        node_idx_ = torch.nonzero(data.node_mask_ >= (self.node_lvl - self.pooling)).view(-1)
        idx_ = torch.cuda.LongTensor(n_nodes).fill_(0)
        idx_[node_idx_] = torch.arange(node_idx_.size(0)).to(idx_.device)
        edge_index = idx_[edge_index]

        # Store in new data object
        res = Data(edge_index=edge_index)
        if hasattr(data, 'connection'):
            res.edge_attr, res.connection = data.edge_attr[mask], data.connection[mask]
            res.weight = data.node_weight[res.edge_index[1]] if hasattr(data, 'node_weight') else data.weight[mask]
        return res
