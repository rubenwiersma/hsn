import os.path as osp
import torch
import numpy as np
import vectorheat as vh

from torch_sparse import coalesce
from torch_geometric.nn import radius, fps
from torch_geometric.utils import to_undirected


class MultiscaleRadiusGraph(object):
    r"""Creates a radius graph for multiple pooling levels.
    The nodes and adjacency matrix for each pooling level can be accessed by masking
    tensors with values for nodes and edges with data.node_mask and data.edge_mask, respectively.

    Edges can belong to multiple levels,
    therefore we store the membership of an edge for a certain level with a bitmask:
        - The bit at position 2 * n corresponds to the edges used for pooling to level n
        - The bit at position 2 * n + 1 corresponds to the edges used for convolution in level n

    To find out if an edge belongs to a level, use a bitwise AND:
        `edge_mask & (0b1 << lvl) > 0`

    Args:
        ratios (list): the ratios for downsampling at each pooling layer.
        radii (list): the radius of the kernel support for each scale.
        max_neighbours (int, optional): the maximum number of neighbors per vertex,
            important to set higher than the expected number of neighbors.
        sample_n (int, optional): if provided, constructs a graph for only sample_n vertices.
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        cache_file (string, optional): if set, cache the precomputation
            in the given file and reuse for every following shape.
    """

    def __init__(self, ratios, radii, max_neighbours=512, sample_n=None, loop=False, flow='source_to_target', cache_file=None):
        assert len(ratios) == len(radii)
        self.ratios = ratios
        self.radii = radii
        self.max_neighbours = max_neighbours
        self.sample_n = sample_n
        self.loop = loop
        self.flow = flow

        # Cache can be used for datasets where shape is equal for every sample (like a sphere)
        self.get_cache = cache_file is not None and osp.exists(cache_file)
        self.save_cache = cache_file is not None and not self.get_cache
        self.cache_file = cache_file

        if self.get_cache:
            self.edge_index, self.node_mask, self.edge_mask = torch.load(cache_file)


    def __call__(self, data):
        if self.get_cache:
            data.edge_index, data.node_mask, data.edge_mask = self.edge_index, self.node_mask, self.edge_mask
            return data

        data.edge_attr = None
        batch = data.batch if 'batch' in data else None
        pos = data.pos

        # Create empty tensors to store edge indices and masks
        edge_index = torch.LongTensor()
        edge_mask = torch.LongTensor()
        node_mask = torch.zeros(data.num_nodes)
        
        # Sample points on the surface using farthest point sampling if sample_n is given
        if self.sample_n is not None and not self.sample_n > data.pos.size(0):
            sample_idx = fps(pos, batch, ratio=self.sample_n / data.pos.size(0)).sort()[0]
        else:
            sample_idx = torch.arange(data.num_nodes)
        data.sample_idx = sample_idx

        original_idx = torch.arange(sample_idx.size(0))
        pos = pos[sample_idx]
        batch = batch[sample_idx] if batch is not None else None
        for i, r in enumerate(self.ratios):
            # POOLING EDGES
            # Sample a number of points given by ratio r
            # and create edges to sampled points from nearest neighbors
            if r == 1:
                pool_idx = original_idx
            else:
                pool_idx = fps(pos, batch, r).sort()[0]

                # Use heat method to find neighbourhoods of sampled points
                pos_vh, face_vh = data.pos.cpu().numpy(), data.face.cpu().numpy().T
                idx_vh, labels_vh = sample_idx[pool_idx].cpu().numpy(), np.arange(pool_idx.size(0))
                pool_neigh = torch.from_numpy(vh.nearest(pos_vh, face_vh, idx_vh, labels_vh)).round().long().clamp(0, pool_idx.size(0) - 1).view(-1)
                
                # Add edges for pooling
                edge_index = torch.cat((edge_index, torch.stack((original_idx[pool_idx][pool_neigh][sample_idx], original_idx), dim=0)), dim=1)
                # Add corresponding mask entries for pooling edges
                edge_mask = torch.cat((edge_mask, (torch.ones(sample_idx.size(0)) * (0b1 << (i * 2))).long()))

            # Sample nodes
            sample_idx, original_idx, pos, batch = sample_idx[pool_idx], original_idx[pool_idx], pos[pool_idx], batch[pool_idx] if batch is not None else None
            node_mask[sample_idx] = i

            # CONVOLUTION EDGES
            # Create a radius graph for pooled points
            radius_edges = radius(pos, pos, self.radii[i], batch, batch, self.max_neighbours)
            radius_edges = original_idx[radius_edges]
            edge_index = torch.cat((edge_index, radius_edges), dim=1)
            edge_mask = torch.cat((edge_mask, (torch.ones(radius_edges.size(1)) * (0b1 << (i * 2 + 1))).long()))

        # Sort edges and combine duplicates with an add (bitwise OR) operation
        edge_index, edge_mask = coalesce(edge_index, edge_mask, data.num_nodes, data.num_nodes, 'add')

        # Store in data object
        data.edge_index = edge_index
        data.node_mask = node_mask
        data.edge_mask = edge_mask

        if self.save_cache:
            with open(self.cache_file, 'wb') as f:
                torch.save((edge_index, node_mask, edge_mask), f)
            self.save_cache = False
            self.get_cache = True
            self.edge_index, self.node_mask, self.edge_mask = edge_index, node_mask, edge_mask

        return data

    def __repr__(self):
        return '{}(radii={}, ratios={}, sample_n={})'.format(self.__class__.__name__, self.radii, self.ratios, self.sample_n)
