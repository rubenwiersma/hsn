import numpy as np

import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import glorot, zeros

from utils.harmonic import complex_product

class HarmonicConv(MessagePassing):
    r"""
    Harmonic Convolution from Harmonic Surface Networks

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
    """
    def __init__(self, in_channels, out_channels, max_order=1, n_rings=2, prev_order=1, 
                offset=True, separate_streams=True):
        super(HarmonicConv, self).__init__(aggr='add', flow='target_to_source', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prev_order = prev_order
        self.max_order = max_order
        self.n_rings = n_rings
        self.offset = offset
        self.separate_streams = separate_streams

        n_orders = (prev_order + 1) * (max_order + 1) if separate_streams else (max_order + 1)
        self.radial_profile = Parameter(torch.Tensor(n_orders, n_rings, out_channels, in_channels))
        
        if offset:
            self.phase_offset = Parameter(torch.Tensor(n_orders, out_channels, in_channels))
        else:
            self.register_parameter('phase_offset', None)

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.radial_profile)
        glorot(self.phase_offset)


    def forward(self, x, edge_index, precomp, connection=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        assert connection is None or connection.size(1) == 2
        assert precomp.dim() == 4

        out = self.propagate(edge_index=edge_index, x=x, precomp=precomp, connection=connection)

        return out


    def message(self, x_j, precomp, connection):
        """
        Locally aligns features with parallel transport (using connection) and
        applies the precomputed component of the circular harmonic filter to each neighbouring node (the target nodes).

        :param x_j: the feature vector of the target neighbours [n_edges, prev_order + 1, in_channels, 2]
        :param precomp: the precomputed part of harmonic networks [n_edges, max_order + 1, n_rings, 2].
        :param connection: the connection encoding parallel transport for each edge [n_edges, 2].
        :return: the message from each target to the source nodes [n_edges, n_rings, in_channels, prev_order + 1, max_order + 1, 2]
        """

        (N, M, F, C), R  = x_j.size(), self.n_rings

        # Set up result tensors
        res = torch.cuda.FloatTensor(N, R, F, M, self.max_order + 1, C).fill_(0)

        # Compute the convolutions per stream
        for input_order in range(M):
            # Fetch correct input order and reshape for matrix multiplications
            x_j_m = x_j[:, input_order, None, :, :] # [N, 1, in_channels, 2]

            # First apply parallel transport
            if connection is not None and input_order > 0:
                rot_re = connection[:, None, None, 0]
                rot_im = connection[:, None, None, 1]
                x_j_m[..., 0], x_j_m[..., 1] = complex_product(x_j_m[..., 0], x_j_m[..., 1], rot_re, rot_im)

            # Next, apply precomputed component
            for output_order in range(self.max_order + 1):
                m = output_order - input_order
                sign = np.sign(m)
                m = np.abs(m)

                # Compute product with precomputed component
                res[:, :, :, input_order, output_order, 0], res[:, :, :, input_order, output_order, 1] = complex_product(
                    x_j_m[..., 0], x_j_m[..., 1],
                    precomp[:, m, :, 0, None], sign * precomp[:, m, :, 1, None])
                
        return res


    def update(self, aggr_out):
        """
        Updates node embeddings with circular harmonic filters.
        This is done separately for each rotation order stream.
        
        :param aggr_out: the result of the aggregation operation [n_nodes, n_rings, in_channels, prev_order + 1, max_order + 1, complex]
        :return: the new feature vector for x [n_nodes, max_order + 1, out_channels, complex]
        """
        (N, _, F, M, _, C), O = aggr_out.size(), self.out_channels
        res = torch.cuda.FloatTensor(N, M, self.max_order + 1, O, 2).fill_(0)

        for input_order in range(M):
            for output_order in range(self.max_order + 1):
                m = np.abs(output_order - input_order)
                m_idx = input_order * (self.max_order + 1) + output_order if self.separate_streams else m

                aggr_re = aggr_out[:, :, None, :, input_order, output_order, 0] # [N, n_rings, 1, in_channels]
                aggr_im = aggr_out[:, :, None, :, input_order, output_order, 1] # [N, n_rings, 1, in_channels]

                # Apply the radial profile
                aggr_re = (self.radial_profile[m_idx] * aggr_re).sum(dim=1) # [N, out_channels, in_channels]
                aggr_im = (self.radial_profile[m_idx] * aggr_im).sum(dim=1) # [N, out_channels, in_channels]

                # Apply phase offset
                if self.offset:
                    cos = torch.cos(self.phase_offset[m_idx]) # [out_channels, in_channels]
                    sin = torch.sin(self.phase_offset[m_idx]) # [out_channels, in_channels]
                    aggr_re, aggr_im = complex_product(aggr_re, aggr_im, cos, sin)
                
                # Store per rotation stream
                res[:, input_order, output_order, :, 0] = aggr_re.sum(dim=-1)
                res[:, input_order, output_order, :, 1] = aggr_im.sum(dim=-1)

        # The input streams are summed together to retrieve one value per output stream
        return res.sum(dim=1)