import os.path as osp
from math import pi as PI

import torch

from torch_scatter import scatter_add
from torch_geometric.utils import degree

from utils.harmonic import linear_interpolation_weights

class HarmonicPrecomp(object):
    r"""Precomputation for Harmonic Surface Networks.
    Asserts that a logmap and vertex weights have been computed
    and stored in data.edge_attr, data.weight.

    .. math::
        w_j \mu_{\q}(r_{ij}) e^{\i m\theta_{ij}}\right

    Args:
        n_rings (int, optional): number of rings used to parametrize
            the radial profile, defaults to 2.
        max_order (int, optional): the maximum rotation order of the network,
            defaults to 1.
        max_r (float, optional): the radius of the kernel,
            if not supplied, maximum radius is used.
        cache_file (string, optional): if set, cache the precomputation
            in the given file and reuse for every following shape.
    """

    def __init__(self, n_rings=2, max_order=1, max_r=None, cache_file=None):
        self.n_rings = n_rings
        self.max_order = max_order
        self.max_r = max_r

        # Cache can be used for datasets where shape is equal for every sample (like a sphere)
        self.get_cache = cache_file is not None and osp.exists(cache_file)
        self.save_cache = cache_file is not None and not self.get_cache
        self.cache_file = cache_file
        
        if self.get_cache:
            self.precomp = torch.load(cache_file)


    def __call__(self, data):
        assert hasattr(data, 'edge_attr')
        assert hasattr(data, 'weight')

        if self.get_cache:
            n_edges = data.edge_index.size(1)
            data.precomp = self.precomp[:n_edges]
            return data

        # Conveniently name variables
        (row, col), pseudo, weight = data.edge_index, data.edge_attr, data.weight
        N, M, R = row.size(0), self.max_order + 1, self.n_rings
        r, theta = pseudo[:, 0], pseudo[:, 1]

        # Normalize radius to range [0, 1]
        r =  r / self.max_r if self.max_r is not None else r / r.max()

        # Compute interpolation weights for the radial profile function
        radial_profile_weights = linear_interpolation_weights(r, R, zero_falloff=False) # [N, R]
            
        # Compute exponential component for each point
        angles = theta.view(-1, 1) * torch.arange(M).float().view(1, -1).to(theta.device) * 2 * PI
        exponential = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1) # [N, M, 2]

        # Set kernel for center points to 0
        # Why this is necessary: for the center point, r = 0 and theta = 0
        # Thus, the kernel value will always be 1 + i0, pointing to the - arbitrary - choice of basis
        exponential[torch.nonzero(r == 0), :1] = 0

        # Finally, normalize weighting for every neighborhood and multiply with precomputation
        weight = weight / (1e-12 + scatter_add(weight, row)[row]) # [N]

        # Combine precomputation components
        precomp = weight.view(N, 1, 1, 1) * radial_profile_weights.view(N, 1, R, 1) * exponential.view(N, M, 1, 2)

        data.precomp = precomp # [N M, R, 2]

        if self.save_cache:
            with open(self.cache_file, 'wb') as f:
                torch.save((precomp), f)
            self.save_cache = False
            self.get_cache = True
            self.precomp = precomp

        return data

    def __repr__(self):
        return '{}(n_rings={}, max_order={}, max_r={})'.format(self.__class__.__name__,
                                                  self.n_rings, self.max_order, self.max_r)