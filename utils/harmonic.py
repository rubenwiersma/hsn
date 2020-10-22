import torch
import torch.nn.functional as F
import numpy as np
from math import pi as PI
from torch_scatter import scatter_add, scatter_min
from scipy.spatial import Delaunay

EPS = 1e-12

### Network architecture ###
def mask_idx(lvl, edge_mask):
    """
    Converts a multi-scale edge_mask to a list of indices of edges corresponding to scale lvl.
    :param lvl: the desired scale lvl.
    :param edge_mask: a multi-scale edge mask.
    """
    mask = torch.nonzero(edge_mask & (0b1 << lvl), as_tuple=False).flatten()
    return mask


### Complex functions ###
def complex_product(a_re, a_im, b_re, b_im):
    """
    Computes the complex product of a and b, given the real and imaginary components of both.
    :param a_re: real component of a
    :param a_im: imaginary component of a
    :param b_re: real component of a
    :param b_im: imaginary component of a
    :return: tuple of real and imaginary components of result
    """
    a_re_ = a_re * b_re - a_im * b_im
    a_im = a_re * b_im + a_im * b_re
    return a_re_, a_im


### Complex non-linearities ###
def c_nonlin(x, fnc):
    """
    Applies the given non-linear function to the magnitudes of x
    :param x: [..., 2] tensor with complex activations.
    :param fnc: the non-linear function to apply.
    :param eps: offset to add, to overcome zero gradient.
    """
    magnitude = magnitudes(x)
    c = torch.div(fnc(magnitude), magnitude)
    return c * x


def c_batch_norm(x, batch_size, bn, fnc=F.elu):
    """
    Applies the given batch norm and non-linear function to the magnitudes of data.x
    :param data: batch Data object.
    :param bn: the BatchNorm1D object that applies batch norm.
    :param fnc: the non-linear function to apply.
    :param eps: offset to add, to overcome zero gradient.
    """
    # Return magnitudes for each complex number
    magnitude = magnitudes(x, keepdim=False)
    
    # Put magnitudes in correct shape for batch normalization module
    sh = magnitude.size()
    magnitude_bn = magnitude.reshape(batch_size, -1, sh[1] * sh[2]).permute(0, 2, 1)
    rb = bn(magnitude_bn)
    
    # Return shape to original and multiply result with complex input
    rb = rb.permute(0, 2, 1).reshape(sh)
    c = torch.div(fnc(rb), magnitude).unsqueeze(-1)
    return c*x


def magnitudes(x, eps=EPS, keepdim=True):
    """
    Computes the magnitudes of complex activations.
    :param x: the complex activations.
    :param eps: offset to add, to overcome zero gradient.
    :param keepdim: whether to keep the dimensions of the input.
    """
    r = torch.sum(x * x, dim=-1, keepdim=keepdim)
    eps = torch.ones_like(r) * eps
    return torch.sqrt(torch.max(r, eps))


### Interpolation ###
def linear_interpolation_weights(x, n_points, zero_falloff=False):
    """
    Compute linear interpolation weights
    to points at x from regularly interspersed points.
    :param x: coordinates of points to interpolate to, in range [0, 1].
    :param n_points: number of regularly interspersed points.
    :param zero_falloff: if set to True, the interpolated function falls to 0 at x = 1.
    """
    assert(x.dim() == 1)
    if zero_falloff:
        n_points += 1
    x = x * (n_points - 1)
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    x0 = torch.clamp(x0, 0, n_points - 2)
    x1 = torch.clamp(x1, 1, n_points - 1)
    
    w0 = x1.float() - x
    w1 = x - x0.float()

    weights = torch.zeros((x.size(0), n_points), device=x.device)
    weights[torch.arange(x.size(0)), x0] = w0
    weights[torch.arange(x.size(0)), x1] = w1

    if zero_falloff:
        weights = weights[:, :-1]

    return weights


### Dataset processing ###
def edge_to_vertex_labels(faces, labels, n_nodes):
    """
    Converts a set of labels for edges to labels for vertices.
    :param faces: face indices of mesh
    :param labels: labels for edges
    :param n_nodes: number of nodes to map to
    """
    edge2key = set()
    edge_index = torch.LongTensor(0, 2)
    for face in faces.transpose(0, 1):
        edges = torch.stack([face[:2], face[1:], face[::2]], dim=0)
        for idx, edge in enumerate(edges):
            edge = edge.sort().values
            edges[idx] = edge
            if tuple(edge.tolist()) not in edge2key:
                edge2key.add(tuple(edge.tolist()))
                edge_index = torch.cat((edge_index, edge.view(1, -1)))

    res = torch.LongTensor(n_nodes).fill_(0)
    res[edge_index[:, 0]] = labels
    res[edge_index[:, 1]] = labels

    return res - 1