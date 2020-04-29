import torch

class FilterNeighbours(object):
    r"""Filters the adjacency matrix and discards neighbours
    that are farther away than the given radius.

    Args:
        radius (float): neighbours have to be within this radius to be maintained.
    """

    def __init__(self, radius):
        self.radius = radius
        return


    def __call__(self, data):
        mask = torch.nonzero(data.edge_attr[:, 0] <= self.radius)[:, 0]
        data.edge_index = data.edge_index[:, mask]
        data.edge_attr = data.edge_attr[mask]
        if hasattr(data, 'precomp'):
            data.precomp = data.precomp[mask]
        if hasattr(data, 'weight'):
            data.weight = data.weight[mask]
        if hasattr(data, 'connection'):
            data.connection = data.connection[mask]
        return data
