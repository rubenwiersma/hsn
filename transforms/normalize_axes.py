import torch
import torch.nn.functional as F


class NormalizeAxes(object):
    r"""Permutes axes so that the variances along axes are ascending, i.e. fixes up-vector.
    This assumes the shapes are aligned to one of the axes.

    Args:
        normalize_scale (bool, optional): if set to :obj:`True`, normalizes the scale of the shape
            such that the longest axis is in the range [0, 1].
            Should only be set after other precomputation steps have been performed.
    """

    def __init__(self, normalize_scale=True):
        self.normalize_scale = normalize_scale
        return


    def __call__(self, data):
        pos = data.pos

        std = torch.std(pos, dim=0)
        data.pos = data.pos[:, torch.sort(std)[1]]

        if self.normalize_scale:
            scale = 1 / (2 * data.pos.max(0).values[2])
            data.pos = data.pos * scale

        return data


    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)