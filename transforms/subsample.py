import torch

class Subsample(object):
    r"""Samples only the positions and descriptors that are set in data.sample_idx.
    """

    def __init__(self):
        return


    def __call__(self, data):
        assert hasattr(data, 'sample_idx')
        sample_idx = data.sample_idx
        data.pos = data.pos[sample_idx]

        if hasattr(data, 'node_mask'):
            data.node_mask = data.node_mask[sample_idx]

        if hasattr(data, 'desc'):
            data.desc = data.desc[sample_idx]

        return data


    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)