import torch
import vectorheat as vh
import numpy as np

class NormalizeArea(object):
    r"""Centers shapes and normalizes their surface area.
    """

    def __init__(self):
        return


    def __call__(self, data):
        # Center shapes
        data.pos = data.pos - (torch.max(data.pos, dim=0)[0] + torch.min(data.pos, dim=0)[0]) / 2
        
        # Normalize by surface area
        pos_vh, face_vh = data.pos.cpu().numpy(), data.face.cpu().numpy().T
        area = 1 / np.sqrt(vh.surface_area(pos_vh, face_vh))
        data.pos = data.pos * area

        return data


    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)