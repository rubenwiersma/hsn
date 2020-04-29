import torch
import numpy as numpy

def write_ply(file, data, pred, features):
    """ 
    Creates a ply file with the mesh,
    given predictions on each vertex and features from inside the network
    Can be used to visualize features in other software

    :param file: file name to write to
    :param data: mesh object with positions and faces in `pos` and `face`
    :param pred: predictions for each vertex, size [n_vert]
    :param features: complex features for each vertex, size [n_vert, 2]
    """
    with open(file, 'w') as f:
        f.write(
            'ply\n'\
            + 'format ascii 1.0\n'\
            + 'element vertex {}\n'.format(data.pos.size(0))\
            + 'property float x\n'\
            + 'property float y\n'\
            + 'property float z\n'\
            + 'property float fx\n'\
            + 'property float fy\n'\
            + 'property float seg\n'\
            + 'element face {}\n'.format(data.face.size(1))\
            + 'property list uchar uint vertex_indices\n'\
            + 'end_header\n'
            )
        for i in range(data.pos.size(0)):
            f.write('{:1.6f} {:1.6f} {:1.6f} {:1.6f} {:1.6f} {}\n'.format(data.pos[i, 0], data.pos[i, 1], data.pos[i, 2], features[i, 0], features[i, 1], pred[i]))
        for face in data.face.transpose(0, 1):
            f.write('3 {} {} {}\n'.format(face[0], face[1], face[2]))
        
    return