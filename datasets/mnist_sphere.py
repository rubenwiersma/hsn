import os
import os.path as osp
import warnings
from PIL import Image

import numpy as np
import torch
import codecs
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils import grid, contains_self_loops
from torch_geometric.io import read_obj

import progressbar

class MNISTSphere(InMemoryDataset):
    r"""MNIST dataset mapped to a sphere.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(MNISTSphere, self).__init__(root, transform, pre_transform,
                                               pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['training.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        # process and save as torch files
        print('Saving as raw torch files...')
        
        train = np.load(os.path.join(self.raw_dir, 'mnist_rotation_new/rotated_train.npz'))
        valid = np.load(os.path.join(self.raw_dir, 'mnist_rotation_new/rotated_valid.npz'))
        test = np.load(os.path.join(self.raw_dir, 'mnist_rotation_new/rotated_test.npz'))
        training_set = (
            torch.from_numpy(train['x']).reshape(-1, 28, 28),
            torch.from_numpy(train['y'])
        )
        validation_set = (
            torch.from_numpy(valid['x']).reshape(-1, 28, 28),
            torch.from_numpy(valid['y'])
        )
        test_set = (
            torch.from_numpy(test['x']).reshape(-1, 28, 28),
            torch.from_numpy(test['y'])
        )

        training_set = (torch.cat((training_set[0], validation_set[0])), torch.cat((training_set[1], validation_set[1])))
        with open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.raw_dir, self.raw_file_names[1]), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def process(self):
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            x, y = torch.load(raw_path)
            x, y = x.to(torch.float), y.to(torch.long)

            sphere = read_obj('datasets/sphere_642.obj')
            face, pos = sphere.face, sphere.pos

            # Map from the sphere to the image and take value from image
            u, v = sphere_to_disc(pos)
            sq_x, sq_y = disc_to_square(u, v)
            sq_x = (sq_x + 1) / 2 * 27
            sq_y = (sq_y + 1) / 2 * 27

            # Interpolate
            x = bilinear_interpolate(x, sq_x, sq_y)

            # Store original sizes
            m, n, n_f = y.size(0), x.size(1), face.size(1)

            # Repeat for each image
            face = face.repeat(1, m)
            pos = pos.repeat(m, 1)

            # Flatten to list nodes
            x = x.view(m * n, 1)

            # Provide slices to access each graph separately
            node_slice = torch.arange(0, (m + 1) * n, step=n, dtype=torch.long)
            face_slice = torch.arange(0, (n_f + 1) * m, step=n_f, dtype=torch.long)
            graph_slice = torch.arange(m + 1, dtype=torch.long)

            self.data = Data(x=x, face=face, y=y, pos=pos)
            self.slices = {
                'x': node_slice,
                'face': face_slice,
                'y': graph_slice,
                'pos': node_slice
            }

            if self.pre_filter is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [d for d in data_list if self.pre_filter(d)]
                self.data, self.slices = self.collate(data_list)

            if self.pre_transform is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [self.pre_transform(data) for data in progressbar.progressbar(data_list)]
                self.data, self.slices = self.collate(data_list)

            torch.save((self.data, self.slices), path)


def disc_to_square(u, v):
    x = (1/2 * torch.sqrt(2 + u**2 - v**2 + 2 * np.sqrt(2) * u)
        - 1/2 * torch.sqrt(2 + u**2 - v**2 - 2 * np.sqrt(2) * u))
    y = (1/2 * torch.sqrt(2 - u**2 + v**2 + 2 * np.sqrt(2) * v)
        - 1/2 * torch.sqrt(2 - u**2 + v**2 - 2 * np.sqrt(2) * v))
    return x, y


def sphere_to_disc(pos):
    centerpoint = torch.Tensor([[0], [0], [1]])
    rho = torch.acos(pos @ centerpoint).view(-1)
    rho = torch.where(rho < 1/2 * np.pi, rho, np.pi - rho)
    rho = rho / (1/2 * np.pi)
    theta = torch.atan2(pos[:, 1], pos[:, 0])
    u = rho * torch.cos(theta)
    v = rho * torch.sin(theta)
    return u, v


def bilinear_interpolate(im, x, y):
    # Find grid indices for x and y
    x0 = x.floor()
    y0 = y.floor()
    x1 = x0 + 1
    y1 = y0 + 1

    # Make sure indices are within grid size
    x0 = x0.clamp(0, im.size(1) - 2).long()
    y0 = y0.clamp(0, im.size(2) - 2).long()
    x1 = x1.clamp(1, im.size(1) - 1).long()
    y1 = y1.clamp(1, im.size(2) - 1).long()
    
    Ia = im[:, x0, y0]
    Ib = im[:, x1, y0]
    Ic = im[:, x0, y1]
    Id = im[:, x1, y1]
    
    wa = ((x1.float() - x) * (y1.float() - y)).view(1, -1)
    wb = ((x1.float() - x) * (y - y0.float())).view(1, -1)
    wc = ((x - x0.float()) * (y1.float() - y)).view(1, -1)
    wd = ((x - x0.float()) * (y - y0.float())).view(1, -1)

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)