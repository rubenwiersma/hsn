# CNNs on Surfaces using Rotation-Equivariant Features

## [[Paper]](https://rubenwiersma.nl/assets/pdf/CNNs_Surfaces_Rotation_Equivariant_Features.pdf) [[Project page]](https://rubenwiersma.nl/hsn) 

Code for Harmonic Surface Networks, an approach for deep learning on surfaces operating on vector-valued, rotation-equivariant features. This is achieved by learning circular harmonic kernels and separating features in streams of different equivariance classes. The advantage of our approach is that the rotational degree of freedom, arising when a filter kernel is transported along a surface, has no effect on the network. The filters can be evaluated in arbitrarily chosen coordinate systems.

## Contents
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Replicating experiments from the paper](#replicating-experiments)

<img src="img/representative.jpg" width="80%">

## Dependencies
This project requires the following dependencies. The version numbers have been tested and shown to work, other versions are likely, but not guaranteed, to work.
- [PyTorch >= 1.5](https://pytorch.org)
- [PyTorch Geometric and its dependencies](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
  <br />The version used in the experiments is PyTorch Geometric 1.4.3
- [Progressbar2](https://pypi.org/project/progressbar2/)
- [Suitesparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) (used for the Vector Heat Method)

## Installation
Clone this repository and its submodules
```
$ git clone --recurse-submodules https://github.com/rubenwiersma/hsn.git
```

Install the `vectorheat` python module, explained in the following subsection.

### The vectorheat extension

**[Update April 2021]** Nick Sharp has created his own bindings for the Geometry Central library, called [Potpourri3d](https://github.com/nmwsharp/potpourri3d). This library includes computations of the vector heat method on meshes **and point clouds**. You can install it with pip:
```
$ pip install potpourri3d
```
The transforms used in this repository make use of our own binding, which can be installed as follows:

To perform computations on the mesh - e.g. Vector Heat Method, surface area computation, etc. - we use [Geometry Central](https://geometry-central.net). We have created a small Python binding into a C++ function that computes these quantities globally for each vertex in a mesh. Run the following commands in your shell to install the extension:
```
$ pip install ./vectorheat
```

**Having trouble building?**
First make sure that you have the latest version of CMake installed (> 3.10). Next, check that every dependency is present in this repo (pybind11, geometrycentral). If not, you might not have cloned submodules. To fix this:
```
$ git submodule update --recursive
```

**Suitesparse?** When processing shapes with the Vector Heat Method, you might run into a solver error. This is fixed when you build the `vectorheat` extension with suitesparse. Installation in Linux with:

```
$ apt-get install libsuitesparse-dev

```

## Replicating experiments
We have compiled four Jupyter Notebooks to easily replicate the experiments from the paper:
- Shape classification
- Shape segmentation
- Correspondence
- Classification of MNIST mapped to a sphere

These notebooks can be accessed in the root of this repo.

### Datasets
To use the datasets for these experiments, follow these steps:
- Create a `data` folder in a convenient place (notebooks assume the repo root).
- Download and unzip the dataset for the experiment you want to replicate and move it to the `data` folder. Note: we've zipped the folder structure for easy drag and drop. The folders contain a zip file that should not be unzipped.
    - [Shape classification (3.4MB)](https://surfdrive.surf.nl/files/index.php/s/ifhelkX4cd7ky8W) [1]
    - [Shape segmentation (74.1MB)](https://surfdrive.surf.nl/files/index.php/s/L68uSYpHtfO6dLa) [2]
    - [Correspondence (remeshed) (17.4MB)](https://surfdrive.surf.nl/files/index.php/s/KLSxAN0QEsfJuBV) [3]
        - Download the original FAUST dataset [here](http://faust.is.tue.mpg.de/) [4]
    - [Classification of MNIST mapped to a sphere (55.8MB)](https://surfdrive.surf.nl/files/index.php/s/KzE1pqfGDwBHQ16) [5]

<small>[1] Haggai Maron, Meirav Galun, Noam Aigerman, Miri Trope, Nadav Dym, Ersin Yumer,
Vladimir G Kim, and Yaron Lipman. 2017. Convolutional neural networks on surfaces
via seamless toric covers. ACM Trans. Graph 36, 4 (2017).

[2] Zhouhui Lian et al. 2011. SHREC ’11 Track: Shape
Retrieval on Non-rigid 3D Watertight Meshes. Eurographics Workshop on 3D Object
Retrieval.

[3] Adrien Poulenard and Maks Ovsjanikov. 2018. Multi-directional geodesic neural net-
works via equivariant convolution. ACM Trans. Graph. 37, 6 (2018).

[4] Federica Bogo, Javier Romero, Matthew Loper, and Michael J. Black. 2014. FAUST:
Dataset and evaluation for 3D mesh registration. In CVPR. IEEE.

[5] Hugo Larochelle et al. 2007. An empirical evaluation of deep architectures on problems with many factors
of variation. In ICML. ACM.

</small>

<hr/>

Author: [Ruben Wiersma](https://www.rubenwiersma.nl)

Please cite our paper if this code contributes to an academic publication:
```
@Article{Wiersma2020,
  author    = {Ruben Wiersma, Elmar Eisemann, Klaus Hildebrandt},
  journal   = {Transactions on Graphics},
  title     = {CNNs on Surfaces using Rotation-Equivariant Features},
  year      = {2020},
  month     = jul,
  number    = {4},
  volume    = {39},
  doi       = {10.1145/3386569.3392437},
  publisher = {ACM},
}
```
