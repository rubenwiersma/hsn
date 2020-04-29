# Harmonic Surface Networks
Code for [CNNs on Surfaces using Rotation-Equivariant Features](https://doi.org/10.1145/3386569.3392437).

Jump to:
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Replicating experiments from the paper](#replicating-experiments)

<img src="img/representative.jpg" width="50%">

## Abstract
This paper is concerned with a fundamental problem in geometric deep learning that arises in the construction of convolutional neural networks on surfaces. Due to curvature, the transport of filter kernels on surfaces results in a rotational ambiguity, which prevents a uniform alignment of these kernels on the surface. We propose a network architecture for surfaces that consists of vector-valued, rotation-equivariant features. The equivariance property makes it possible to locally align features, which were computed in arbitrary coordinate systems, when aggregating features in a convolution layer. The resulting network is agnostic to the choices of coordinate systems for the tangent spaces on the surface. We implement our approach for triangle meshes. Based on circular harmonic functions, we introduce convolution filters for meshes that are rotation-equivariant at the discrete level. We evaluate the resulting networks on shape correspondence and shape classifications tasks and compare their performance to other approaches.

## Dependencies
This project requires the following dependencies. The version numbers have been tested and shown to work.
- [PyTorch 1.5](https://pytorch.org)
- [PyTorch Geometric 1.4.3 and its dependencies](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Progressbar2](https://pypi.org/project/progressbar2/)

## Installation
Clone this repository and its submodules
```
$ git clone --recurse-submodules https://github.com/rubenwiersma/hsn.git
```

Install the `vectorheat` python module, explained in the following subsection.

### The vectorheat extension
To compute perform operations on the mesh - e.g. Vector Heat Method, surface area computation, etc. - we use [Geometry Central](https://geometry-central.net). We have created a small Python binding into a C++ function that computes these quantities globally for each vertex in a mesh. Run the following commands in your shell to install the extension:
```
$ pip install ./vectorheat
```

#### Having problems?
First make sure that you have the latest version of CMake installed (> 3.10). Next, check that every dependency is present in this repo (pybind11, geometrycentral). If not, you might not have cloned submodules. To fix this:
```
$ git submodule update --recursive
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
- Create a `data` folder in a convenient place (notebooks assume in the repo root).
- Download and unzip the dataset you want to replicate and move it to the `data` folder.
    - [Shape classification](https://surfdrive.surf.nl/files/index.php/s/ifhelkX4cd7ky8W)
    - [Shape segmentation](https://surfdrive.surf.nl/files/index.php/s/L68uSYpHtfO6dLa)
    - [Correspondence](https://surfdrive.surf.nl/files/index.php/s/dS6upV07js2nVjR)
    - [Classification of MNIST mapped to a sphere](https://surfdrive.surf.nl/files/index.php/s/KzE1pqfGDwBHQ16)

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
