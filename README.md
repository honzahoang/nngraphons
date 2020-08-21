# nngraphons

This Python package implements graphon estimation using neural networks and was created as part of my master's thesis at CTU FEE.

## Getting Started

### Experiment Notebooks
The `experiment_notebooks/` directory contains `.ipynb` notebooks that showcase the usage of this Python package.

### Installation

#### Cloud-Hosted Python Enviroments
I highly recommend running the experiment notebooks and playing with the code inside 
[Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) or 
[Kaggle Kernels](https://www.kaggle.com/kernels) to have a Python enviroment ready, preinstalled with most of the needed Python packages, and also for the GPUs that they provide. This option is the most convenient one because you don't have to install anything on your local machine.

To get started, just copy the notebooks from this repository into the cloud-hosted enviroments and run them.

#### Local Python Environments
If you want to run the code on your local machine, install [Jupyter Notebook](https://jupyter.org/) optionally with [JupterLab](https://jupyter.org/install.html), install [Poetry](https://python-poetry.org/) and create an environment/kernel by running
```
poetry install
```
in the cloned repository folder. The dependencies/required packages are listed in the `pyproject.toml` file.

### Package structure
To get to know the in's and out's of the package it is best to play with the notebooks located in the `experiment_notebooks/` directory. Here is a brief overview of the package, subpackage, and modules structure:

* `nngraphons/architectures/` contains `.py` scripts with classes that implement different neural network architectures using PyTorch,
* `nngraphons/data_manipulation/` contains various scripts that manipulate graph data,
    * `nngraphons/data_manipulation/synthetic_graphons.py` contains synthetic graphons defined with Python functions
    * `nngraphons/data_manipulation/graphon_sampling.py` contains functions that sample random graphs from graphons (synthetic or neural network graphons)
    * `nngraphons/data_manipulation/small_graphs.py` contains functions that generate small graphs F with which to calculate homomorphism densities
    * `nngraphons/data_manipulation/networkx_conversion.py` contains helper functions that convert numpy representation of graphs to networkx format and back
* `nngraphons/learning/` contains scripts related to training neural networks to represent a graphon
    * `nngraphons/learning/gradient_learning.py` contains our bare-bones learning algorithm and other gradient-based learning algorithms
    * `nngraphons/learning/homomorphism_densities.py` contains functions that approximate homomorphism densities between finite graphs or neural network graphons.
* `nngraphons/visualization` contains all scripts related to the visualization of graphons, neural networks, learning process and so on.

## Literature
If you're not familiar with [PyTorch](https://pytorch.org/), which is used extensively in this package, get started with the [PyTorch 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) tutorial.

The main source for graphon theory that I use is [Lovász's book](http://web.cs.elte.hu/~lovasz/bookxx/hombook-almost.final.pdf).


## Credits

The learning algorithms in this package were developed with the help of my thesis supervisor [Ondřej Kuželka](https://www.linkedin.com/in/ondrejkuzelka/?originalSubdomain=cz).

Uses code from https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer for the pytorch RBF layer.

## Contributors

- Vu Huy Hoang

## License & copyright

© Vu Huy Hoang

Licensed under the [GNU GPLv3 License](LICENSE)
