![PsiZ logo](docs/img/full_logo_300.png)

[![PyPI version](https://badge.fury.io/py/psiz.svg)](https://badge.fury.io/py/psiz)
[![Python](https://img.shields.io/pypi/pyversions/psiz.svg?style=plastic)](https://badge.fury.io/py/psiz)
[![Documentation Status](https://readthedocs.org/projects/psiz/badge/?version=latest)](https://psiz.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/roads/psiz/branch/main/graph/badge.svg?token=UIK748KI5I)](https://codecov.io/gh/roads/psiz)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---
**WARNING:** This package is pre-release and the API is not stable. All APIs are subject to change and all releases are alpha.

---

## Purpose

PsiZ provides computational tools for modeling how people perceive the world. The primary use case of PsiZ is to infer psychological representations from human behavior (e.g., similarity judgments). The package integrates cognitive theory with modern computational methods.

## Resources
* Official Psiz Documentation: [psiz.readthedocs.io/en/latest](https://psiz.readthedocs.io/en/latest/)
* [PsiZ Examples](examples/)

## What's in a name?

The name PsiZ (pronounced like the word *size*, /sʌɪz/) is meant to serve as shorthand for the term *psychological embedding*. The greek letter Psi is often used to denote the field of psychology and the matrix variable **Z** is often used in machine learning to denote a latent feature space.

## Installation

There are two different ways to install: PyPI or git. Installing via git has the advantage of including examples and tests in the cloned repository.

### Using PyPI
```
$ pip install psiz
```
You can optionally install the python packages necessary for running package tests (e.g., `pytest`):
```
$ pip install "psiz[test]"
```

### Using git
```
# Clone the PsiZ repository from GitHub to your local machine.
$ git clone https://github.com/psiz-org/psiz.git
# Use `pip` to install the cloned repository.
$ pip install /local/path/to/psiz
```

**Notes:**
* PsiZ depends on TensorFlow. Please see the [TF compatibility matrix](https://www.tensorflow.org/install/source#gpu) for supported Python and CUDA versions for each version of TensorFlow.
* PsiZ versions <=0.5.0 must be installed using git clone and editable mode (e.g., `pip install -e /local/path/to/psiz`).
* You can install specific releases:
    * using PyPI: `pip install 'psiz==0.5.1'`
    * using git: `git clone https://github.com/psiz-org/psiz.git --branch v0.5.1`

## Attribution
If you use PsiZ in your work please cite at least one of the following:
```
@InProceedings{Roads_Love_2021:CVPR,
    title     = {Enriching ImageNet with Human Similarity Judgments and Psychological Embeddings},
    author    = {Brett D. Roads and Bradley C. Love},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
    month     = {6},
    pages     = {3547--3557}
    doi       = {10.1109/CVPR46437.2021.00355},
}
```
```
@Article{Roads_Mozer_2019:BRM,
    title   = {Obtaining psychological embeddings through joint kernel and metric learning},
    author  = {Brett D. Roads and Michael C. Mozer},
    journal = {Behavior Research Methods},
    year    = {2019},
    volume  = {51},
    pages   = {2180–-2193},
    doi     = {10.3758/s13428-019-01285-3}
}
```

## Contribution Guidelines
If you would like to contribute please see the [contributing guidelines](CONTRIBUTING.md).

This project uses a [Code of Conduct](CODE.md) adapted from the [Contributor Covenant](https://www.contributor-covenant.org/)
version 2.0, available at <https://www.contributor-covenant.org/version/2/0/code_of_conduct.html>.

## Licence
This project is licensed under the Apache Licence 2.0 - see LICENSE file for details.
