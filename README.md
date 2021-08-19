![PsiZ logo](docs/full_logo_300.png)

[![Python](https://img.shields.io/pypi/pyversions/psiz.svg?style=plastic)](https://badge.fury.io/py/psiz)

**WARNING:** This package is pre-release and the API is not stable.

## What's in a name?

The name PsiZ (pronounced like the word *size*, /sʌɪz/) is meant to serve as shorthand for the term *psychological embedding*. The greek letter Psi is often used to represent the field of psychology and the matrix variable **Z** is often used in machine learning to denote a latent feature space.

## Purpose

PsiZ provides the computational tools to infer psychological representations from human behavior (i.e., a psychological embedding). It integrates cognitive theory with contemporary computational methods.

## Installation

---
**WARNING:** There is not yet a stable version. All APIs are subject to change and all releases are alpha.

---

### Install using PyPI
```
pip install psiz
```

### Install using git
This method includes examples and tests in the repository.
1. Use `git` to clone the repository from GitHub to your local machine. 
```
git clone https://github.com/roads/psiz.git
```
2. Then use `pip` to install the cloned repo.
```
pip install /local/path/to/psiz
```

### Notes
* PsiZ depends on TensorFlow. Please see the [TF compatibility matrix](https://www.tensorflow.org/install/source#gpu) for supported Python and CUDA versions for each version of TF.
* Versions 0.5.0 and older must be installed using git clone and editable mode (`pip install -e /local/path/to/psiz`).
* You can install specific releases:
    * using PyPI: `pip install 'psiz==0.5.1'`
    * using git: `git clone https://github.com/roads/psiz.git --branch v0.5.1`

## Resources
* [Psiz Documentation](https://psiz.readthedocs.io/en/latest/)
* [PsiZ Examples](examples/)

## Attribution
If you use PsiZ in your work please cite one of the following:
```
@InProceedings{Roads_Love_2021:CVPR,
    title     = {Enriching ImageNet with Human Similarity Judgments and Psychological Embeddings},
    author    = {Brett D. Roads and Bradley C. Love},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
    month     = {6},
    pages     = {3547--3557}
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
