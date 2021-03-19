# PsiZ: A Psychological Embedding Package

WARNING: This package is pre-release and the API is not stable.

## What's in a name?

The name PsiZ (pronounced like the word *size*, /sʌɪz/) is meant to serve as shorthand for the term *psychological embedding*. The greek letter Psi is often used to represent the field of psychology and the matrix variable **Z** is often used in machine learning to denote a latent feature space.

## Purpose

PsiZ provides the computational tools to infer a continuous, multivariate stimulus representation using similarity relations. It integrates cognitive theory with contemporary computational methods.

## Installation

There is not yet a stable version. All APIs are subject to change and all releases are alpha.

To install the latest development version, clone from GitHub and install the local repo using pip.
1. Use `git` to clone the latest version to your local machine: `git clone https://github.com/roads/psiz.git`
2. Use `pip` to install the cloned repo (using editable mode): `pip install -e /local/path/to/psiz`.
By using editable mode, you can easily update your local copy by use `git pull origin master` inside your local copy of the repo. You do not have to re-install with `pip`.

The package can also be obtained by:
* Manually downloading the latest version at https://github.com/roads/psiz.git
* Use git to clone a specific release, for example: `git clone https://github.com/roads/psiz.git --branch v0.3.0`
* Using PyPi to install older alpha releases: ``pip install psiz``. The versions available through PyPI lag behind the GitHub versions.

**Note:** PsiZ also requires TensorFlow. In older versions of TensorFlow, CPU only versions were targeted separately. For Tensorflow >=2.0, both CPU-only and GPU versions are obtained via `tensorflow`. The current `setup.py` file fulfills this dependency by downloading the `tensorflow` package using `pip`.

## Contribution Guidelines
If you would like to contribute please see the [contributing guidelines](CONTRIBUTING.md).

This project uses a [Code of Conduct](CODE.md) adapted from the [Contributor Covenant](https://www.contributor-covenant.org/)
version 2.0, available at <https://www.contributor-covenant.org/version/2/0/code_of_conduct.html>.

## Resources
* [Psiz Documentation](https://psiz.readthedocs.io/en/latest/)
* [PsiZ Examples](examples/)

## Licence
This project is licensed under the Apache Licence 2.0 - see LICENSE file for details.
