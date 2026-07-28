![PsiZ logo](docs/img/full_logo_300.png)

[![PyPI version](https://badge.fury.io/py/psiz.svg)](https://badge.fury.io/py/psiz)
[![Python](https://img.shields.io/pypi/pyversions/psiz.svg?style=plastic)](https://badge.fury.io/py/psiz)
[![TensorFlow backend](https://img.shields.io/badge/backend-tensorflow-orange.svg)](https://www.tensorflow.org/)
[![PyTorch backend](https://img.shields.io/badge/backend-pytorch-red.svg)](https://pytorch.org/)
[![JAX backend](https://img.shields.io/badge/backend-jax-blue.svg)](https://jax.readthedocs.io/)
[![Documentation Status](https://readthedocs.org/projects/psiz/badge/?version=latest)](https://psiz.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/psiz-org/psiz/branch/main/graph/badge.svg?token=UIK748KI5I)](https://codecov.io/gh/psiz-org/psiz)
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
```bash
pip install psiz
```
Install a backend runtime (choose one):
```bash
pip install "psiz[backend-tensorflow]"
```
```bash
pip install "psiz[backend-torch]"
```
```bash
pip install "psiz[backend-jax]"
```
If you are in a local PsiZ checkout, you can optionally install the Python packages necessary for running package tests (e.g., `pytest`):
```bash
pip install --group test .
```
For backend runtime tests from source checkout, install both test and one backend group:
```bash
pip install --group test --group backend-torch .
```

### Using uv
```bash
uv sync
```
Install with test dependencies:
```bash
uv sync --group test
```
Install with backend runtime dependencies:
```bash
uv sync --group backend-tensorflow
```
```bash
uv sync --group backend-torch
```
```bash
uv sync --group backend-jax
```
Install all groups:
```bash
uv sync --all-groups
```

### Using git
```bash
# Clone the PsiZ repository from GitHub to your local machine.
git clone https://github.com/psiz-org/psiz.git
# Use pip to install the cloned repository.
pip install /local/path/to/psiz
```

## Backend Selection (Keras 3)

PsiZ uses the active Keras backend. Set your backend before importing `keras` or `psiz`.
If no explicit override is provided and no active Keras backend is available,
PsiZ defaults to the `torch` backend.

### Option 1: Environment variable (recommended)
```bash
export KERAS_BACKEND=tensorflow
```
```bash
export KERAS_BACKEND=torch
```
```bash
export KERAS_BACKEND=jax
```

### Option 2: Keras config file
Edit `~/.keras/keras.json` and set the `backend` field.

Example:
```json
{
    "backend": "torch"
}
```

You can verify the selected backend in Python:
```python
import keras
print(keras.backend.backend())
```

## Save and Load .psiz Artifacts

PsiZ now provides a PsiZ-managed storage format for model artifacts.
This helps keep shared research assets stable over time by reducing dependency on backend-specific serialization details that can drift across versions.
The layout is also designed to align closely with hosting workflows on platforms like Hugging Face, making asset publishing and reuse easier.

Canonical API: use the function-based save/load helpers from `psiz.keras`.

Recommended API from `psiz.keras`:
```python
import psiz

# Save to a PsiZ artifact directory.
psiz.keras.save_psiz_model(model, "my_model.psiz")

# Load from a PsiZ artifact directory.
loaded_model = psiz.keras.load_psiz_model("my_model.psiz")
```

If your model subclasses `psiz.keras.StochasticModel`, you can also use the
method-based form as an optional convenience:
```python
model.save_psiz("my_model.psiz")
loaded_model = MyModel.load_psiz("my_model.psiz")
```

## Migrate Legacy .keras Assets to .psiz

PsiZ v0.14 includes a Python API for migrating legacy `.keras` assets into
PsiZ-managed `.psiz` artifacts.

```python
import numpy as np
import psiz

report = psiz.migration.migrate_model_from_keras(
    "legacy_model.keras",
    "migrated_model.psiz",
    backend_override="torch",
    validate_parity=True,
    parity_inputs=np.array([[0.1, 0.2, 0.3]], dtype="float32"),
)

print(report["status"])           # success
print(report["resolved_backend"]) # torch
```

Notes:
* v0.14 migration supports `.keras` legacy assets.
* `.h5`/`.hdf5` migration is intentionally out of scope for v0.14.
* Optional parity validation compares migrated predictions against the legacy model on fixed inputs and tolerance settings.

### Storage Lifecycle Policy

PsiZ uses a two-tier storage strategy with different lifecycle intent:

* Training checkpoints: use Keras checkpointing (for example, `keras.callbacks.ModelCheckpoint`) to support fault tolerance and exact training resume workflows.
* Durable artifacts: use PsiZ-managed `.psiz` directories for long-term portability, sharing, and publication.

Recommended workflow:

1. During training, write checkpoints using Keras-native checkpoint formats.
2. After selecting a final or best model, export once to a `.psiz` artifact directory.

This separation is intentional: checkpoints optimize training continuity, while `.psiz` artifacts optimize stability and reuse.

Notes:
* Artifacts use `model.safetensors` for weights and include strict schema/version checks.
* You can override backend resolution during save/load using `backend_override="tensorflow"`, `"torch"`, or `"jax"`.
* If the target output directory already exists and is non-empty, save raises an error.

**Notes:**
* PsiZ backend dependencies are intentionally modular in v0.14+: install one or more backend groups depending on your workflow.
* TensorFlow GPU workflows still require a CUDA-compatible TensorFlow build that matches your Python/CUDA environment. See the [TF compatibility matrix](https://www.tensorflow.org/install/source#gpu).
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
