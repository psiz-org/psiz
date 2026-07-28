# Contributing to PsiZ

When contributing to this repository, please first discuss the change you wish to make via issue or email with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

PsiZ's scope is restricted to computational modeling of human behavioral data. This includes similarity ratings, similarity rankings, pile-sorts, and categorization of stimuli. Not all of this functionality is implemented. Contributions that support this functionality are welcome.

PsiZ closely adheres to Keras idioms across TensorFlow, PyTorch, and JAX backends. Model components are implemented as layers. Custom Keras objects are placed in `psiz.keras` and intentionally mirror the module structure of `keras` in order to leverage developers and users pre-existing knowledge of Keras' organization.

## Issues

* Please tag your issue with `bug`, `enhancement`, or `question` to help us effectively respond.
* Please include the versions of PsiZ, Keras, and your selected backend runtime (TensorFlow, PyTorch, or JAX).
* If your issue involves TensorFlow-specific stochastic behavior, include TensorFlow Probability version as well.
* Please provide the command line or code you ran as well as the log output.

## Pull Requests

Please send in fixes and feature additions through Pull Requests.

## Testing

* PsiZ uses a number of tools for testing.
    * `pytest` for testing
    * `pytest-cov` for coverage analytics
    * `tox` for local backend and Python matrix validation.
* These packages can be installed via dependency groups (for example, `uv sync --group test --group backend-torch` or `pip install --group test --group backend-torch .`).
* See `pytest.ini` for a list and description of all pytest markers (e.g., `adapter_surface`, `backend_runtime`, `backend_slow`).
    * NOTE: All pytest markers must be registered in `pytest.ini`, unregistered markers will generate an error.

### Useful Commands for Local Checks
* `KERAS_BACKEND=torch uv run pytest -m "not slow and not backend_slow"`
    * Only run tests that are not marked as `slow`.
* `KERAS_BACKEND=tensorflow uv run pytest tests/psiz/backend/test_backend_matrix_smoke.py -m "backend_runtime and backend_tensorflow" -q`
    * Run true-runtime backend smoke tests under TensorFlow.
* `KERAS_BACKEND=jax uv run pytest tests/psiz/backend/test_backend_matrix_smoke.py::test_backend_matrix_slow_subset -m "backend_slow" -q`
    * Run reduced slow backend subset under JAX.
* `uv run tox -e py311-torch`
    * Run tox backend runtime checks for one Python/backend target.
* `uv run pytest --cov-report term-missing --cov=psiz tests`
    * Output a coverage report to the terminal that includes which statements were not covered by the tests.

### Linting
* Useful commands:
    * `flake8 src --ignore=F401,E501,W503 --count --show-source --statistics`
    * `pylint --disable=R,C,W --ignored-modules=tensorflow src`

## Versioning
* Versions are released following [Semantic Versioning 2.0.0](https://semver.org/) which follows the MAJOR.MINOR.PATCH format.

---
**WARNING:** This package is pre-release and the API is not stable. Minor version changes may contain breaking changes until `v1.0.0` is released.

---

### Branches
* `main`: Always points to the latest stable version.
* `r<MAJOR>.<MINOR>`: The major-minor branches serve as a development branches. Note that a new branch is **NOT** created for each patch version.
    * Developers should create feature branches that branch from a particular "major-minor branch" `rX.Y`. 

### Tags and Releases
* All releases are tagged using the format `v<MAJOR>.<MINOR>.<PATCH>`, thus a given release branch can have multiple tags that differ by patch number.
* Only tagged releases on a major-minor branch are merged with `main`.

## Docs

* To test generating the docs, you will need to make sure you have the appropriate dependencies installed: `python -m pip install --upgrade --no-cache-dir -r docs/requirements.txt`.
* You can then build the docs by executing the following command inside the `docs` directory: `make html`.
* The built html files can be found in `docs/_build/html`