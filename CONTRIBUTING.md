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

### Warning Filter Policy

PsiZ test warnings should prioritize signals that are under PsiZ control.

* We keep deprecation and runtime warnings visible when they indicate issues in PsiZ source or tests.
* We may add narrowly scoped filters for known third-party warnings (for example, upstream framework/runtime warnings) when:
    * the warning does not indicate a PsiZ defect, and
    * it materially obscures actionable warnings from PsiZ code.

Filter scope requirements:

* Prefer filters constrained by both warning message signature and emitting third-party module path.
* Avoid broad filters that could hide new warnings from PsiZ modules.

Maintenance expectations:

* Third-party warning filters require periodic audit/review as dependency versions evolve.
* During periodic updates, re-check whether each filter is still necessary and remove obsolete filters promptly.

### Understanding tox.ini

`tox` is the canonical way to run PsiZ tests in a controlled Python+backend matrix.

#### Matrix and environment selection
* `[tox] minversion = 4.20`
    * Ensures developers are on a tox version that supports the syntax used in this repo.
* `envlist = py{310,311,312,313}-{tensorflow,torch,jax}`
    * Defines the default runtime matrix: each Python version combined with each backend.
* `envlist` also includes `py310-{tensorflow,torch,jax}-slow`
    * Defines explicit slow-test environments to keep normal local runs fast.

#### Shared testenv behavior
* `[testenv] package = wheel`
    * Builds and installs PsiZ as a wheel in each env before tests run. This validates packaging and catches import/install issues.
* `install_command = python -m pip install {opts} {packages}`
    * Default package install path for all backends.
* `extras = test` and `extras = backend-{env:PSIZ_BACKEND}`
    * Installs core test dependencies and backend-specific extras based on the selected backend.

#### Environment variables set by tox
* `PYTHONHASHSEED = 0`
    * Improves reproducibility across runs.
* `PSIZ_EXPECTED_BACKEND = {env:KERAS_BACKEND:}`
    * Captures expected backend from ambient environment when provided.
* Backend-scoped blocks set:
    * `KERAS_BACKEND` (forces Keras backend in that tox env)
    * `PSIZ_BACKEND` (selects matching extras)
    * `PSIZ_BACKEND_FILTER` (pytest marker filter that excludes other backend-specific tests)
* Torch-only block also sets:
    * `CUDA_VISIBLE_DEVICES = {env:PSIZ_TORCH_CUDA_VISIBLE_DEVICES:0}`
    * Default behavior runs torch tests on GPU 0 when available; override with `PSIZ_TORCH_CUDA_VISIBLE_DEVICES`.

#### Torch install behavior
* `torch: install_command = python -m pip install --extra-index-url {env:PSIZ_TORCH_EXTRA_INDEX_URL:https://pypi.ngc.nvidia.com} {opts} {packages}`
    * Keeps torch environments aligned with NVIDIA-recommended wheel source by default.
    * Override via `PSIZ_TORCH_EXTRA_INDEX_URL` if you need a different index mirror.

#### passenv and debug support
* `passenv` forwards selected host env vars into tox envs:
    * `KERAS_BACKEND`, `PSIZ_EXPECTED_BACKEND`
    * `CUDA_VISIBLE_DEVICES`
    * `CUDA_LAUNCH_BLOCKING`, `TORCH_USE_CUDA_DSA`
* What each forwarded variable does:
    * `KERAS_BACKEND`
        * Declares the backend you expect Keras to use (`tensorflow`, `torch`, or `jax`).
        * In tox, backend-specific `setenv` still controls the active backend for each env; this variable is mainly useful for explicit intent and sanity checking.
    * `PSIZ_EXPECTED_BACKEND`
        * Used by `commands_pre` for fail-fast validation.
        * If the runtime backend does not match this value, tox fails before running tests.
    * `CUDA_VISIBLE_DEVICES`
        * Controls which GPU IDs are visible to the test process.
        * Useful for pinning to a specific GPU or forcing CPU fallback (for example, set to `-1` where supported).
    * `CUDA_LAUNCH_BLOCKING`
        * When set to `1`, CUDA kernels execute synchronously.
        * Slower, but stack traces point closer to the actual failing operation, which is helpful for debugging device-side asserts.
    * `TORCH_USE_CUDA_DSA`
        * Enables PyTorch CUDA device-side assertions.
        * Helpful for diagnosing invalid index and bounds issues in GPU kernels; may add overhead and should generally be used for debugging runs.
* Why this exists:
    * Lets you enable CUDA debugging and device-selection controls without editing `tox.ini`.

#### Safety check before tests
* `commands_pre` verifies active Keras backend equals expected backend.
    * Fails fast if a tox env is misconfigured, preventing misleading test results.

#### Test command routing
* Default command:
    * `python -m pytest -m "not slow and not backend_slow and {env:PSIZ_BACKEND_FILTER}" {posargs}`
    * Runs non-slow tests while excluding tests for other backends.
* `slow` factor command:
    * `python -m pytest -m "backend_slow and {env:PSIZ_BACKEND_FILTER}" {posargs}`
    * Runs only backend slow tests for envs ending in `-slow`.
* `{posargs}`
    * Allows appending file paths, test IDs, or flags after `--` on the tox command line.

#### Common usage patterns
* Run one backend env:
    * `uv run tox -e py310-torch`
* Run one test file in that env:
    * `uv run tox -e py310-torch -- tests/psiz/keras/models/test_rank_model.py -q`
* Run one test node:
    * `uv run tox -e py310-torch -- tests/psiz/keras/models/test_rank_model.py::TestSoftRank::test_usage_subclass_a -q`
* Run with CUDA debug flags:
    * `CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 uv run tox -e py310-torch -- -q`
* Select a different GPU for torch tox env:
    * `PSIZ_TORCH_CUDA_VISIBLE_DEVICES=1 uv run tox -e py310-torch`
* Override torch package index source:
    * `PSIZ_TORCH_EXTRA_INDEX_URL=https://pypi.org/simple uv run tox -e py310-torch`

### Useful Commands for Local Checks
* `KERAS_BACKEND=torch uv run pytest -m "not slow and not backend_slow"`
    * Only run tests that are not marked as `slow`.
* `KERAS_BACKEND=tensorflow uv run pytest tests/psiz/backend/test_backend_matrix_smoke.py -m "backend_runtime and backend_tensorflow" -q`
    * Run true-runtime backend smoke tests under TensorFlow.
* `KERAS_BACKEND=jax uv run pytest tests/psiz/backend/test_backend_matrix_smoke.py::test_backend_matrix_slow_subset -m "backend_slow" -q`
    * Run reduced slow backend subset under JAX.
* `uv run tox -e py310-torch`
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