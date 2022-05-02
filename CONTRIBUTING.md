# Contributing to PsiZ

When contributing to this repository, please first discuss the change you wish to make via issue or email with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

PsiZ's scope is restricted to computational modeling of human behavioral data. This includes similarity ratings, similarity rankings, pile-sorts, and categorization of stimuli. Not all of this functionality is implemented. Contributions that support this functionality are welcome.

PsiZ closely adheres to TensorFlow and Keras idioms. Model components are implemented as layers. Custom Keras objects are placed in `psiz.keras` and intentionally mirror the module structure of `tensorflow.keras` in order to leverage developers and users pre-existing knowledge of TensorFlow's organization.

## Issues

* Please tag your issue with `bug`, `enhancement`, or `question` to help us effectively respond.
* Please include the versions of TensorFlow, TensorFlow Probability and PsiZ you are running.
* Please provide the command line or code you ran as well as the log output.

## Pull Requests

Please send in fixes and feature additions through Pull Requests.

## Testing

* PsiZ uses a number of tools for testing.
    * `pytest` for testing
    * `pytest-cov` for coverage analytics
    * `tox` for locally testing multiple python versions (`tox` is not used for remote GitHub Actions testing).
* These packages should be installed separately by the tester, but most of these packages can be installed at the same time as the core dependencies of psiz using the option `pip install "psiz[test]"`.
* See `pytest.ini` for a list and description of all pytest markers (e.g., `slow`).
    * NOTE: All pytest markers must be registered in `pytest.ini`, unregistered markers will generate an error.

### Useful Commands for Local Checks
* `pytest -m "not slow"`
    * Only run tests that are not marked as `slow`.
* `pytest --cov-report term-missing --cov=psiz tests`
    * Output a coverage report to the terminal that includes which statements were not covered by the tests.


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
