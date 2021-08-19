# Contributing to PsiZ

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

PsiZ's scope is restricted to computational modeling of human behavioral data. This includes similarity ratings, similarity rankings, and pile sorts of stimuli. Not all of this functionality is implemented. Contributions that support this functionality are welcome.

## Issues

* Please tag your issue with `bug`, `enhancement`, or `question` to help us effectively respond.
* Please include the versions of TensorFlow and PsiZ you are running.
* Please provide the command line you ran as well as the log output.

## Pull Requests

Please send in fixes and feature additions through Pull Requests.

## Testing

* PsiZ uses `pytest` for testing, `pytest-cov` for coverage analytics, and `tox` for testing multiple python versions. These packages should be installed separately by the tester.
* A coverage report can be generated using `pytest --cov`
* An xml coverage report for uploading to codecov can be generated using `py.test --cov-report=xml --cov=psiz tests/`
* Tests that take more than a few seconds are marked as `slow` and can be skipped by running `pytest -m "not slow"`.
* All pytest markers must be registered in `pytest.ini`, unregistered markers will generate an error.

# Additional Guidance

PsiZ closely adheres to TensorFlow and Keras idioms. Model components are implemented as layers. Custom Keras objects are placed in `psiz.keras` and intentionally mirror the module structure of `tensorflow.keras` in order to leverage developers pre-existing knowledge of TensorFlow's organization.

## Module: trials
* Code to the appropriate interface.

### Trials
* All trial objects have a n_trial attribute.
* All trial objects have a stimulus_set attribute.

### Observations
* requires as_datatset() method

## Module: models
* take an appropriate Dataset as an argument
* for get_config(): follow the {'class_name': str, 'config': dict} pattern

## Contributing to psiz.keras

Since the models used in this package are prone to find sub-optimal solutions, multiple restarts are necessary. This results in a point of divergence from typical TensorFlow projects. TensorFlow does not provide a pre-packaged strategy for performing multiple restarts of a model. To accommodate this need, some basic restart functionality is provided by the Restarter class.

Register the class via the decorator @tf.keras.utils.register_keras_serializable in order to facilitate model loading.

Guidance for other Keras objects is listed below.

### Custom `Constraint`
* implement `__call__` method
* get_config()
    * Do not need to implement get_config if no class attributes
    * If implementing, do not need to call `super(_).get_config()`
    * don't need to always return dtype

### Custom `Initializers`
* `__init__`
    * Should not take dtype argument.
* implement `__call__` method
    * Should take `shape` and `dtype` argument
* get_config()
    * Do not need to implement get_config if no class attributes
    * If implementing, do not need to call `super(_).get_config()`
* seed=None should be in initialization signature and seed should be returned in config.


### Custom `Layer`
* Subclass `Layer`.
* `__init__`
    * include `**kwargs` argument and pass to `super().__init__(**kwargs)`
* implement `call` not `__call__` method
* `get_config()`
    * If custom layer requires initialization arguments, then implement by first calling `super(_).get_config()` and then update the dictionary with the custom layer's attributes.
    * Following TensorFlow convention, the returned configuration should be a dictionary of the form: {'class_name': str, 'config': dict}.

#### Kernel
* Some kernels require a dimensionality be specified in advance. If so, they should implement @property `n_dim` like `AttentionKernel` since the dimensionality will be checked for agreement with the embedding layer.
* MAYBE Kernels should assign a dictionary to the attribute `theta` which keeps track of all the kernel variables.

### Custom `Model`
* `init()`
    * use `**kwargs` in arguments and call `super().__init__(**kwargs)`
* implement `call` method
    * decorate `call` method with `@tf.function` to enable graph execution.
* `get_config()`
    * TODO

### Custom `Regularizer`
* implement `__call__` method
* `get_config()`
    * Do not need to implement get_config if no class attributes
    * If implementing, do not need to call `super(_).get_config()`
