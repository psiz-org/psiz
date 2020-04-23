# Contributing to PsiZ

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

PsiZ's scope is restricted to computational modeling of *similarity* data. This includes ratings, rankings, and sorts of stimuli. Not all of this functionality is implemented. Contributions that support this functionality are welcome.

PsiZ attempts to closely adhere to TensorFlow and Keras idioms. Model components are implemented as layers. All custom Keras objects places in `psiz.keras` and intentionally mirror the module structure of `tensorflow.keras` in order to leverage developers pre-existing knowledge of TensorFlow's organization.

## Pull Request Process
* TODO

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

Since the models used in this package are prone to find sub-optimal solutions, multiple restarts are necessary. This results in a point of divergence from typical TensorFlow projects. TensorFlow does not provide a great strategy 'resetting' a model. To accommodate this need, developers should implement custom Layers with a `reset_weights()` method.

Register the class via the decorator @tf.keras.utils.register_keras_serializable in order to facilitate model loading.

Guidance for other Keras objects is listed below.

### Custom `Constraint`
* implement `__call__` method
* get_config()
    * Do not need to implement get_config if no class attributes
    * If implementing, do not need to call `super(_).get_config()`
    * don't need to always return dtype

### Custom `Initializers`
* implement `call` method, not `__call__` method
* get_config()
    * Do not need to implement get_config if no class attributes
    * If implementing, do not need to call `super(_).get_config()`

### Custom `Layer`
* Subclass `LayerRe` which assumes a `reset_weights()` method.
* `__init__`
    * include `**kwargs` argument and pass to `super().__init__(**kwargs)`
* implement `call` not `__call__` method
* `get_config()`
    * If custom layer requires initialization arguments, then implement by first calling `super(_).get_config()` and then update the dictionary with the custom layer's attributes.
    * Following TensorFlow convention, the returned configuration should be a dictionary of the form: {'class_name': str, 'config': dict}.
* `reset_weights()`
    * Implement.

### Custom `Model`
* `init()`
    * use `**kwargs` in arguments and call `super().__init__(**kwargs)`
* implement `call` method
    * decorate `call` method with `@tf.function` to enable graph execution.
* `get_config()`
    * TODO
* `reset_weights()`
    * Must implement.

### Custom `Regularizer`
* implement `__call__` method
* `get_config()`
    * Do not need to implement get_config if no class attributes
    * If implementing, do not need to call `super(_).get_config()`

## Code of Conduct

### Our Pledge

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

### Our Standards

Examples of behavior that contributes to a positive environment for our community include:

* Demonstrating empathy and kindness toward other people
* Being respectful of differing opinions, viewpoints, and experiences
* Giving and gracefully accepting constructive feedback
* Accepting responsibility and apologizing to those affected by our mistakes, and learning from the experience
* Focusing on what is best not just for us as individuals, but for the overall community

Examples of unacceptable behavior include:

*The use of sexualized language or imagery, and sexual attention or advances of any kind
*Trolling, insulting or derogatory comments, and personal or political attacks
*Public or private harassment
*Publishing others’ private information, such as a physical or email address, without their explicit permission
*Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement Responsibilities

Community leaders are responsible for clarifying and enforcing our standards of acceptable behavior and will take appropriate and fair corrective action in response to any behavior that they deem inappropriate, threatening, offensive, or harmful.

Community leaders have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to this Code of Conduct, and will communicate reasons for moderation decisions when appropriate.

### Scope

This Code of Conduct applies within all community spaces, and also applies when an individual is officially representing the community in public spaces. Examples of representing our community include using an official e-mail address, posting via an official social media account, or acting as an appointed representative at an online or offline event.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to the community leaders responsible for enforcement at brett.roads@gmail.com. All complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the reporter of any incident.

### Enforcement Guidelines

Community leaders will follow these Community Impact Guidelines in determining the consequences for any action they deem in violation of this Code of Conduct:

1. Correction
Community Impact: Use of inappropriate language or other behavior deemed unprofessional or unwelcome in the community.

Consequence: A private, written warning from community leaders, providing clarity around the nature of the violation and an explanation of why the behavior was inappropriate. A public apology may be requested.

2. Warning
Community Impact: A violation through a single incident or series of actions.

Consequence: A warning with consequences for continued behavior. No interaction with the people involved, including unsolicited interaction with those enforcing the Code of Conduct, for a specified period of time. This includes avoiding interactions in community spaces as well as external channels like social media. Violating these terms may lead to a temporary or permanent ban.

3. Temporary Ban
Community Impact: A serious violation of community standards, including sustained inappropriate behavior.

Consequence: A temporary ban from any sort of interaction or public communication with the community for a specified period of time. No public or private interaction with the people involved, including unsolicited interaction with those enforcing the Code of Conduct, is allowed during this period. Violating these terms may lead to a permanent ban.

4. Permanent Ban
Community Impact: Demonstrating a pattern of violation of community standards, including sustained inappropriate behavior, harassment of an individual, or aggression toward or disparagement of classes of individuals.

Consequence: A permanent ban from any sort of public interaction within the community.

### Attribution
This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 2.0, available at <https://www.contributor-covenant.org/version/2/0/code_of_conduct.html>.

Community Impact Guidelines were inspired by Mozilla’s code of conduct enforcement ladder.

For answers to common questions about this code of conduct, see the FAQ at <https://www.contributor-covenant.org/faq>. Translations are available at <https://www.contributor-covenant.org/translations>.

[homepage]: http://contributor-covenant.org