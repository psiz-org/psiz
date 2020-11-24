# PsiZ: A Psychological Embedding Package

## What's in a name?

The name PsiZ (pronounced like the word *size*, /sʌɪz/) is meant to serve as shorthand for the term *psychological embedding*. The greek letter Psi is often used to represent the field of psychology and the matrix variable **Z** is often used in machine learning to denote a latent feature space.

## Purpose

PsiZ provides the computational tools to infer a continuous, multivariate stimulus representation using similarity relations. It integrates well-established cognitive theory with contemporary computational methods.

## Installation

There is not yet a stable version. All APIs are subject to change and all releases are alpha.

To install the latest development version, clone from GitHub and instal the local repo using pip.
1. Use `git` to clone the latest version to your local machine: `git clone https://github.com/roads/psiz.git`
2. Use `pip` to install the cloned repo (using editable mode): `pip install -e /local/path/to/psiz`
By using editable mode, you can easily update your local copy by use `git pull origin master` inside your local copy of the repo. You do not have to re-install with `pip`.

The package can also be obtained by:
* Manually downloading the latest version at https://github.com/roads/psiz.git
* Use git to clone a specific release, for example: `git clone https://github.com/roads/psiz.git --branch v0.3.0`
* Using PyPi to install older alpha releases: ``pip install psiz``. The versions available through PyPI lag behind the latest GitHub version.

**Note:** PsiZ also requires TensorFlow. In older versions of TensorFlow, CPU only versions were targeted separately. For Tensorflow >=2.0, both CPU-only and GPU versions are obtained via `tensorflow`. The current `setup.py` file fulfills this dependency by downloading the `tensorflow` package using `pip`.

## Quick Start

Similarity relations can be collected using a variety of paradigms. You will need to use the appropriate model for your data. In addition to a model choice, you need to provide two additional pieces of information:

1. The observed similarity relations (referred to as observations or *obs*).
2. The number of unique stimuli that will be in your embedding (`n_stimuli`).


The following minimalist example uses a `Rank` psychological embedding to model a predefined set of ordinal similarity relations.
```python
import psiz

# Load observations from a predefined dataset.
(obs, catalog) = psiz.datasets.load('birds-16')
# Create a TensorFlow embedding layer for the stimuli.
# NOTE: Since we will use masking, we increment n_stimuli by one.
stimuli = tf.keras.layers.Embedding(
    catalog.n_stimuli+1, mask_zero=True
)
# Use a default kernel (exponential with p-norm).
kernel = psiz.keras.layers.Kernel()
# Create a Rank model that subclasses TensorFlow Keras Model.
model = psiz.models.Rank(stimuli=stimuli, kernel=kernel)
# Wrap the model in convenient proxy class.
emb = psiz.models.Proxy(model)
# Compile the model.
emb.compile()
# Fit the psychological embedding using observations.
emb.fit(obs)
# Optionally save the fitted model.
emb.save('my_embedding')
```

Check out the [examples](examples/) directory to explore examples that take advantage of the various features that PsiZ offers.


## Trials and Observations

Inference is performed by fitting a model to a set of observations. In this package, a single observation is comprised of trial where multiple stimuli that have been judged by an agent (human or machine) based on their similarity. There are currently three different types of trials: *rank*, *rate* and *sort*.

### Rank

In the simplest case, an observation is obtained from a trial consisting of three stimuli: a *query* stimulus (*q*) and two *reference* stimuli (*a* and *b*). An agent selects the reference stimulus that they believe is more similar to the query stimulus. For this simple trial, there are two possible outcomes. If the agent selected reference *a*, then the observation for the *i*th trial would be recorded as the vector: 

D<sub>*i*</sub> = [*q* *a* *b*]

Alternatively, if the agent had selected reference *b*, the observation would be recorded as:

D<sub>*i*</sub> = [*q* *b* *a*]

In addition to a simple *triplet* rank trial, this package is designed to handle a number of different rank trial configurations. A trial may have 2-8 reference stimuli and an agent may be required to select and rank more than one reference stimulus. A companion Open Access article dealing with rank trials is available at https://link.springer.com/article/10.3758/s13428-019-01285-3.

### Rate

In the simplest case, an observation is obtained from a trial consisting of two stimuli. An agent provides a numerical rating regarding the similarity between the stimuli. *This functionality is not currently available and is under development.*

### Sort

In the simplest case, an observation is obtained from a trial consisting of three stimuli. Ag agent sorts the stimuli into two groups based on similarity. *This functionality is not currently available and is under development.*

## Using Your Own Data

To use your own data, you should place your data in an appropriate subclass of `psiz.trials.Observations`. Once the `Observations` object has been created, you can save it to disk by calling its `save` method. It can be loaded later using the function `psiz.trials.load(filepath)`. Consider the following example that creates random rank observations:

```python
import numpy as np
import psiz

# Let's assume that we have 10 unique stimuli.
stimuli_list = np.arange(0, 10, dtype=int)

# Let's create 100 trials, where each trial is composed of a query and
# four references. We will also assume that participants selected two
# references (in order of their similarity to the query.)
n_trial = 100
n_reference = 4
stimulus_set = np.empty([n_trial, n_reference + 1], dtype=int)
n_select = 2 * np.ones((n_trial), dtype=int)
for i_trial in range(n_trial):
    # Randomly selected stimuli and randomly simulate behavior for each
    # trial (one query, four references).
    stimulus_set[i_trial, :] = np.random.choice(
        stimuli_list, n_reference + 1, replace=False
    )

# Create the observations object and save it to disk.
obs = psiz.trials.RankObservations(stimulus_set, n_select=n_select)
obs.save('path/to/obs.hdf5')

# Load the observations from disk.
obs = psiz.trials.load_trials('path/to/obs.hdf5')
```
Note that the values in `stimulus_set` are assumed to be contiguous integers [0, N[, where N is the number of unique stimuli. Their order is also important. The query is listed in the first column, an agent's selected references are listed second (in order of selection if there are more than two) and then any remaining unselected references are listed (in any order).

## Design Philosophy

PsiZ is built around the TensorFlow ecosystem and strives to follow TensorFlow idioms as closely as possible. See [CONTRIBUTING](CONTRIBUTING.md) for additional guidance.

### Model, Layer, Variable

Package-defined models are built by sub-classing `tf.keras.Model`. Components of a model are built using the `tf.keras.layers.Layer` API. A free parameter is implemented as a `tf.Variable`.

In PsiZ, a model can be thought as having two major components. The first component is a psychological embedding which describes how the agent of interest perceives similarity between set of stimuli. This component includes a conventional embedding (representing the stimuli in psychological space) and a kernel that defines similarities between embedding points. The second component describes how similarities are converted into an observed behavior, such as rankings or ratings.

PsiZ includes a number of predefined layers to facilitate the construction of arbitrary models. For example, there are four predefined similarity functions (implemented as subclasses of `tf.keras.layers.Layer`) which can be used to create a kernel:

1. `psiz.keras.layers.InverseSimilarity`
2. `psiz.keras.layers.ExponentialSimilarity`
3. `psiz.keras.layers.HeavyTailedSimilarity`
4. `psiz.keras.layers.StudentsTSimilarity`

Each similarity function has its own set of parameters (i.e., `tf.Variable`s). The `ExponentialSimilarity`, which is widely used in psychology, has four variables. Users can also implement there own similarity functions by sub-classing `tf.keras.layers.Layers`.

### Deviations from TensorFlow

The models in PsiZ are susceptible to local optima. While the usual tricks help, such as stochastic gradient decent, we typically require multiple restarts with different initializations to be confident in the solution. In an effort to shield users from the burden of writing restart logic, PsiZ includes a restart module that is employed by the `fit` method of the `Proxy` class. The state of most TensorFlow objects can be reset using a serialization/deserialization strategy. However, `tf.keras.callbacks` do not permit this strategy. To fix this problem, PsiZ implements a subclass of `tf.keras.callbacks.Callback`, which adds a `reset` method. This solution is unattractive and is likely to change when a more elegant solution is found. 


## Modules
* `agents` - Simulate an agent making similarity judgments.
* `catalog` - Class for storing stimulus information.
* `datasets` - Functions for loading some pre-defined catalogs and observations.
* `dimensionality` - Routine for selecting the dimensionality of the embedding.
* `generators` - Generate new trials randomly or using active selection.
* `keras` - A module containing Keras related classes.
* `models` - A set of pre-defined psychological embedding models.
* `preprocess` - Functions for preprocessing observations.
* `restart` - Classes and functionality for performing model restarts.
* `trials` - Classes and functions for creating and managing observations.
* `utils` - Utility functions.
* `visualize` - Functions for visualizing embeddings.

## Authors
* Brett D. Roads
* Michael C. Mozer
* See also the list of contributors who participated in this project.

## Licence
This project is licensed under the Apache Licence 2.0 - see LICENSE file for details.

## Code of Conduct
This project uses a Code of Conduct [CODE](CODE.md) adapted from the [Contributor Covenant][homepage], version 2.0, available at <https://www.contributor-covenant.org/version/2/0/code_of_conduct.html>.

## References
* van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic triplet
   embedding. In Machine learning for signal processing (mlsp), 2012 IEEE
   international workshop on (p. 1-6). doi:10.1109/MLSP.2012.6349720
* Roads, B. D., & Mozer, M. C. (2019). Obtaining psychological
   embeddings through joint kernel and metric learning. Behavior Research
   Methods. 51(5), 2180-2193. doi:10.3758/s13428-019-01285-3
* Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). The
   Caltech-UCSD Birds-200-2011 Dataset (Tech. Rep. No. CNS-TR-2011-001).
   California Institute of Technology.
