# PsiZ: A Psychological Embedding Package

## Purpose
PsiZ provides the computational tools to infer a continuous, multivariate stimulus representation using ordinal similarity relations. It integrates well-established cognitive theory with contemporary computational methods. The companion Open Access article is available at https://link.springer.com/article/10.3758/s13428-019-01285-3.

## Installation
There are two ways to install PsiZ:

1. Install from PyPI using pip: ``pip install psiz``
2. Clone the repository from GitHub and install using pip: `pip install /local/path/to/psiz`. The repository can be cloned in a number of ways:
    * Manually download the latest version at https://github.com/roads/psiz.git
    * Use git to clone the latest version: `git clone https://github.com/roads/psiz.git`
    * Use git to clone a specific release, for example: `git clone https://github.com/roads/psiz.git --branch v0.2.2`

**Note:** PsiZ also requires TensorFlow 2.0, which is not installed automatically since it is not included as a dependency of the PsiZ package (in setup.py). You must explicitly install TensorFlow 2.0 (tensorflow or tensorflow-gpu). This allows users to specify whether they use a GPU enabled version of TensorFlow.

## Quick Start
There are four predefined embedding models to choose from:

1. Inverse
2. Exponential
3. HeavyTailed
4. StudentsT

Once you have selected an embedding model, you must provide two pieces of information in order to infer an embedding.

1. The similarity judgment observations (abbreviated as obs).
2. The number of unique stimuli that will be in your embedding.

```python
import psiz

# Load some observations (i.e., judged trials).
(obs, catalog) = psiz.datasets.load('birds-16')
# Initialize an embedding model.
emb = psiz.models.Exponential(catalog.n_stimuli)
# Fit the embedding model using similarity judgment observations.
emb.fit(obs)
# Optionally save the fitted model.
emb.save('my_embedding.h5')
```

## Trials and Observations
Inference is performed by fitting a model to a set of observations. In this package, a single observation is comprised of multiple stimuli that have been judged by an agent (human or machine) based on their similarity. 

In the simplest case, an observation is obtained from a trial consisting of three stimuli: a *query* stimulus (Q) and two *reference* stimuli (A and B). An agent selects the reference stimulus that they believe is more similar to the query stimulus. For this simple trial, there are two possible outcomes. If the agent selected reference A, then the observation for the ith trial would be recorded as the vector: 

D_i = [Q A B]

Alternatively, if the agent had selected reference B, the observation would be recorded as:

D_i = [Q B A]

In addition to a simple *triplet* trial, this package is designed to handle a number of different trial configurations. A trial may have 2-8 reference stimuli and an agent may be required to select and rank more than one reference stimulus. 

## Using Your Own Data

To use your own data, you should place your data in a `psiz.trials.Observations` object. Once the Observations object has been created, you can save it to disk by calling its `save` method. It can be loaded later using the function `psiz.trials.load_trials()`. Consider the following example that uses randomly generated data:

```python
import numpy as np
import psiz.trials

# Let's assume that we have 10 unique stimuli.
stimuli_list = np.arange(0, 10, dtype=int)

# Let's create 100 trials, where each trial is composed of a query and
# four references. We will also assume that participants selected two
# references (in order of their similarity to the query.)
n_trial = 100
n_reference = 4
response_set = np.empty([n_trial, n_reference + 1], dtype=int)
n_select = 2 * np.ones((n_trial), dtype=int)
for i_trial in range(n_trial):
    # Randomly selected stimuli and randomly simulate behavior for each
    # trial (one query, four references).
    response_set[i_trial, :] = np.random.choice(
        stimuli_list, n_reference + 1, replace=False
    )

# Create the observations object and save it to disk.
obs = psiz.trials.Observations(response_set, n_select=n_select)
obs.save('path/to/obs.hdf5')

# Load the observations from disk.
obs = psiz.trials.load_trials('path/to/obs.hdf5')
```
Note that the values in `response_set` are assumed to be contiguous integers [0, N[, where N is the number of unique stimuli. Their order is also important. The query is listed in the first column, an agent's selected references are listed second (in order of selection if the trial is ranked) and then any remaining unselected references are listed (in any order).

## Common Use Cases
Optionally, you can provide additional information.

1. The dimensionality of the embedding (default=2).
2. The number of unique population groups (default=1).

```python
n_stimuli = 100
emb = psiz.models.Exponential(n_stimuli, n_dim=4, n_group=2)
emb.fit(obs)
```

If you know some of the free parameters already, you can set them to the desired value and then make those parameters untrainable.
```python
n_stimuli = 100
emb = psiz.models.Exponential(n_stimuli, n_dim=2)
emb.rho = 2
emb.tau = 1
emb.trainable({'rho': False, 'tau': False})
emb.fit(obs)
```

## Modules
* `dimensionality` - Function for selecting the dimensionality of the embedding.
* `generator` - Generate new trials randomly or using active selection.
* `models` - A set of pre-defined pscyhological embedding models.
* `preprocess` - Functions for preprocessing observations.
* `simulate` - Simulate an agent making similarity judgements.
* `trials` - Data structure used for trials and observations.
* `utils` - Utility functions.
* `visualize` - Functions for visualizing embeddings.
* `datasets` - Functions for loading pre-collecgted datasets.

## Authors
* Brett D. Roads
* Michael C. Mozer
* See also the list of contributors who participated in this project.

## What's in a name?
The name PsiZ (pronounced *sigh zeee*) is meant to serve as shorthand for the term *psychological embedding*. The greek letter Psi is often used to represent the field of psychology and the matrix variable **Z** is often used in machine learning to denote a latent feature space.

## Licence
This project is licensed under the Apache Licence 2.0 - see the LICENSE.txt file for details.

### References
* van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic triplet
   embedding. In Machine learning for signal processing (mlsp), 2012 IEEE
   international workshop on (p. 1-6). doi:10.1109/MLSP.2012.6349720
* Roads, B. D., & Mozer, M. C. (2019). Obtaining psychological
   embeddings through joint kernel and metric learning. Behavior Research
   Methods. 51(5), 2180-2193. doi:10.3758/s13428-019-01285-3
* Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). The
   Caltech-UCSD Birds-200-2011 Dataset (Tech. Rep. No. CNS-TR-2011-001).
   California Institute of Technology.
