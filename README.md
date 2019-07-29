# PsiZ: A Psychological Embedding Package

## Purpose
PsiZ provides the computational tools to infer a continuous, multivariate representation for a set of stimuli from ordinal similarity relations.

## Installation
There are two ways to install PsiZ:

1. Install from PyPI using pip: ``pip install psiz``
2. Clone from Git Hub: https://github.com/roads/psiz.git

Note: PsiZ also requires TensorFlow, which is not installed automatically since is not included as a dependency of the PsiZ package (in setup.py). You must explicitly install the latest stable TensorFlow package (tensorflow or tensorflow-gpu). This allows users to specify whether they use a GPU enabled version of TensorFlow.


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
from psiz import datasets
from psiz.models import Exponential

# Load some observations (i.e., judged trials).
(obs, catalog) = datasets.load_dataset('birds-16')
# Initialize an embedding model.
emb = Exponential(catalog.n_stimuli)
# Fit the embedding model using similarity judgment observations.
emb.fit(obs)
# Optionally save the fitted model.
emb.save('my_embedding.h5')
```

## Trials and Observations
Inference is performed by fitting a model to a set of observations. In this package, a single observation is comprised of multiple stimuli that have been judged by an agent (human or machine) based on their similarity. 

In the simplest case, an observation is obtained from a trial consisting of three stimuli: a query stimulus (Q) and two reference stimuli (A and B). An agent selects the reference stimulus that they believe is more similar to the query stimulus. For this simple trial, there are two possible outcomes. If the agent selected reference A, then the observation for the ith trial would be recorded as the vector: 

D_i = [Q A B]

Alternatively, if the agent had selected reference B, the observation would be recorded as:

D_i = [Q B A]

In addition to a simple \emph{triplet} trial, this package is designed to handle a number of different trial configurations. A trial may have 2-8 reference stimuli and an agent may be required to select and rank more than one reference stimulus. 

## Common Use Cases
Optionally, you can also provide additional information.

1. The dimensionality of the embedding (default=2).
2. The number of unique population groups (default=1).

```python
n_stimuli = 100
emb = psiz.models.Exponential(n_stimuli, n_dim=4, n_group=2)
emb.fit(obs)
```

If you know some of the free parameters already, you can set them to the desired value and then make those parametres untrainable.
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
The name PsiZ (pronounced *sigh zeee*) is meant to serve as shorthard for the term *psychological embedding*. The greek letter Psi is often used to represent the field of psychology and the matrix variable **Z** is often used in machine learning to denote a latent feature space.

## Licence
This project is licensed under the Apache Licence 2.0 - see the LICENSE.txt file for details.

### References
* van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic triplet
   embedding. In Machine learning for signal processing (mlsp), 2012 IEEE
   international workshop on (p. 1-6). doi:10.1109/MLSP.2012.6349720
* Roads, B. D., & Mozer, M. C. (in press). Obtaining psychological
   embeddings through joint kernel and metric learning. Behavior Research
   Methods.
* Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). The
   Caltech-UCSD Birds-200-2011 Dataset (Tech. Rep. No. CNS-TR-2011-001).
   California Institute of Technology.
