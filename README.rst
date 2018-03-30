=======================================
PsiZ: A Psychological Embedding Package
=======================================

Purpose
-------
PsiZ provides the computational tools to infer an continuous, multivariate
representation for a set of stimuli using ordinal similarity relations.

What's in a name?
-----------------
The name PsiZ (pronounced like the word *size*) is meant to serve as shorthard
for the term *psychological embedding*. The greek letter Psi is often used to
represent the field of psychology and the matrix variable **Z** is often used in
computer vision to denote an embedding.

Getting Started
---------------
In order to infer an embedding, you must provide two pieces of information.

   1. The number of unique stimuli.
   2. The similarity judgment observations.

```
n_stimuli = 100
embedding = psiz.models.Exponential(n_stimuli=n_stimuli)
embedding.fit(obs)
```
Optionally, you can also provide additional information.

   1. The dimensionality of the embedding (default=2).
   2. The number of unique population groups (default=1).

```
n_stimuli = 100
n_dim = 4
n_group = 2
embedding = psiz.models.Exponential(n_stimuli=n_stimuli, n_dim=n_dim, n_group=n_group)
embedding.fit(obs)
```

Common Use Cases:
know free parameters (set)
don’t know free parameters (fit)

Similarity Judgment Observations
--------------------------------
To infer an embedding, multiple observations are necessary. A single 
observation is comprised of a multiple stimuli that have been judged by an 
agent (human or machine) based on their similarity. 

In the simplest case, an observation is made for three stimuli: a query
stimulus (Q) and two reference stimuli (A and B). The agent is tasked with selecting the 
reference stimulus that they believe is more similar to the query stimulus.
If the agent selected reference A, then the observation would be recorded as;

D_i = [Q A B]

If the agent had selected reference B, the observation would be recorded as:

D_i = [Q B A]

The simplest observation is a triplet of the form:
Query: Reference A > Reference B

This package is designed to handle a number of different observations.

[vanderMaaten]_, [Wah2011]_, [RoadsA]_,

Embedding Models
----------------

Modules
-------
``models``
``dimensionality``
``visualize``
``utils``

Guiding principles
------------------

Installation
------------
There are two ways to install PsiZ:

   1. Install from PyPI using pip: ``pip install psiz``
   2. Clone from Git Hub: https://github.com/roads/psiz.git

Support
-------

Authors
-------
- Brett D. Roads
- Michael C. Mozer
- See also the list of contributors who participated in this project.

Licence
-------
This project is licensed under the GNU GPLv3 License - see the LICENSE.txt file for details.

.. [vanderMaaten] van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic triplet
   embedding. In Machine learning for signal processing (mlsp), 2012 IEEE
   international workshop on (p. 1-6). doi:10.1109/MLSP.2012.6349720
.. [RoadsA] Roads, B. D., & Mozer, M. C. (in preparation). Obtaining psychological
   embeddings through joint kernel and metric learning.
.. [Wah2011] Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). The
   Caltech-UCSD Birds-200-2011 Dataset (Tech. Rep. No. CNS-TR-2011-001).
   California Institute of Technology.
