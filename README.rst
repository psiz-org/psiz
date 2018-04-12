=======================================
PsiZ: A Psychological Embedding Package
=======================================

Purpose
-------
PsiZ provides the computational tools to infer a continuous, multivariate
representation for a set of stimuli using ordinal similarity relations.

What's in a name?
-----------------
The name PsiZ (pronounced like the word *size*) is meant to serve as shorthard
for the term *psychological embedding*. The greek letter Psi is often used to
represent the field of psychology and the matrix variable **Z** is often used in
computer vision to denote an embedding.

Quick Start
-----------
There are three built-in embedding models to choose from:

   1. Exponential
   2. HeavyTailed
   3. StudentsT

Once you have selected an embedding model, you must provide two pieces of information
in order to infer an embedding.

   1. The similarity judgment observations.
   2. The number of unique stimuli.

.. code-block:: python

  # Load some example observations.
  (obs, n_stimuli) = datasets.load_obs('birds-16') TODO
  # Initialize an embedding model.
  embedding = psiz.models.Exponential(n_stimuli)
  # Fit the embedding model using observations.
  embedding.fit(obs)

Similarity Judgment Trial TODO
-------------------------
To infer an embedding, multiple observations are necessary. A single 
observation is comprised of multiple stimuli that have been judged by an 
agent (human or machine) based on their similarity. 

In the simplest case, an observation is obtained from three stimuli: a query
stimulus (Q) and two reference stimuli (A and B). An agent selects the 
reference stimulus that they believe is more similar to the query stimulus.
If the agent selected reference A, then the observation would be recorded as
the vector: 

D_i = [Q A B]

If the agent had selected reference B, the observation would be recorded as:

D_i = [Q B A]

The simplest observation is a triplet of the form:
Query: Reference A > Reference B

This package is designed to handle a number of different observations.

[vanderMaaten]_, [Wah2011]_, [RoadsA]_,

Common Use Cases
----------------
Optionally, you can also provide additional information.

   1. The dimensionality of the embedding (default=2).
   2. The number of unique population groups (default=1).

.. code-block:: python
  
  n_stimuli = 100
  n_dim = 4
  n_group = 2
  embedding = psiz.models.Exponential(n_stimuli=n_stimuli, n_dim=n_dim, n_group=n_group)
  embedding.fit(obs)


know free parameters (set)
don’t know free parameters (fit)

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
