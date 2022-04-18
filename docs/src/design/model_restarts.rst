##############
Model Restarts
##############

:Author: Brett D. Roads

Motivation
==========

Unfortunately, embeddings are susceptible to discovering local optima during inference. While the usual tricks help, such as stochastic gradient decent, one often requires multiple restarts with different initializations to be confident in the solution.

Solution
========

You can use a hypertuning package such as Keras Tuner to manage multiple restarts or executions.