#####################
Variational Inference
#####################

:Author: Brett D. Roads

Variational inference (VI) in PsiZ provides a practical way to learn
distribution-valued model parameters instead of single point estimates.
This supports uncertainty-aware embeddings and predictive behavior,
which is often useful when data are sparse, noisy, or actively collected.

What VI means in PsiZ
=====================

At a high level, VI optimizes an evidence lower bound (ELBO):

- Expected data fit (for example, categorical cross-entropy terms).
- A KL regularization term that keeps posteriors close to priors.

In PsiZ, this typically appears as variational embedding layers and
stochastic model components that register KL losses during training.

Why use VI
==========

Use VI when you need one or more of the following:

- Better calibration and uncertainty estimates over latent representations.
- Regularized fitting under limited observations.
- Hierarchical shrinkage across related groups of stimuli.
- Better behavior under active learning loops.

Common workflow
===============

Most VI workflows in PsiZ follow this pattern:

1. Define the percept/embedding architecture with variational layers.
2. Choose data batching and optimizer/scheduler settings.
3. Train while monitoring data-fit metrics and KL-informed objectives.
4. Evaluate predictive metrics and inspect uncertainty-sensitive behavior.

Related pages
=============

- For hierarchical construction patterns and component-level guidance, see
	:doc:`Hierarchical Variational Inference <hierarchical_vi>`.
- For hyperparameter strategy and stabilization, see
	:doc:`Tuning <tuning>`.
- For active data collection context, see
	:doc:`Active Learning <active_learning>`.