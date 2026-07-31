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

Dataset strategy note (v0.14)
=============================

For new training pipelines, prefer PsiZ dataset artifacts and runtime ingestion:

* :code:`psiz.data.Dataset.save(...)`
* :code:`psiz.data.load(...)`
* Optional Tier A adapters such as :code:`dataset.tensorflow()`

Some existing tutorials still rely on :code:`psiz-datasets` TensorFlow assets.
Those notebook paths remain temporarily legacy and will be migrated after
upstream dataset assets adopt the v0.14 artifact format.

Checkpoint and export workflow
==============================

Use Keras-native checkpoints during training, then export a finalized PsiZ
artifact once training is complete.

.. code-block:: python

	from pathlib import Path

	import keras
	import psiz

	checkpoint_path = Path("checkpoints") / "best.model.keras"
	callbacks = [
		keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_path,
			monitor="val_loss",
			save_best_only=True,
			save_weights_only=False,
			mode="min",
		)
	]

	model.fit(train_ds, validation_data=val_ds, callbacks=callbacks)

	# Export durable/shareable artifact after training.
	psiz.keras.save_psiz_model(model, "release_model.psiz")

This preserves robust resume behavior during training while producing a
portable artifact for long-term storage and sharing.

Related pages
=============

- For hierarchical construction patterns and component-level guidance, see
	:doc:`Hierarchical Variational Inference <hierarchical_vi>`.
- For hyperparameter strategy and stabilization, see
	:doc:`Tuning <tuning>`.
- For active data collection context, see
	:doc:`Active Learning <active_learning>`.