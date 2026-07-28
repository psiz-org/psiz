# PsiZ Examples

If you are looking for a place to start, we recomend `rank/mle_1g.py`. This example demonstrates the simplest use case.

Before running examples, set a Keras backend and ensure matching backend dependencies are installed.
For example:
* `export KERAS_BACKEND=tensorflow`
* `export KERAS_BACKEND=torch`
* `export KERAS_BACKEND=jax`

The examples are organized into separate directories based on the type of behavioural data collected (i.e., trial type).
* `rank`
* `rate`

Where appropriate, part of the filename indicates the type of inference performed.
* `mle` indicates maximum liklihood inference which yields a point estimate model.
* `vi` indicates variational inference which yields a posterior probability model.

Storage note:
* Canonical `.psiz` save/load usage in examples and scripts should use function APIs:
	`psiz.keras.save_psiz_model(...)` and `psiz.keras.load_psiz_model(...)`.
* Method-based `.save_psiz()` and `.load_psiz()` calls are optional convenience
	for `psiz.keras.StochasticModel` subclasses.
* For in-training resume checkpoints, prefer Keras-native checkpointing; use
	`.psiz` for finalized assets intended for durable sharing and reuse.
