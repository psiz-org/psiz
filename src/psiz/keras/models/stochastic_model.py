# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Module of models.

Classes:
    StochasticModel:  An abstract Keras model that accomodates
        stochastic layers.

"""

import keras

if keras.backend.backend() == "tensorflow":
    import tensorflow as tf
elif keras.backend.backend() == "jax":
    pass
    # import jax  # TODO(roads) support jax backend.
elif keras.backend.backend() == "torch":
    import torch
else:
    raise RuntimeError(f"Unrecognized Keras Backend '{keras.backend.backend()}'.")


@keras.saving.register_keras_serializable(
    package="psiz.keras.models", name="StochasticModel"
)
class StochasticModel(keras.Model):
    """An abstract Keras model that accomodates stochastic layers.

    Incoming data is transformed by repeating all samples in the batch
    axis `n_sample` times for the forward pass. When `n_sample` is
    greater than 1, the computed losses and metrics are a better
    estimate of the expectation. As a side-effect, gradient updates
    tend to be smoother, reducing the risk of unstable training.

    When making predictions, an average across samples is returned.

    When calling the model in isolation via the `call` method, no
    modifications are made to the inputs.

    Attributes:
        n_sample: See `init` method.

    Methods:
        See `keras.Model` for inherited methods.
        repeat_samples_in_batch_axis: Transforms data structure by
            repeating all samples in the batch axis `n_sample` times.
        average_repeated_samples: Transforms data structure by
            averaging over repeated samples.
        disentangle_repeated_samples: Moves repeated samples to a new
            axis that has "repeated samples" semantics.

    """

    def __init__(self, n_sample=1, **kwargs):
        """Initialize.

        Args:
            n_sample (optional): A positive integer indicating the
                number of repeated samples in the batch axis. Only
                useful if using stochastic layers (e.g., variational
                models).
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super(StochasticModel, self).__init__(**kwargs)
        self._n_sample = int(n_sample)
        self._inputs_are_dict = None

    @property
    def n_sample(self):
        return self._n_sample

    @n_sample.setter
    def n_sample(self, n_sample):
        self._n_sample = int(n_sample)

    def call(self, inputs, training=None):
        """Call."""
        raise NotImplementedError(
            "Unimplemented `keras.StochasticModel.call()`: "
            "subclass `StochasticModel` with an overridden `call()` "
            " method."
        )

    def train_step(self, *args, **kwargs):
        """Logic for one training step.

        Args:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values that will be passed to
            `keras.callbacks.CallbackList.on_train_batch_end`.
            Typically, the values of the `Model`'s metrics are
            returned. Example: `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        if keras.backend.backend() == "jax":
            return self._jax_train_step(*args, **kwargs)
        elif keras.backend.backend() == "tensorflow":
            return self._tensorflow_train_step(*args, **kwargs)
        elif keras.backend.backend() == "torch":
            return self._torch_train_step(*args, **kwargs)

    def _jax_train_step(self, state, data):
        raise NotImplementedError("JAX backend not yet supported.")

    def _tensorflow_train_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        # Adjust `x`, `y` and `sample_weight` batch axis to reflect multiple
        # samples.
        x = self.repeat_samples_in_batch_axis(x, self._n_sample)
        y = self.repeat_samples_in_batch_axis(y, self._n_sample)
        if sample_weight is not None:
            sample_weight = self.repeat_samples_in_batch_axis(
                sample_weight, self._n_sample
            )

        # pylint: disable-next=used-before-assignment, possibly-used-before-assignment
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compute_loss(
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply(gradients, trainable_vars)

        # Update the metrics.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def _torch_train_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        # Adjust `x`, `y` and `sample_weight` batch axis to reflect multiple
        # samples.
        x = self.repeat_samples_in_batch_axis(x, self._n_sample)
        y = self.repeat_samples_in_batch_axis(y, self._n_sample)
        if sample_weight is not None:
            sample_weight = self.repeat_samples_in_batch_axis(
                sample_weight, self._n_sample
            )

        # Clear the leftover gradients.
        self.zero_grad()

        # Compute loss
        y_pred = self(x, training=True)
        loss = self.compute_loss(
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        # pylint: disable-next=used-before-assignment, possibly-used-before-assignment
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """The logic for one evaluation step.

        Standard prediction is performmed with one sample. To
        accommodate variational inference, the log probability of the
        data is computed by averaging over samples from the model:
        p(heldout | train) = int_model p(heldout|model) p(model|train)
                          ~= 1/n * sum_{i=1}^n p(heldout | model_i)
        where model_i is a draw from the posterior p(model|train).

        Args:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values that will be passed to
            `keras.callbacks.CallbackList.on_train_batch_end`.
            Typically, the values of the `Model`'s metrics are
            returned.

        """
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        # Adjust `x`, `y` and `sample_weight` batch axis to reflect multiple
        # samples.
        x = self.repeat_samples_in_batch_axis(x, self._n_sample)
        y = self.repeat_samples_in_batch_axis(y, self._n_sample)
        if sample_weight is not None:
            sample_weight = self.repeat_samples_in_batch_axis(
                sample_weight, self._n_sample
            )

        y_pred = self(x, training=False)

        loss = self.compute_loss(
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )

        # Update the metrics.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        """The logic for one inference step.

        Standard prediction is performmed with one sample. To
        accommodate variational inference, the predictions are averaged
        over multiple samples from the model.

        Args:
            data: A nested structure of `Tensor`s.

        Returns:
            The result of one forward pass step, typically the output of
            calling the `Model` on data.

        """
        x, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        x = self.repeat_samples_in_batch_axis(x, self._n_sample)
        y_pred = self(x, training=False)

        # For prediction, we average over the samples. The batch and
        # "repeated sample" axis are disentangled first to make averaging
        # simple.
        y_pred = self.average_repeated_samples(y_pred, self._n_sample)
        return y_pred

    def get_config(self):
        """Return model configuration."""
        config = super(StochasticModel, self).get_config()
        config.update(
            {
                "n_sample": self.n_sample,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def repeat_samples_in_batch_axis(self, data, n_sample):
        """Create repeated samples in batch axis.

        Each batch is repeated `n_sample` times. Repeated batch items
        occur in blocks. For example, if `n_sample=2` then each new
        `data` Tensor is structured like:
            `[batch_0, batch_0, batch_1, batch_1, ...]`.

        The block structure is leveraged later when `keras.ops.reshape`
        is used.

        Args:
            data: A data structure of Tensors. Can be a single Tensor,
                tuple of Tensors, or a single-level dictionary of
                Tensors.
            n_sample: Integer indicating the number of times to repeat.

        Returns:
            data: A Tensor or dictionary of Tensors that has been
                adjusted.

        """
        if isinstance(data, dict):
            new_data = {}
            for key in data:
                new_data[key] = keras.ops.repeat(data[key], repeats=n_sample, axis=0)
        elif isinstance(data, tuple):
            new_data = []
            for i_data in data:
                new_data.append(keras.ops.repeat(i_data, repeats=n_sample, axis=0))
            new_data = tuple(new_data)
        else:
            new_data = keras.ops.repeat(data, repeats=n_sample, axis=0)
        return new_data

    def average_repeated_samples(self, data, n_sample):
        """Average over repeated samples.

        Assumes `keras.ops.repeat` repitition rules were used to create
        repeated samples.

        Args:
            data: Data structure of Tensors. Can be a single Tensor or
                a single-level dictionary of Tensors.
            n_sample: Integer indicating the number of repeated
                samples.

        Returns:
            A new data structure of Tensors that is an average over
                repeated samples

        """
        if isinstance(data, dict):
            for key in data:
                val = self.disentangle_repeated_samples(data[key], n_sample)
                data[key] = keras.ops.mean(val, axis=1)
        else:
            data = self.disentangle_repeated_samples(data, n_sample)
            data = keras.ops.mean(data, axis=1)
        return data

    def disentangle_repeated_samples(self, data, n_sample):
        """Move repeated samples to new axis.

        Assumes `keras.ops.repeat` repitition rules were used to create
        repeated samples.

        Args:
            data: Tensor.
            n_sample: Integer indicating the number of repeated
                samples.

        Returns:
            A Tensor with a new "repated samples" axis at index=1.

        """
        new_shape = keras.ops.concatenate(
            [
                keras.ops.convert_to_tensor([-1], dtype="int32"),
                keras.ops.convert_to_tensor([n_sample], dtype="int32"),
                keras.ops.convert_to_tensor(keras.ops.shape(data)[1:], dtype="int32"),
            ],
            0,
        )
        return keras.ops.reshape(data, new_shape)
