# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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

import warnings

import tensorflow as tf
from tensorflow.python.eager import backprop


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.models", name="StochasticModel"
)
class StochasticModel(tf.keras.Model):
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
        See `tf.keras.Model` for inherited methods.
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
            "Unimplemented `tf.keras.StochasticModel.call()`: "
            "subclass `StochasticModel` with an overridden `call()` "
            " method."
        )

    def train_step(self, data):
        """Logic for one training step.

        Args:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`.
            Typically, the values of the `Model`'s metrics are
            returned. Example: `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Adjust `x`, `y` and `sample_weight` batch axis to reflect multiple
        # samples.
        x = self.repeat_samples_in_batch_axis(x, self._n_sample)
        y = self.repeat_samples_in_batch_axis(y, self._n_sample)
        if sample_weight is not None:
            sample_weight = self.repeat_samples_in_batch_axis(
                sample_weight, self._n_sample
            )

        # NOTE: During computation of gradients, IndexedSlices are
        # created which generates a TensorFlow warning. I cannot
        # find an implementation that avoids IndexedSlices. The
        # following catch environment silences the offending
        # warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, module=r".*indexed_slices"
            )
            with backprop.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(
                    y, y_pred, sample_weight, regularization_losses=self.losses
                )

            # Custom training steps:
            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            # NOTE: There is an open issue for using constraints with
            # embedding-like layers (e.g., tf.keras.layers.Embedding)
            # see:
            # https://github.com/tensorflow/tensorflow/issues/33755.
            # There are also issues when using Eager Execution. A
            # work-around is to convert the problematic gradients, which
            # are returned as tf.IndexedSlices, into dense tensors.
            for idx in range(len(gradients)):
                if gradients[idx].__class__.__name__ == "IndexedSlices":
                    gradients[idx] = tf.convert_to_tensor(gradients[idx])

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
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
            `tf.keras.callbacks.CallbackList.on_train_batch_end`.
            Typically, the values of the `Model`'s metrics are
            returned.

        """
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Adjust `x`, `y` and `sample_weight` batch axis to reflect multiple
        # samples.
        x = self.repeat_samples_in_batch_axis(x, self._n_sample)
        y = self.repeat_samples_in_batch_axis(y, self._n_sample)
        if sample_weight is not None:
            sample_weight = self.repeat_samples_in_batch_axis(
                sample_weight, self._n_sample
            )

        y_pred = self(x, training=False)

        self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
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
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
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

        The block structure is leveraged later when `tf.reshape` is
        used.

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
                new_data[key] = tf.repeat(data[key], repeats=n_sample, axis=0)
        elif isinstance(data, tuple):
            new_data = []
            for i_data in data:
                new_data.append(tf.repeat(i_data, repeats=n_sample, axis=0))
            new_data = tuple(new_data)
        else:
            new_data = tf.repeat(data, repeats=n_sample, axis=0)
        return new_data

    def average_repeated_samples(self, data, n_sample):
        """Average over repeated samples.

        Assumes `tf.repeat` repitition rules were used to create
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
                data[key] = tf.reduce_mean(val, axis=1)
        else:
            data = self.disentangle_repeated_samples(data, n_sample)
            data = tf.reduce_mean(data, axis=1)
        return data

    def disentangle_repeated_samples(self, data, n_sample):
        """Move repeated samples to new axis.

        Assumes `tf.repeat` repitition rules were used to create
        repeated samples.

        Args:
            data: Tensor.
            n_sample: Integer indicating the number of repeated
                samples.

        Returns:
            A Tensor with a new "repated samples" axis at index=1.

        """
        new_shape = tf.concat([[-1], [n_sample], tf.shape(data)[1:]], 0)
        return tf.reshape(data, new_shape)
