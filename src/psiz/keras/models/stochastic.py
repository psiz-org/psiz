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
"""Module of PsiZ models.

Classes:
    Stochastic:  An abstract Keras model that accomodates stochastic
        layers.

"""

import warnings

import tensorflow as tf
from tensorflow.python.eager import backprop


class Stochastic(tf.keras.Model):
    """An abstract Keras model that accomodates stochastic layers.

    These models assume that the `call` method returns a Tensor or
    dictionary of Tensors where each Tensor has a "sample" axis, that
    represents `n_sample` samples. In `train_step` and `test_step`, the
    "sample" axis is combined with the "batch" axis to compute a
    batch-and-sample loss.

    Attributes:
        sample_axis: Integer indicating sample axis.
        n_sample: Integer indicating the number of samples to draw for
            stochastic layers. Only useful if using stochastic layers
            (e.g., variational models).

    """

    def __init__(self, n_sample=1, **kwargs):
        """Initialize.

        Args:
            n_sample (optional): Integer indicating the number of
                samples to draw for stochastic layers. Only useful if
                using stochastic layers (e.g., variational models).
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super().__init__(**kwargs)
        self._sample_axis = 2
        self._n_sample = n_sample

    @property
    def sample_axis(self):
        return self._sample_axis

    @property
    def n_sample(self):
        return self._n_sample

    @n_sample.setter
    def n_sample(self, n_sample):
        self._n_sample = n_sample

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
        # Adjust `y` and `sample_weight` batch axis to reflect multiple
        # samples since `y_pred` has samples.
        y = self._repeat_samples_in_batch_axis(y)
        sample_weight = self._repeat_samples_in_batch_axis(sample_weight)

        # NOTE: During computation of gradients, IndexedSlices are
        # created which generates a TensorFlow warning. I cannot
        # find an implementation that avoids IndexedSlices. The
        # following catch environment silences the offending
        # warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, module=r'.*indexed_slices'
            )
            with backprop.GradientTape() as tape:
                y_pred = self(x, training=True)
                # Reshape `y_pred` samples axis into batch axis.
                y_pred = self._reshape_samples_into_batch(y_pred)
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
                if gradients[idx].__class__.__name__ == 'IndexedSlices':
                    gradients[idx] = tf.convert_to_tensor(
                        gradients[idx]
                    )

        self.optimizer.apply_gradients(zip(
            gradients, trainable_variables)
        )

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
        # Adjust `y` and `sample_weight` batch axis to reflect multiple
        # samples since `y_pred` has samples.
        y = self._repeat_samples_in_batch_axis(y)
        sample_weight = self._repeat_samples_in_batch_axis(sample_weight)

        y_pred = self(x, training=False)
        # Reshape `y_pred` samples axis into batch axis.
        y_pred = self._reshape_samples_into_batch(y_pred)

        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses
        )
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
        y_pred = self(x, training=False)
        # For prediction, we simply average over the sample axis.
        y_pred = self._mean_of_sample_axis(y_pred)
        return y_pred

    def get_config(self):
        """Return model configuration."""
        config = {
            'n_sample': self.n_sample,
        }
        return config

    def _repeat_samples_in_batch_axis(self, data):
        """Create "samples" for `y` or `sample_weight`.

        Each batch is repeated `n_sample` times. Repeated batch items
        occur in blocks. For example, if `n_sample=2` then each new
        `data` Tensor is structured like:
            `[batch_0, batch_0, batch_1, batch_1, ...]`.

        The block structure is leveraged later when `tf.reshape` is
        also used on `y_pred`, which also results in blocks of batch
        items.

        Args:
            data: A Tensor or dictionary of Tensors representing target
                outputs `y` or `sample_weight`.

        Returns:
            data: A Tensor or dictionary of Tensors that has been
                adjusted.

        """
        if isinstance(data, dict):
            for key in data:
                data[key] = tf.repeat(data[key], repeats=self.n_sample, axis=0)
        else:
            data = tf.repeat(data, repeats=self.n_sample, axis=0)
        return data

    def _reshape_samples_into_batch(self, y_pred):
        """Reshape sample axis into batch axis.

        The reshape operation combines the batch and sample axis such
        that samples for a given batch item occur in contiguous blocks.

        First the timestep axis and sample axis are swapped, keeping
        all other axes the same. Then a reshape operation uses "-1" to
        infer the shape of the new batch-sample axis and explicitly
        grabs the shape of the remaining axes.

        Args:
            y_pred: Tensor or dictionary of Tensors representing model
                predictions (i.e., outputs).

        Returns:
            y_pred: A Tensor or dictionary of Tensors that has been
                reshaped.

        """
        # TODO Write `new_order` in generic `sample_axis` way.
        if isinstance(y_pred, dict):
            for key in y_pred:
                new_order = tf.concat(
                    (
                        tf.constant([0, 2, 1]),
                        tf.range(tf.rank(y_pred[key]) - 3) + 3
                    ), axis=0
                )
                y_pred[key] = tf.transpose(y_pred[key], perm=new_order)
                new_shape = tf.concat(
                    [[-1], tf.shape(y_pred[key])[2:]], 0
                )
                y_pred[key] = tf.reshape(y_pred[key], new_shape)
        else:
            new_order = tf.concat(
                (tf.constant([0, 2, 1]), tf.range(tf.rank(y_pred) - 3) + 3),
                axis=0
            )
            y_pred = tf.transpose(y_pred, perm=new_order)
            new_shape = tf.concat([[-1], tf.shape(y_pred)[2:]], 0)
            y_pred = tf.reshape(y_pred, new_shape)
        return y_pred

    def _mean_of_sample_axis(self, y_pred):
        """Take the mean over the sample axis.

        The axis=1 of the Tensor returned from calling the model is
        assumed to be a "sample" axis. If a singleton dimension, taking
        the mean is equivalent to a squeeze operation.

        Args:
            y_pred: Tensor predictions of unkown structure that have an
                extra `n_sample` dimension.

        Returns:
            y_pred: Tensor predictions with sample dimension correctly
                handled for different structures.

        """
        if isinstance(y_pred, dict):
            for key in y_pred:
                y_pred[key] = tf.reduce_mean(
                    y_pred[key], axis=self._sample_axis
                )
        else:
            y_pred = tf.reduce_mean(y_pred, axis=self._sample_axis)
        return y_pred
