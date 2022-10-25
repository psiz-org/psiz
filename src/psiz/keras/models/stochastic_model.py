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
    StochasticModel:  An abstract Keras model that accomodates stochastic
        layers via a "sample axis".

"""

import copy
import warnings

import tensorflow as tf
from tensorflow.python.eager import backprop

from psiz.keras.mixins.stochastic_mixin import StochasticMixin


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.models', name='StochasticModel'
)
class StochasticModel(tf.keras.Model):
    """An abstract Keras model that accomodates stochastic layers.

    When subclassing, your `call` method should first call
    `inputs = self.expand_inputs_with_sample_axis(inputs)`. This method
    will add a sample axis to all the input Tensors.

    The model assume that the `call` method returns a Tensor or
    dictionary of Tensors where each Tensor has a "sample axis", that
    represents `n_sample` samples. In `train_step` and `test_step`, the
    "sample axis" is combined with the "batch axis" to compute a
    batch-and-sample loss.

    Attributes:
        sample_axis: See `init` method.
        n_sample: See `init` method.
        preserved_inputs: See `init` method.

    Methods:
        See `tf.keras.Model` for inherited methods.
        expand_inputs_with_sample_axis: Expands `inputs` with a sample
            axis. Default behavior assumes `inputs` is a dictionary.
            The user can override the method to handle different
            input formats.

    """

    def __init__(
            self, sample_axis=None, n_sample=1, preserved_inputs=None,
            **kwargs):
        """Initialize.

        Args:
            sample_axis: Integer indicating which axis in the Tensor
                will serve as the "sample axis". Valid values are `1`
                or `2`.
            n_sample (optional): A positive integer indicating the
                number of samples that will populate the "sample axis".
                Only useful if using stochastic layers (e.g.,
                variational models).
            preserved_inputs (optional): List of dictionary keys
                indicating which values of `inputs` should not be
                expanded with a sample axis. Only applies if `inputs`
                is a dictionary. By default, no keys are preserved.
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super(StochasticModel, self).__init__(**kwargs)

        if sample_axis is None:
            raise TypeError(
                'You must provide an integer (1 or 2) indicating the '
                '`sample_axis`.'
            )
        if sample_axis not in [1, 2]:
            raise ValueError(
                'The `sample_axis` must be 1 or 2.'
            )
        self._sample_axis_outermost = int(sample_axis)
        self._n_sample = int(n_sample)

        if preserved_inputs is None:
            self.preserved_inputs = []
        else:
            self.preserved_inputs = preserved_inputs

        self._inputs_are_dict = None

    def build(self, input_shape):
        """Build."""
        # Propogate "stochastic settings" to all layers with `StochasticMixin`.
        self._set_stochastic_mixin(self.layers)

    @property
    def sample_axis(self):
        return self._sample_axis_outermost

    @property
    def n_sample(self):
        return self._n_sample

    # NOTE: There is no setter for `sample_axis` because (in general) the
    # model cannot be safely changed after instantiation. The layers may
    # make assumptions about the location of the `sample_axis`.

    @n_sample.setter
    def n_sample(self, n_sample):
        n_sample = int(n_sample)
        if n_sample != self._n_sample:
            self._n_sample = n_sample
            # Propogate change to children.
            self._set_stochastic_mixin(self.layers)

    def call(self, inputs, training=None):
        """Call.

        When subclassing, your `call` method should make the follwing
        call first:

        inputs = self.expand_inputs_with_sample_axis(inputs)

        """
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
        # NOTE: Inputs `x` is adjusted inside the `call` method.
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
        # NOTE: Inputs `x` is adjusted inside the `call` method.
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
        # NOTE: Inputs `x` is adjusted inside the `call` method.
        y_pred = self(x, training=False)
        # For prediction, we simply average over the sample axis.
        y_pred = self._mean_of_sample_axis(y_pred)
        return y_pred

    def get_config(self):
        """Return model configuration."""
        config = super(StochasticModel, self).get_config()
        config.update({
            'sample_axis': self.sample_axis,
            'n_sample': self.n_sample,
        })
        if len(self.preserved_inputs) != 0:
            config.update({
                'preserved_inputs': self.preserved_inputs
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def expand_inputs_with_sample_axis(self, inputs):
        """Expand input Tensor(s) with singleton 'sample axis'.

        NOTE: The default implementation assumes `inputs` is a
        dictionary. The user can override with method to accomodate
        non-dictionary `inputs`. For example, for `inputs` that is a
        single Tensor, the method would look like:

        ```
        def expand_inputs_with_sample_axis(self, inputs):
            inputs_with_sample_axis = tf.expand_dims(
                inputs, axis=self._sample_axis_outermost
            )
            return inputs_with_sample_axis
        ```

        Args:
            inputs

        Returns:
            inputs_with_sample_axis

        """
        inputs_with_sample_axis = copy.copy(inputs)
        # Pop keys that should be preserved, i.e., that should not have a
        # sample axis added.
        preserved_dict = {}
        for preserved_key in self.preserved_inputs:
            preserved_value = inputs_with_sample_axis.pop(
                preserved_key, None
            )
            if preserved_value is not None:
                preserved_dict[preserved_key] = preserved_value

        key_list = inputs_with_sample_axis.keys()
        for key in key_list:
            inputs_with_sample_axis[key] = tf.expand_dims(
                inputs_with_sample_axis[key],
                axis=self._sample_axis_outermost
            )
        # Recombine altered and preserved dictionaries.
        inputs_with_sample_axis = inputs_with_sample_axis | preserved_dict
        return inputs_with_sample_axis

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

        First the sample axis is moved to axis=1, keeping all other
        axes the same. Then a reshape operation is performed to combine
        axis=0 (batch) and axis=1 (sample). The argument "-1" is used
        to infer the shape of the new batch-sample axis and explicitly
        grabs the shape of the remaining axes.

        Args:
            y_pred: Tensor or dictionary of Tensors representing model
                predictions (i.e., outputs).

        Returns:
            y_pred: A Tensor or dictionary of Tensors that has been
                reshaped.

        """
        if isinstance(y_pred, dict):
            for key in y_pred:
                y_pred[key] = self._move_sample_axis_to_axis1(y_pred[key])
                y_pred[key] = self._combine_axis0_axis1(y_pred[key])
        else:
            y_pred = self._move_sample_axis_to_axis1(y_pred)
            y_pred = self._combine_axis0_axis1(y_pred)
        return y_pred

    def _move_sample_axis_to_axis1(self, x):
        """Move sample axis to axis 1.

        Args:
            x: Tensor

        Returns
            Tensor with sample axis at axis 1.

        """
        # MAYBE Write in a way that accomdates scenarios other than
        # sample_axis=1 or sample_axis=2. The trick is figuring out
        # how to write `new_order` in generic `sample_axis` way.
        if self.sample_axis == 2:
            # Only need to rearrange axes when `sample_axis!=1`.
            # NOTE: This implementation assumes sample_axis=2.
            new_order = tf.concat(
                (
                    tf.constant([0, 2, 1]), tf.range(tf.rank(x) - 3) + 3
                ), axis=0
            )
            x = tf.transpose(x, perm=new_order)
        return x

    def _combine_axis0_axis1(self, x):
        """Combine axis 0 and axis 1 of Tensor.

        Arguments:
            x: Tensor

        Returns:
            Reshaped Tensor with axis 0 and axis 1 combined.

        """
        new_shape = tf.concat([[-1], tf.shape(x)[2:]], 0)
        return tf.reshape(x, new_shape)

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
                    y_pred[key], axis=self._sample_axis_outermost
                )
        else:
            y_pred = tf.reduce_mean(y_pred, axis=self._sample_axis_outermost)
        return y_pred

    def _set_stochastic_mixin(self, layers, is_inside_rnn=False):
        """Build `StochasticMixin` for relevant layers.

        The objective of this method is to synchronize all layers that
        inherit from `StochasticMixin`. The method loops over all
        layers---while keeping track of the layer context (i.e., RNN
        context)---and sets layer-owned stochastic attributes.

        Args:
            layers: A list of Keras Layers.
            is_inside_rnn: Boolean indicating if layer context is
                inside an RNN Layer (i.e., an RNN cell).

        """
        for layer in layers:
            # Check if current Layer is an RNN and set `is_next_inside_rnn`
            # for next recursive call.
            if isinstance(layer, tf.keras.layers.RNN):
                is_next_inside_rnn = True
            else:
                is_next_inside_rnn = is_inside_rnn

            # Set `StochasticMixin` attributes of current Layer (if member).
            if isinstance(layer, StochasticMixin):
                layer.set_stochastic_mixin(
                    self._sample_axis_outermost, self._n_sample, is_inside_rnn
                )

            # Recurse to next level if current layer has children.
            if len(layer.submodules) > 0:
                self._set_stochastic_mixin(
                    layer.submodules, is_inside_rnn=is_next_inside_rnn
                )
