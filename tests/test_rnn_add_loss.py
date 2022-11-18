# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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
"""Test RNN `fit` method that uses `add_loss`."""

import pytest
import tensorflow as tf
# from tensorflow.python.eager import backprop


class FooModel(tf.keras.Model):
    """A basic model for testing.

    Attributes:
        cell: The RNN cell layer.

    """

    def __init__(self, rnn=None, **kwargs):
        """Initialize.

        Args:
            cell: A Keras layer.
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super().__init__(**kwargs)

        # Assign layers.
        self.rnn = rnn

    def build(self, input_shape):
        """Build."""
        self.rnn.build(input_shape)

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
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """Compute the total loss, validate it, and return it."""
        del x  # The default implementation does not use `x`.
        loss = self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses
        )
        return loss

    def call(self, inputs, training=None):
        """Call.

        Args:
            inputs: A dictionary of inputs.
            training (optional): Boolean indicating if training mode.

        """
        output = self.rnn(inputs, training=training)
        return output


class FooModel2(tf.keras.Model):
    """A basic model for testing.

    Attributes:
        cell: The RNN cell layer.

    """

    def __init__(self, cell=None, **kwargs):
        """Initialize.

        Args:
            cell: A Keras layer.
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super().__init__(**kwargs)

        # Assign layers.
        self.cell = cell

    def call(self, inputs, training=None):
        """Call.

        Args:
            inputs: A dictionary of inputs.
            training (optional): Boolean indicating if training mode.

        """
        states = tf.constant([0., 0., 0., 0., 0.])
        output, states_tplus1 = self.cell(inputs, states, training=training)
        return output

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
        # x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        # with backprop.GradientTape() as tape:
        #     y_pred = self(x, training=True)
        #     loss = self.compiled_loss(
        #         y, y_pred, sample_weight, regularization_losses=self.losses
        #     )

        # # Custom training steps:
        # trainable_variables = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_variables)
        # self.optimizer.apply_gradients(zip(
        #     gradients, trainable_variables)
        # )

        # self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # return {m.name: m.result() for m in self.metrics}
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)


class MyRegularizer(tf.keras.regularizers.Regularizer, tf.Module):
    def __init__(self):
        """Initialize."""
        super(MyRegularizer, self).__init__()
        self._kl_divergence_fn = _make_kl_divergence_fn()

    def __call__(self, x):
        """Call."""
        return self._kl_divergence_fn(x)


def _make_kl_divergence_fn():
    def _fn(x):
        x = tf.reduce_sum(x, name='kl_reduce')
        return tf.add(x, tf.constant(1.0), name='kl_add')
    return _fn


class BarCell(tf.keras.layers.Layer):
    """RNN cell for testing."""
    def __init__(self, **kwargs):
        """Initialize.

        Args:

        """
        super(BarCell, self).__init__(**kwargs)

        # Satisfy RNNCell contract.
        self.state_size = [tf.TensorShape([1])]
        self.output_size = tf.TensorShape([1])

        self._regularizer = MyRegularizer()
        self._extra_variables = self._regularizer.variables

    # def build(self, input_shape):
    #     """Build."""
    #     # super(BarCell, self).build(input_shape)

    # def compute_output_shape(self, input_shape):
    #     """Compute output shape."""
    #     x = None

    def call(self, inputs, states, training=None):
        """Call."""
        # self.add_loss(tf.reduce_sum(inputs))
        # self.add_loss(tf.reduce_sum(inputs), inputs=inputs)
        # self.add_loss(lambda: tf.reduce_sum(inputs)) Does not work
        kl_loss = self._regularizer(inputs)
        self.add_loss(kl_loss)

        output = tf.reduce_sum(inputs, axis=1) + tf.constant(1.0)
        output = tf.expand_dims(output, axis=1)
        states_tplus1 = [states[0] + 1]
        return output, states_tplus1


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_nornn_fit_with_add_loss(is_eager):
    """Test fit method (triggering backprop)."""
    tf.config.run_functions_eagerly(is_eager)

    # Some dummy input formatted as a TF Dataset.
    n_example = 5
    x = tf.constant([
        [1, 2, 3],
        [1, 13, 8],
        [1, 5, 6],
        [1, 5, 12],
        [1, 5, 6],
    ], dtype=tf.float32)
    y = tf.constant(
        [
            [1],
            [10],
            [4],
            [4],
            [4],
        ], dtype=tf.float32
    )
    tfds = tf.data.Dataset.from_tensor_slices((x, y))
    tfds = tfds.batch(n_example, drop_remainder=False)

    # A minimum model to reproduce the issue.
    bar_cell = BarCell()
    model = FooModel2(cell=bar_cell)
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
    }
    model.compile(**compile_kwargs)

    # Call fit which will trigger gradient computations and raise an error
    # during graph execution.
    model.fit(tfds, epochs=1)


@pytest.mark.parametrize(
    "is_eager,unroll",
    [
        (True, True),
        (True, False),
        (False, True),
        pytest.param(False, False, marks=pytest.mark.xfail)
    ]
)
def test_withrnn_fit_with_add_loss(is_eager, unroll):
    """Test fit method (triggering backprop)."""
    tf.config.run_functions_eagerly(is_eager)

    # Some dummy input formatted as a TF Dataset.
    x = tf.constant([
        [[1, 2, 3], [2, 0, 0], [3, 0, 0], [4, 3, 4]],
        [[1, 13, 8], [2, 0, 0], [3, 0, 0], [4, 13, 8]],
        [[1, 5, 6], [2, 8, 0], [3, 16, 0], [0, 0, 0]],
        [[1, 5, 12], [2, 14, 15], [3, 17, 18], [4, 5, 6]],
        [[1, 5, 6], [2, 14, 15], [3, 17, 18], [4, 5, 6]],
        [[1, 5, 6], [2, 14, 15], [3, 17, 18], [0, 0, 0]],
    ], dtype=tf.float32)
    y = tf.constant(
        [
            [[1], [2], [1], [2]],
            [[10], [2], [1], [7]],
            [[4], [2], [6], [0]],
            [[4], [2], [1], [2]],
            [[4], [2], [1], [2]],
            [[4], [2], [1], [0]],
        ], dtype=tf.float32
    )
    w = tf.constant(
        [
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 0.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 0.],
        ], dtype=tf.float32
    )
    tfds = tf.data.Dataset.from_tensor_slices((x, y, w))
    tfds = tfds.batch(3, drop_remainder=False)

    # A minimum model to reproduce the issue.
    cell = BarCell()
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, unroll=unroll)
    model = FooModel(rnn=rnn)
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.losses.MeanSquaredError(name='mse')
        ],
    }
    model.compile(**compile_kwargs)

    # Call fit which will trigger gradient computations and raise an error
    # during graph execution.
    model.fit(tfds, epochs=1)
