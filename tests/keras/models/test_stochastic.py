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
"""Module for testing models.py."""

import pytest
import tensorflow as tf

from psiz.keras.mixins.stochastic_mixin import StochasticMixin
from psiz.keras.models.stochastic import Stochastic


class CustomLayerA(StochasticMixin, tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomLayerA, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = tf.keras.initializers.Constant(1.)

    def build(self, input_shape):
        """Build."""
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, training=False):
        x = tf.matmul(a=inputs[:, 0], b=self.kernel)
        x = tf.expand_dims(x, axis=self.sample_axis)
        x = tf.repeat(x, self.n_sample, axis=self.sample_axis)
        return x

    def get_config(self):
        config = super(CustomLayerA, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomLayerB(StochasticMixin, tf.keras.layers.Layer):
    """A simple repeat layer for testing."""

    def __init__(self, **kwargs):
        """Initialize."""
        super(CustomLayerB, self).__init__(**kwargs)
        self.w0_initializer = tf.keras.initializers.Constant(1.)

    def build(self, input_shape):
        """Build."""
        self.w0 = self.add_weight(
            "w0",
            shape=[],
            initializer=self.w0_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, training=None):
        """Call."""
        # NOTE: Attributes `n_sample` and `sample_axis` provided by mixin.
        return self.w0 * tf.repeat(
            inputs, self.n_sample, axis=self.sample_axis
        )

    def get_config(self):
        return super(CustomLayerB, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomModelControl(tf.keras.Model):
    """A non-stochastic model to use as a control case."""
    def __init__(self):
        super(CustomModelControl, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(3)

    def call(self, inputs):
        x = inputs
        x = self.dense_layer(x)
        return x

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomModelA(Stochastic):
    """A stochastic model with no custom layers."""
    def __init__(self, **kwargs):
        super(CustomModelA, self).__init__(**kwargs)
        self.dense_layer = tf.keras.layers.Dense(3)

    def call(self, inputs):
        x = self.dense_layer(inputs)
        x = tf.repeat(
            x, self.n_sample, axis=self.sample_axis
        )
        return x

    def get_config(self):
        return super(CustomModelA, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomModelB(Stochastic):
    """A stochastic model with a custom layer.

    Assumes single tensor input.

    """
    def __init__(self, **kwargs):
        super(CustomModelB, self).__init__(**kwargs)
        self.dense_layer = CustomLayerA(3)

    def call(self, inputs):
        x = self.dense_layer(inputs)
        return x

    def get_config(self):
        """Return model configuration."""
        return super(CustomModelB, self).get_config()

    @classmethod
    def from_config(cls, config):
        """Create model from configuration.

        Args:
            config: A hierarchical configuration dictionary.

        Returns:
            An instantiated and configured TensorFlow model.

        """
        return cls(**config)


class CustomModelC(Stochastic):
    """A stochastic model with a custom layer.

    Assumes dictionary of tensors input.

    """

    def __init__(self, **kwargs):
        """Initialize."

        Args:
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super(CustomModelC, self).__init__(**kwargs)
        self.branch_0 = CustomLayerB()
        self.branch_1 = CustomLayerB()
        self.add_layer = tf.keras.layers.Add()

    def call(self, inputs, training=None):
        """Call.

        Args:
            inputs: A dictionary of inputs.
            training (optional): Boolean indicating if training mode.

        """
        x0 = inputs['x0']
        x1 = inputs['x1']
        # Execute two branches using custom layers with `StochasticMixin`.
        x0 = self.branch_0(x0, training=training)
        x1 = self.branch_1(x1, training=training)
        return self.add_layer([x0, x1], training=training)

    def get_config(self):
        """Return model configuration."""
        return super(CustomModelC, self).get_config()

    @classmethod
    def from_config(cls, config):
        """Create model from configuration.

        Args:
            config: A hierarchical configuration dictionary.

        Returns:
            An instantiated and configured TensorFlow model.

        """
        return cls(**config)


@pytest.fixture(scope="module")
def ds_rank2_x0():
    """Rank observations dataset."""
    n_example = 6
    x0 = tf.constant([
        [0.1, 1.1, 2.1],
        [0.2, 1.2, 2.2],
        [0.3, 1.3, 2.3],
        [0.4, 1.4, 2.4],
        [0.5, 1.5, 2.5],
        [0.6, 1.6, 2.6]
    ], dtype=tf.float32)
    x = x0
    y = tf.constant([
        [10.1, 11.1, 12.1],
        [10.2, 11.2, 12.2],
        [10.3, 11.3, 12.3],
        [10.4, 11.4, 12.4],
        [10.5, 11.5, 12.5],
        [10.6, 11.6, 12.6]
    ], dtype=tf.float32)

    w = tf.constant([1., 1., 0.2, 1., 1., 0.8], dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, y, w))
    ds = ds.batch(n_example, drop_remainder=False)

    inputs_shape = tf.TensorShape(x.shape)

    return {'ds': ds, 'inputs_shape': inputs_shape}


@pytest.fixture(scope="module")
def ds_rank2_x0_x1_x2():
    """Rank observations dataset."""
    n_example = 6
    x0 = tf.constant([
        [0.1, 1.1, 2.1],
        [0.2, 1.2, 2.2],
        [0.3, 1.3, 2.3],
        [0.4, 1.4, 2.4],
        [0.5, 1.5, 2.5],
        [0.6, 1.6, 2.6]
    ], dtype=tf.float32)
    x1 = tf.constant([
        [10.1, 11.1, 12.1],
        [10.2, 11.2, 12.2],
        [10.3, 11.3, 12.3],
        [10.4, 11.4, 12.4],
        [10.5, 11.5, 12.5],
        [10.6, 11.6, 12.6]
    ], dtype=tf.float32)
    x2 = tf.constant([
        [20.1, 21.1, 22.1],
        [20.2, 21.2, 22.2],
        [20.3, 21.3, 22.3],
        [20.4, 21.4, 22.4],
        [20.5, 21.5, 22.5],
        [20.6, 21.6, 22.6]
    ], dtype=tf.float32)

    x = {
        'x0': x0,
        'x1': x1,
        'x2': x2,
    }
    y = tf.constant([
        [10.1, 11.1, 12.1],
        [10.2, 11.2, 12.2],
        [10.3, 11.3, 12.3],
        [10.4, 11.4, 12.4],
        [10.5, 11.5, 12.5],
        [10.6, 11.6, 12.6]
    ], dtype=tf.float32)

    w = tf.constant([1., 1., 0.2, 1., 1., 0.8], dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, y, w))
    ds = ds.batch(n_example, drop_remainder=False)

    inputs_shape = {
        'x0': tf.TensorShape(x0.shape),
        'x1': tf.TensorShape(x1.shape),
        'x2': tf.TensorShape(x2.shape),
    }

    return {'ds': ds, 'inputs_shape': inputs_shape}


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_mse_model_ax1(ds_rank2_x0_x1_x2, is_eager):
    """Test MSE model with sample_axis=1."""
    tf.config.run_functions_eagerly(is_eager)
    ds = ds_rank2_x0_x1_x2['ds']
    inputs_shape = ds_rank2_x0_x1_x2['inputs_shape']

    model = CustomModelC(sample_axis=1, n_sample=2)
    model.build(inputs_shape)

    # Explicilty check that `sample_axis` and `n_sample` attributes of child
    # layers were set correctly.
    assert model.sample_axis == 1
    assert model.n_sample == 2
    assert model.branch_0.sample_axis == 1
    assert model.branch_0.n_sample == 2
    assert model.branch_1.sample_axis == 1
    assert model.branch_1.n_sample == 2

    x0_desired = tf.constant([
        [[0.1, 1.1, 2.1]],
        [[0.2, 1.2, 2.2]],
        [[0.3, 1.3, 2.3]],
        [[0.4, 1.4, 2.4]],
        [[0.5, 1.5, 2.5]],
        [[0.6, 1.6, 2.6]]
    ], dtype=tf.float32)
    x1_desired = tf.constant([
        [[10.1, 11.1, 12.1]],
        [[10.2, 11.2, 12.2]],
        [[10.3, 11.3, 12.3]],
        [[10.4, 11.4, 12.4]],
        [[10.5, 11.5, 12.5]],
        [[10.6, 11.6, 12.6]]
    ], dtype=tf.float32)
    x2_desired = tf.constant([
        [[20.1, 21.1, 22.1]],
        [[20.2, 21.2, 22.2]],
        [[20.3, 21.3, 22.3]],
        [[20.4, 21.4, 22.4]],
        [[20.5, 21.5, 22.5]],
        [[20.6, 21.6, 22.6]]
    ], dtype=tf.float32)

    y_desired = tf.constant([
        [10.1, 11.1, 12.1],
        [10.1, 11.1, 12.1],
        [10.2, 11.2, 12.2],
        [10.2, 11.2, 12.2],
        [10.3, 11.3, 12.3],
        [10.3, 11.3, 12.3],
        [10.4, 11.4, 12.4],
        [10.4, 11.4, 12.4],
        [10.5, 11.5, 12.5],
        [10.5, 11.5, 12.5],
        [10.6, 11.6, 12.6],
        [10.6, 11.6, 12.6]
    ], dtype=tf.float32)

    sample_weight_desired = tf.constant(
        [1., 1., 1., 1., 0.2, 0.2, 1., 1., 1., 1., 0.8, 0.8], dtype=tf.float32
    )

    y_pred_desired = tf.constant([
        [10.2, 12.2, 14.2],
        [10.2, 12.2, 14.2],
        [10.4, 12.4, 14.4],
        [10.4, 12.4, 14.4],
        [10.6, 12.6, 14.6],
        [10.6, 12.6, 14.6],
        [10.8, 12.8, 14.8],
        [10.8, 12.8, 14.8],
        [11., 13., 15.],
        [11., 13., 15.],
        [11.2, 13.2, 15.2],
        [11.2, 13.2, 15.2]
    ], dtype=tf.float32)

    # Check "sample axis" added correctly to inputs.
    for data in ds:
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        x = model.expand_inputs_with_sample_axis(x)
        tf.debugging.assert_equal(x['x0'], x0_desired)
        tf.debugging.assert_equal(x['x1'], x1_desired)
        tf.debugging.assert_equal(x['x2'], x2_desired)

    # Perform a `test_step`.
    for data in ds:
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Adjust `x` to include singleton "sample axis".
        x = model.expand_inputs_with_sample_axis(x)
        # Adjust `y` and `sample_weight` batch axis to reflect multiple
        # samples since `y_pred` has samples.
        y = model._repeat_samples_in_batch_axis(y)
        sample_weight = model._repeat_samples_in_batch_axis(sample_weight)
        # Assert `y` and `sample_weight` handled correctly.
        tf.debugging.assert_equal(y, y_desired)
        tf.debugging.assert_equal(sample_weight, sample_weight_desired)

        y_pred = model(x, training=False)
        # Reshape `y_pred` samples axis into batch axis.
        y_pred = model._reshape_samples_into_batch(y_pred)
        # Assert `y_pred` handled correctly.
        tf.debugging.assert_near(y_pred, y_pred_desired)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_mse_model_ax2(ds_rank2_x0_x1_x2, is_eager):
    """Test MSE model with sample_axis=2."""
    tf.config.run_functions_eagerly(is_eager)

    ds = ds_rank2_x0_x1_x2['ds']
    inputs_shape = ds_rank2_x0_x1_x2['inputs_shape']
    model = CustomModelC(sample_axis=2, n_sample=2)
    model.build(inputs_shape)

    # Explicilty check that `sample_axis` and `n_sample` attributes of child
    # Layer were set correctly.
    assert model.sample_axis == 2
    assert model.n_sample == 2
    assert model.branch_0.sample_axis == 2
    assert model.branch_0.n_sample == 2
    assert model.branch_1.sample_axis == 2
    assert model.branch_1.n_sample == 2

    x0_desired = tf.constant([
        [[0.1], [1.1], [2.1]],
        [[0.2], [1.2], [2.2]],
        [[0.3], [1.3], [2.3]],
        [[0.4], [1.4], [2.4]],
        [[0.5], [1.5], [2.5]],
        [[0.6], [1.6], [2.6]]
    ], dtype=tf.float32)
    x1_desired = tf.constant([
        [[10.1], [11.1], [12.1]],
        [[10.2], [11.2], [12.2]],
        [[10.3], [11.3], [12.3]],
        [[10.4], [11.4], [12.4]],
        [[10.5], [11.5], [12.5]],
        [[10.6], [11.6], [12.6]]
    ], dtype=tf.float32)
    x2_desired = tf.constant([
        [[20.1], [21.1], [22.1]],
        [[20.2], [21.2], [22.2]],
        [[20.3], [21.3], [22.3]],
        [[20.4], [21.4], [22.4]],
        [[20.5], [21.5], [22.5]],
        [[20.6], [21.6], [22.6]]
    ], dtype=tf.float32)

    y_desired = tf.constant([
        [10.1, 11.1, 12.1],
        [10.1, 11.1, 12.1],
        [10.2, 11.2, 12.2],
        [10.2, 11.2, 12.2],
        [10.3, 11.3, 12.3],
        [10.3, 11.3, 12.3],
        [10.4, 11.4, 12.4],
        [10.4, 11.4, 12.4],
        [10.5, 11.5, 12.5],
        [10.5, 11.5, 12.5],
        [10.6, 11.6, 12.6],
        [10.6, 11.6, 12.6]
    ], dtype=tf.float32)

    sample_weight_desired = tf.constant(
        [1., 1., 1., 1., 0.2, 0.2, 1., 1., 1., 1., 0.8, 0.8], dtype=tf.float32
    )

    y_pred_desired = tf.constant([
        [10.2, 12.2, 14.2],
        [10.2, 12.2, 14.2],
        [10.4, 12.4, 14.4],
        [10.4, 12.4, 14.4],
        [10.6, 12.6, 14.6],
        [10.6, 12.6, 14.6],
        [10.8, 12.8, 14.8],
        [10.8, 12.8, 14.8],
        [11., 13., 15.],
        [11., 13., 15.],
        [11.2, 13.2, 15.2],
        [11.2, 13.2, 15.2]
    ], dtype=tf.float32)

    # Check "sample axis" added correctly to inputs.
    for data in ds:
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        x = model.expand_inputs_with_sample_axis(x)
        tf.debugging.assert_equal(x['x0'], x0_desired)
        tf.debugging.assert_equal(x['x1'], x1_desired)
        tf.debugging.assert_equal(x['x2'], x2_desired)

    # Perform a `test_step`.
    for data in ds:
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Adjust `x` to include singleton "sample axis".
        x = model.expand_inputs_with_sample_axis(x)
        # Adjust `y` and `sample_weight` batch axis to reflect multiple
        # samples since `y_pred` has samples.
        y = model._repeat_samples_in_batch_axis(y)
        sample_weight = model._repeat_samples_in_batch_axis(sample_weight)
        # Assert `y` and `sample_weight` handled correctly.
        tf.debugging.assert_equal(y, y_desired)
        tf.debugging.assert_equal(sample_weight, sample_weight_desired)

        y_pred = model(x, training=False)
        # Reshape `y_pred` samples axis into batch axis.
        y_pred = model._reshape_samples_into_batch(y_pred)
        # Assert `y_pred` handled correctly.
        tf.debugging.assert_near(y_pred, y_pred_desired)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_mse_model_ax2_ignore(ds_rank2_x0_x1_x2, is_eager):
    """Test MSE model with sample_axis=2."""
    tf.config.run_functions_eagerly(is_eager)
    ds = ds_rank2_x0_x1_x2['ds']
    inputs_shape = ds_rank2_x0_x1_x2['inputs_shape']

    model = CustomModelC(sample_axis=2, n_sample=2, preserved_inputs=['x2'])
    model.build(inputs_shape)

    # Explicilty check that `sample_axis` and `n_sample` attributes of child
    # Layer were set correctly.
    assert model.branch_0.sample_axis == 2
    assert model.branch_0.n_sample == 2

    x0_desired = tf.constant([
        [[0.1], [1.1], [2.1]],
        [[0.2], [1.2], [2.2]],
        [[0.3], [1.3], [2.3]],
        [[0.4], [1.4], [2.4]],
        [[0.5], [1.5], [2.5]],
        [[0.6], [1.6], [2.6]]
    ], dtype=tf.float32)
    x1_desired = tf.constant([
        [[10.1], [11.1], [12.1]],
        [[10.2], [11.2], [12.2]],
        [[10.3], [11.3], [12.3]],
        [[10.4], [11.4], [12.4]],
        [[10.5], [11.5], [12.5]],
        [[10.6], [11.6], [12.6]]
    ], dtype=tf.float32)
    x2_desired = tf.constant([
        [20.1, 21.1, 22.1],
        [20.2, 21.2, 22.2],
        [20.3, 21.3, 22.3],
        [20.4, 21.4, 22.4],
        [20.5, 21.5, 22.5],
        [20.6, 21.6, 22.6]
    ], dtype=tf.float32)

    y_desired = tf.constant([
        [10.1, 11.1, 12.1],
        [10.1, 11.1, 12.1],
        [10.2, 11.2, 12.2],
        [10.2, 11.2, 12.2],
        [10.3, 11.3, 12.3],
        [10.3, 11.3, 12.3],
        [10.4, 11.4, 12.4],
        [10.4, 11.4, 12.4],
        [10.5, 11.5, 12.5],
        [10.5, 11.5, 12.5],
        [10.6, 11.6, 12.6],
        [10.6, 11.6, 12.6]
    ], dtype=tf.float32)

    sample_weight_desired = tf.constant(
        [1., 1., 1., 1., 0.2, 0.2, 1., 1., 1., 1., 0.8, 0.8], dtype=tf.float32
    )

    y_pred_desired = tf.constant([
        [10.2, 12.2, 14.2],
        [10.2, 12.2, 14.2],
        [10.4, 12.4, 14.4],
        [10.4, 12.4, 14.4],
        [10.6, 12.6, 14.6],
        [10.6, 12.6, 14.6],
        [10.8, 12.8, 14.8],
        [10.8, 12.8, 14.8],
        [11., 13., 15.],
        [11., 13., 15.],
        [11.2, 13.2, 15.2],
        [11.2, 13.2, 15.2]
    ], dtype=tf.float32)

    # Check "sample axis" added correctly to inputs.
    for data in ds:
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        x = model.expand_inputs_with_sample_axis(x)
        tf.debugging.assert_equal(x['x0'], x0_desired)
        tf.debugging.assert_equal(x['x1'], x1_desired)
        tf.debugging.assert_equal(x['x2'], x2_desired)

    # Perform a `test_step`.
    for data in ds:
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Adjust `x` to include singleton "sample axis".
        x = model.expand_inputs_with_sample_axis(x)
        # Adjust `y` and `sample_weight` batch axis to reflect multiple
        # samples since `y_pred` has samples.
        y = model._repeat_samples_in_batch_axis(y)
        sample_weight = model._repeat_samples_in_batch_axis(sample_weight)
        # Assert `y` and `sample_weight` handled correctly.
        tf.debugging.assert_equal(y, y_desired)
        tf.debugging.assert_equal(sample_weight, sample_weight_desired)

        y_pred = model(x, training=False)
        # Reshape `y_pred` samples axis into batch axis.
        y_pred = model._reshape_samples_into_batch(y_pred)
        # Assert `y_pred` handled correctly.
        tf.debugging.assert_near(y_pred, y_pred_desired)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_model_nsample_change(ds_rank2_x0_x1_x2, is_eager):
    """Test model where number of samples changes between use."""
    tf.config.run_functions_eagerly(is_eager)
    ds = ds_rank2_x0_x1_x2['ds']

    model = CustomModelC(sample_axis=2, n_sample=2)
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
    }
    model.compile(**compile_kwargs)

    model.fit(ds)
    assert model.branch_0.sample_axis == 2
    assert model.branch_0.n_sample == 2

    # Change model's `n_sample` attribute.
    model.n_sample = 5

    # When running model, we now expect the following:
    y_desired = tf.constant([
        [10.1, 11.1, 12.1],
        [10.1, 11.1, 12.1],
        [10.1, 11.1, 12.1],
        [10.1, 11.1, 12.1],
        [10.1, 11.1, 12.1],
        [10.2, 11.2, 12.2],
        [10.2, 11.2, 12.2],
        [10.2, 11.2, 12.2],
        [10.2, 11.2, 12.2],
        [10.2, 11.2, 12.2],
        [10.3, 11.3, 12.3],
        [10.3, 11.3, 12.3],
        [10.3, 11.3, 12.3],
        [10.3, 11.3, 12.3],
        [10.3, 11.3, 12.3],
        [10.4, 11.4, 12.4],
        [10.4, 11.4, 12.4],
        [10.4, 11.4, 12.4],
        [10.4, 11.4, 12.4],
        [10.4, 11.4, 12.4],
        [10.5, 11.5, 12.5],
        [10.5, 11.5, 12.5],
        [10.5, 11.5, 12.5],
        [10.5, 11.5, 12.5],
        [10.5, 11.5, 12.5],
        [10.6, 11.6, 12.6],
        [10.6, 11.6, 12.6],
        [10.6, 11.6, 12.6],
        [10.6, 11.6, 12.6],
        [10.6, 11.6, 12.6],
    ], dtype=tf.float32)
    sample_weight_desired = tf.constant(
        [
            1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1.,
            0.2, 0.2, 0.2, 0.2, 0.2,
            1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1.,
            0.8, 0.8, 0.8, 0.8, 0.8,
        ], dtype=tf.float32
    )
    y_pred_shape_desired = tf.TensorShape([30, 3])

    # Perform a `test_step` to verify `n_sample` took effect.
    for data in ds:
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Adjust `x` to include singleton "sample axis".
        x = model.expand_inputs_with_sample_axis(x)
        # Adjust `y` and `sample_weight` batch axis to reflect multiple
        # samples since `y_pred` has samples.
        y = model._repeat_samples_in_batch_axis(y)
        sample_weight = model._repeat_samples_in_batch_axis(sample_weight)
        # Assert `y` and `sample_weight` handled correctly.
        # Assert `y` and `sample_weight` handled correctly.
        tf.debugging.assert_equal(y, y_desired)
        tf.debugging.assert_equal(sample_weight, sample_weight_desired)

        y_pred = model(x, training=False)
        # Reshape `y_pred` samples axis into batch axis.
        y_pred = model._reshape_samples_into_batch(y_pred)
        # Assert `y_pred` handled correctly.
        tf.debugging.assert_equal(tf.shape(y_pred), y_pred_shape_desired)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_model_control_serialization(ds_rank2_x0, is_eager, tmpdir):
    """Test model serialization."""
    tf.config.run_functions_eagerly(is_eager)
    ds = ds_rank2_x0['ds']

    model = CustomModelControl()
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
    }
    model.compile(**compile_kwargs)
    model.fit(ds, epochs=2)
    # config = model.get_config()
    fp_model = tmpdir.join('test_model')
    model.save(fp_model)
    del model
    _ = tf.keras.models.load_model(
        fp_model, custom_objects={"CustomModelControl": CustomModelControl}
    )


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_model_a_serialization(ds_rank2_x0, is_eager, tmpdir):
    """Test model serialization."""
    tf.config.run_functions_eagerly(is_eager)
    ds = ds_rank2_x0['ds']

    model = CustomModelA(sample_axis=1, n_sample=2)
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
    }
    model.compile(**compile_kwargs)
    model.fit(ds, epochs=2)
    results_0 = model.evaluate(ds, return_dict=True)
    fp_model = tmpdir.join('test_model')
    model.save(fp_model)
    del model
    loaded = tf.keras.models.load_model(
        fp_model, custom_objects={"CustomModelA": CustomModelA}
    )
    results_1 = loaded.evaluate(ds, return_dict=True)

    # Test for model equality.
    assert loaded.sample_axis == 1
    assert loaded.n_sample == 2
    assert len(loaded.preserved_inputs) == 0
    assert results_0['loss'] == results_1['loss']


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_model_b_serialization(ds_rank2_x0, is_eager, tmpdir):
    """Test model serialization."""
    tf.config.run_functions_eagerly(is_eager)
    ds = ds_rank2_x0['ds']

    model = CustomModelB(sample_axis=1, n_sample=2)
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
    }
    model.compile(**compile_kwargs)
    model.fit(ds, epochs=2)
    results_0 = model.evaluate(ds, return_dict=True)
    fp_model = tmpdir.join('test_model')
    model.save(fp_model)
    del model
    loaded = tf.keras.models.load_model(
        fp_model, custom_objects={"CustomModelB": CustomModelB}
    )
    results_1 = loaded.evaluate(ds, return_dict=True)

    # Test for model equality.
    assert loaded.sample_axis == 1
    assert loaded.n_sample == 2
    assert loaded.dense_layer.sample_axis == 1
    assert loaded.dense_layer.n_sample == 2
    assert len(loaded.preserved_inputs) == 0
    assert results_0['loss'] == results_1['loss']


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_model_c_serialization(ds_rank2_x0_x1_x2, is_eager, tmpdir):
    """Test model serialization."""
    tf.config.run_functions_eagerly(is_eager)
    ds = ds_rank2_x0_x1_x2['ds']

    model = CustomModelC(sample_axis=2, n_sample=7, preserved_inputs=['x2'])
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
    }
    model.compile(**compile_kwargs)

    model.fit(ds, epochs=2)
    results_0 = model.evaluate(ds, return_dict=True)

    # Save the model.
    fp_model = tmpdir.join('test_model')
    model.save(fp_model)
    del model
    # Load the saved model.
    loaded = tf.keras.models.load_model(
        fp_model, custom_objects={"CustomModelC": CustomModelC}
    )
    results_1 = loaded.evaluate(ds, return_dict=True)

    # Test for model equality.
    assert loaded.sample_axis == 2
    assert loaded.n_sample == 7
    assert loaded.branch_0.sample_axis == 2
    assert loaded.branch_0.n_sample == 7
    assert loaded.branch_1.sample_axis == 2
    assert loaded.branch_1.n_sample == 7
    assert len(loaded.preserved_inputs) == 1
    assert loaded.preserved_inputs[0] == 'x2'
    assert results_0['loss'] == results_1['loss']
