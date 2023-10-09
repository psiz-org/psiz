# -*- coding: utf-8 -*-
# Copyright 2021 The PsiZ Authors. All Rights Reserved.
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
"""Test trials module."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz
from psiz.tf.information_theory import ig_model_categorical


class RankModel(psiz.keras.StochasticModel):
    """A behavior model.

    No Gates.

    """

    def __init__(self, percept=None, proximity=None, soft_2rank1=None, **kwargs):
        """Initialize."""
        super(RankModel, self).__init__(**kwargs)
        self.stimuli_axis = 1
        self.percept = percept
        self.proximity = psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant(1.0),
            activation=psiz.keras.layers.ExponentialSimilarity(
                trainable=False,
                beta_initializer=tf.keras.initializers.Constant(10.0),
                tau_initializer=tf.keras.initializers.Constant(1.0),
                gamma_initializer=tf.keras.initializers.Constant(0.0),
            ),
            trainable=False,
        )
        self.soft_2rank1 = psiz.keras.layers.SoftRank(n_select=1)

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given2rank1_stimulus_set"])
        z_q, z_r = tf.split(z, [1, 2], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_2rank1(s)


def base_circular_model():
    """Build base model.

    Circular arrangement with equal variance.

    """
    # Create embedding points arranged in a circle.
    center_point = np.array([[0.25, 0.25]])
    # Determine polar coordiantes.
    r = 0.15
    theta = np.linspace(0, 2 * np.pi, 9)
    theta = theta[0:-1]  # Drop last point, which is a repeat of first.
    theta = np.expand_dims(theta, axis=1)

    # Convert to Cartesian coordinates.
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Translate coordinates.
    loc = np.hstack((x, y)) + center_point
    # Add center point.
    loc = np.concatenate((center_point, loc), axis=0)
    (n_stimuli, n_dim) = loc.shape
    # Add placeholder.
    loc = np.vstack((np.zeros([1, 2]), loc))

    # Create model.
    prior_scale = 0.17
    percept = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(loc),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        ),
    )

    model = RankModel(percept=percept)
    # Call build to force initializers to execute.
    model.percept.build([None, None, None])
    return model, loc


@pytest.fixture
def model_0():
    model, loc = base_circular_model()
    (n_point, n_dim) = loc.shape

    # Do not modify loc, but give one point 3x higher scale.
    scale = 0.01 * np.ones([n_point, n_dim], dtype=np.float32)
    scale[4, :] = 0.03

    # Assign `loc` and `scale` to model.
    model.percept.loc.assign(loc)
    model.percept.untransformed_scale.assign(tfp.math.softplus_inverse(scale))
    return model


@pytest.fixture
def model_1():
    model, loc = base_circular_model()
    (n_point, n_dim) = loc.shape

    # Translate loc and give one point 3x higher scale.
    loc = loc - 0.1
    scale = 0.01 * np.ones([n_point, n_dim], dtype=np.float32)
    scale[4, :] = 0.03

    # Assign `loc` and `scale` to model.
    model.percept.loc.assign(loc)
    model.percept.untransformed_scale.assign(tfp.math.softplus_inverse(scale))
    return model


@pytest.fixture
def ds_2rank1():
    """Rank docket dataset."""
    stimulus_set = np.array(
        [[[4, 3, 5]], [[4, 2, 5]], [[4, 3, 6]], [[4, 2, 6]], [[4, 5, 6]], [[7, 1, 3]]],
        dtype=np.int32,
    )
    content = psiz.data.Rank(stimulus_set, n_select=1)
    tfds = psiz.data.Dataset([content]).export(
        export_format="tfds", with_timestep_axis=False
    )
    tfds = tfds.batch(stimulus_set.shape[0], drop_remainder=False)
    return tfds


def test_1_model(model_0, ds_2rank1):
    """Test IG computation using one model."""
    n_sample = 10000

    ig = []
    for x in ds_2rank1:
        ig.append(ig_model_categorical([model_0], x, n_sample))
    ig = tf.concat(ig, 0)

    # Assert IG values are in the right ballpark.
    # Old target.
    # ig_desired = tf.constant(
    #     [
    #         0.01855242, 0.01478302, 0.01481092, 0.01253963, 0.0020327,
    #         0.00101262
    #     ], dtype=tf.float32
    # )
    ig_desired = tf.constant(
        [0.03496134, 0.02489483, 0.02482754, 0.02341402, 0.003456, 0.00146809],
        dtype=tf.float32,
    )
    tf.debugging.assert_near(ig, ig_desired, rtol=0.1)

    # Assert that IG of first and last in correct order.
    tf.debugging.assert_greater(ig[0], ig[-1])


def test_2_models(model_0, model_1, ds_2rank1):
    """Test IG computation using twp models.

    The second model is merely a translation of the first, so basic IG
    tests remain unchanged.

    """
    n_sample = 10000

    ig = []
    for x in ds_2rank1:
        ig.append(ig_model_categorical([model_0, model_1], x, n_sample))
    ig = tf.concat(ig, 0)

    # Assert IG values are in the right ballpark.
    # Old target.
    # ig_desired = tf.constant(
    #     [
    #         0.01855242, 0.01478302, 0.01481092, 0.01253963, 0.0020327,
    #         0.00101262
    #     ], dtype=tf.float32
    # )
    ig_desired = tf.constant(
        [0.03436345, 0.02473378, 0.02513063, 0.02297509, 0.00344968, 0.00151819],
        dtype=tf.float32,
    )
    tf.debugging.assert_near(ig, ig_desired, rtol=0.1)

    # Assert that IG of first and last in correct order.
    tf.debugging.assert_greater(ig[0], ig[-1])
