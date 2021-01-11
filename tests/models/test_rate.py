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
"""Module for testing models.py."""


import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz


def test_save_load_rate_wtrace(
        rate_1g_mle, tmpdir, ds_rate_docket, ds_rate_obs):
    """Test loading and saving of embedding model."""
    model = rate_1g_mle
    # Compile
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rate_obs, epochs=1)

    # Predict using original model.
    output_0 = model.predict(ds_rate_docket)

    # Save the model.
    fn = tmpdir.join('embedding_test')
    model.save(fn, overwrite=True, save_traces=True)

    # Load the saved model.
    reconstructed_model = tf.keras.models.load_model(fn)

    # Predict using loaded model.
    output_1 = reconstructed_model.predict(ds_rate_docket)

    # Test for equality.
    np.testing.assert_allclose(output_0, output_1)
    assert reconstructed_model.n_stimuli == model.n_stimuli
    assert reconstructed_model.n_dim == model.n_dim
    assert reconstructed_model.n_group == model.n_group

    # Continue training without recompiling.
    reconstructed_model.fit(ds_rate_obs, epochs=1)


def test_save_load_rate_wotrace(
        rate_1g_mle, tmpdir, ds_rate_docket, ds_rate_obs):
    """Test loading and saving of embedding model."""
    model = rate_1g_mle
    # Compile
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rate_obs, epochs=1)

    # Predict using original model.
    output_0 = model.predict(ds_rate_docket)

    # Save the model.
    fn = tmpdir.join('embedding_test')
    model.save(fn, overwrite=True, save_traces=False)

    # Load the saved model.
    reconstructed_model = tf.keras.models.load_model(fn)

    # Predict using loaded model.
    output_1 = reconstructed_model.predict(ds_rate_docket)

    # Test for equality.
    np.testing.assert_allclose(output_0, output_1)
    assert reconstructed_model.n_stimuli == model.n_stimuli
    assert reconstructed_model.n_dim == model.n_dim
    assert reconstructed_model.n_group == model.n_group

    # Continue training without recompiling.
    reconstructed_model.fit(ds_rate_obs, epochs=1)
