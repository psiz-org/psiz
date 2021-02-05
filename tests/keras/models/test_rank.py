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
from tensorflow.python.keras.engine import data_adapter

import psiz


def test_n_sample_propogation(rank_1g_vi):
    """Test propogation properties."""
    assert rank_1g_vi.n_sample == 1

    # Set n_sample at model level.
    rank_1g_vi.n_sample = 100
    assert rank_1g_vi.n_sample == 100


def test_kl_weight_propogation(rank_1g_vi):
    """Test propogation properties."""
    assert rank_1g_vi.kl_weight == 0.

    # Set kl_weight at model level.
    rank_1g_vi.kl_weight = .001
    # Test property propagated to all relevant layers.
    assert rank_1g_vi.kl_weight == .001
    assert rank_1g_vi.stimuli.embedding.kl_weight == .001


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_call_2groups(
        rank_2g_mle, ds_rank_docket_2g, ds_rank_obs_2g, is_eager):
    """Test call with group-specific kernels."""
    tf.config.run_functions_eagerly(is_eager)
    model = rank_2g_mle
    n_trial = 4
    # n_submodule = len(model.submodules)

    # Compile
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    for data in ds_rank_docket_2g:
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        output = model(x, training=False)


def test_save_load_rank_wtrace(
        rank_1g_mle, tmpdir, ds_rank_docket, ds_rank_obs_2g):
    """Test loading and saving of embedding model."""
    model = rank_1g_mle
    # n_submodule = len(model.submodules)

    # Compile
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_2g, epochs=1)

    # Predict.
    output_0 = model.predict(ds_rank_docket)

    # Save the model.
    fn = tmpdir.join('embedding_test')
    model.save(fn, overwrite=True, save_traces=True)
    # Load the saved model.
    reconstructed_model = tf.keras.models.load_model(fn)

    # Check submodule agreement. TODO
    # n_submodule_recon = len(reconstructed_model.submodules)
    # assert n_submodule == n_submodule_recon

    # Predict using loaded model.
    output_1 = reconstructed_model.predict(ds_rank_docket)

    # Test for equality.
    np.testing.assert_allclose(output_0, output_1)
    assert reconstructed_model.n_stimuli == model.n_stimuli
    assert reconstructed_model.n_dim == model.n_dim
    assert reconstructed_model.n_group == model.n_group

    # Continue training without recompiling.
    reconstructed_model.fit(ds_rank_obs_2g, epochs=1)


def test_save_load_rank_wotrace(
        rank_1g_mle, tmpdir, ds_rank_docket, ds_rank_obs_2g):
    """Test loading and saving of embedding model."""
    model = rank_1g_mle
    # n_submodule = len(model.submodules)

    # Compile
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_2g, epochs=1)

    # Predict.
    output_0 = model.predict(ds_rank_docket)

    # Save the model.
    fn = tmpdir.join('embedding_test')
    model.save(fn, overwrite=True, save_traces=False)
    # Load the saved model.
    reconstructed_model = tf.keras.models.load_model(fn)

    # Check submodule agreement. TODO
    # TODO Why are submodules destroyed during save/load?
    # n_submodule_recon = len(reconstructed_model.submodules)
    # assert n_submodule == n_submodule_recon

    # Predict using loaded model.
    output_1 = reconstructed_model.predict(ds_rank_docket)

    # Test for equality.
    np.testing.assert_allclose(output_0, output_1)
    assert reconstructed_model.n_stimuli == model.n_stimuli
    assert reconstructed_model.n_dim == model.n_dim
    assert reconstructed_model.n_group == model.n_group

    # Continue training without recompiling.
    reconstructed_model.fit(ds_rank_obs_2g, epochs=1)

    # TODO test without compile

    # np.testing.assert_array_equal(
    #     loaded_embedding.z,
    #     rank_1g_vi.z
    # )
    # np.testing.assert_array_equal(
    #     loaded_embedding._z["value"],
    #     rank_1g_vi._z["value"]
    # )
    # assert loaded_embedding._z['trainable'] == rank_1g_vi._z['trainable']

    # assert loaded_embedding._theta == rank_1g_vi._theta

    # np.testing.assert_array_equal(
    #     loaded_embedding.w,
    #     rank_1g_vi.w
    # )
    # for param_name in rank_1g_vi._phi:
    #     np.testing.assert_array_equal(
    #         loaded_embedding._phi[param_name]['value'],
    #         rank_1g_vi._phi[param_name]['value']
    #     )
    #     np.testing.assert_array_equal(
    #         loaded_embedding._phi[param_name]['trainable'],
    #         rank_1g_vi._phi[param_name]['trainable']
    #     )
