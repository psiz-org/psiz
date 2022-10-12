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


def build_mle_kernel(similarity, n_dim):
    """Build kernel for single group."""
    mink = psiz.keras.layers.Minkowski(
        rho_trainable=False,
        rho_initializer=tf.keras.initializers.Constant(2.),
        w_constraint=psiz.keras.constraints.NonNegNorm(
            scale=n_dim, p=1.
        ),
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=mink,
        similarity=similarity
    )
    return kernel


def build_vi_kernel(similarity, n_dim, kl_weight):
    """Build kernel for single group."""
    mink_prior = psiz.keras.layers.MinkowskiStochastic(
        rho_loc_trainable=False, rho_scale_trainable=True,
        w_loc_trainable=False, w_scale_trainable=False,
        w_scale_initializer=tf.keras.initializers.Constant(.1)
    )

    mink_posterior = psiz.keras.layers.MinkowskiStochastic(
        rho_loc_trainable=False, rho_scale_trainable=True,
        w_loc_trainable=True, w_scale_trainable=True,
        w_scale_initializer=tf.keras.initializers.Constant(.1)
    )

    mink = psiz.keras.layers.MinkowskiVariational(
        prior=mink_prior, posterior=mink_posterior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=mink,
        similarity=similarity
    )
    return kernel


@pytest.fixture(scope="module")
def ds_rank_obs_3g():
    """Rank observations dataset."""
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    groups = np.array(([0], [0], [1], [2]), dtype=np.int32)

    obs = psiz.trials.RankObservations(
        stimulus_set, n_select=n_select, groups=groups
    )
    ds_obs = obs.as_dataset().batch(n_trial, drop_remainder=False)

    return ds_obs


@pytest.fixture
def rank_1g_mle_v2():
    """Rank, one group, MLE."""
    n_stimuli = 20
    n_dim = 3

    stimuli = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            # rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )
    model = psiz.keras.models.Rank(stimuli=stimuli, kernel=kernel)
    return model


@pytest.fixture
def rank_1g_mle_nomask_v2():
    """Rank, one group, MLE."""
    n_stimuli = 20
    n_dim = 3

    stimuli = tf.keras.layers.Embedding(
        n_stimuli, n_dim, mask_zero=False
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            # rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )
    model = psiz.keras.models.Rank(stimuli=stimuli, kernel=kernel)
    return model


@pytest.fixture
def rank_3g_mle_v2():
    """Rank, three groups, MLE."""
    n_stimuli = 20
    n_dim = 3

    stimuli = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )

    # Define group-specific kernels.
    kernel_0 = build_mle_kernel(shared_similarity, n_dim)
    kernel_1 = build_mle_kernel(shared_similarity, n_dim)
    kernel_2 = build_mle_kernel(shared_similarity, n_dim)
    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1, kernel_2], gating_index=-1
    )

    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel_group, use_group_kernel=True
    )
    return model


@pytest.fixture
def rank_1g_vi_v2():
    """Rank, one group, MLE."""
    n_stimuli = 20
    n_dim = 3
    kl_weight = .1

    prior_scale = .2
    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli + 1, n_dim, mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )
    stimuli = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    mink = psiz.keras.layers.Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.),
        w_initializer=tf.keras.initializers.Constant(1.),
        trainable=False
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=mink,
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    model = psiz.keras.models.Rank(stimuli=stimuli, kernel=kernel, n_sample=1)
    return model


@pytest.fixture
def rank_1g_emb_w_vi_v2():
    """Rank, one group, MLE."""
    n_stimuli = 20
    n_dim = 3
    kl_weight = .1

    prior_scale = .2
    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli + 1, n_dim, mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )
    stimuli = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    mink_prior = psiz.keras.layers.MinkowskiStochastic(
        rho_loc_trainable=False, rho_scale_trainable=True,
        w_loc_trainable=False, w_scale_trainable=False,
        w_scale_initializer=tf.keras.initializers.Constant(.1)
    )

    mink_posterior = psiz.keras.layers.MinkowskiStochastic(
        rho_loc_trainable=False, rho_scale_trainable=True,
        w_loc_trainable=True, w_scale_trainable=True,
        w_scale_initializer=tf.keras.initializers.Constant(.1)
    )

    mink = psiz.keras.layers.MinkowskiVariational(
        prior=mink_prior, posterior=mink_posterior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=mink,
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    model = psiz.keras.models.Rank(stimuli=stimuli, kernel=kernel, n_sample=1)
    return model


@pytest.fixture
def rank_3g_vi_v2():
    """Rank, one group, MLE."""
    n_stimuli = 20
    n_dim = 3
    kl_weight = .1

    prior_scale = .2

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli + 1, n_dim, mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False
        )
    )
    stimuli = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.),
        trainable=False
    )

    # Define group-specific kernels.
    kernel_0 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_1 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_2 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1, kernel_2], gating_index=-1
    )

    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel_group, use_group_kernel=True
    )
    return model


@pytest.fixture
def rank_3g_vi_v3():
    """Rank, one group, MLE."""
    n_stimuli = 20
    n_dim = 3
    kl_weight = .1

    prior_scale = .2

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli + 1, n_dim, mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False
        )
    )
    stimuli = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.),
        trainable=False
    )

    # Define group-specific kernels.
    kernel_0 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_1 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_2 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1, kernel_2], gating_index=-1
    )

    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel_group, use_group_kernel=True
    )
    return model


def test_n_sample_propogation(rank_1g_vi):
    """Test propogation properties."""
    assert rank_1g_vi.n_sample == 1

    # Set n_sample at model level.
    rank_1g_vi.n_sample = 100
    assert rank_1g_vi.n_sample == 100


def test_properties_0(rank_1g_mle_v2):
    """Test properties."""
    model = rank_1g_mle_v2

    n_stimuli = model.n_stimuli
    n_dim = model.n_dim

    n_stimui_desired = 20
    n_dim_desired = 3
    assert n_stimuli == n_stimui_desired
    assert n_dim == n_dim_desired


def test_properties_1(rank_1g_mle_nomask_v2):
    """Test properties."""
    model = rank_1g_mle_nomask_v2

    n_stimuli = model.n_stimuli
    n_dim = model.n_dim

    n_stimui_desired = 20
    n_dim_desired = 3
    assert n_stimuli == n_stimui_desired
    assert n_dim == n_dim_desired


def test_properties_2(rank_2stim_2kern_determ):
    """Test properties.

    Trigger cases involving BraidGate.

    """
    model = rank_2stim_2kern_determ

    n_stimuli = model.n_stimuli
    n_dim = model.n_dim

    n_stimui_desired = 3
    n_dim_desired = 2
    assert n_stimuli == n_stimui_desired
    assert n_dim == n_dim_desired


def test_properties_3(rank_2stim_2kern_nomask_determ):
    """Test properties.

    Trigger cases involving BraidGate.

    """
    model = rank_2stim_2kern_nomask_determ

    n_stimuli = model.n_stimuli
    n_dim = model.n_dim

    n_stimui_desired = 3
    n_dim_desired = 2
    assert n_stimuli == n_stimui_desired
    assert n_dim == n_dim_desired


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_call_2groups(
        rank_2g_mle, ds_rank_docket_2g, ds_rank_obs_2g, is_eager):
    """Test call with group-specific kernels."""
    tf.config.run_functions_eagerly(is_eager)
    model = rank_2g_mle

    # Compile
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    for data in ds_rank_docket_2g:
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        _ = model(x, training=False)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_fit_mle_1g(rank_1g_mle_v2, ds_rank_obs_2g, is_eager):
    """Test fit."""
    tf.config.run_functions_eagerly(is_eager)
    model = rank_1g_mle_v2
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_2g, epochs=2)

    # Evaluate.
    model.evaluate(ds_rank_obs_2g)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_fit_mle_3g(rank_3g_mle_v2, ds_rank_obs_2g, is_eager):
    """Test fit."""
    is_eager = tf.config.run_functions_eagerly(is_eager)
    model = rank_3g_mle_v2
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_2g, epochs=2)

    # Evaluate.
    model.evaluate(ds_rank_obs_2g)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_fit_vi_1g(rank_1g_vi_v2, ds_rank_obs_2g, is_eager):
    """Test fit."""
    is_eager = tf.config.run_functions_eagerly(is_eager)
    model = rank_1g_vi_v2
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_2g, epochs=2)

    # Evaluate.
    model.evaluate(ds_rank_obs_2g)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_fit_vi_emb_w_1g(rank_1g_emb_w_vi_v2, ds_rank_obs_2g, is_eager):
    """Test fit."""
    is_eager = tf.config.run_functions_eagerly(is_eager)
    model = rank_1g_emb_w_vi_v2
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_2g, epochs=2)

    # Evaluate.
    model.evaluate(ds_rank_obs_2g)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_fit_vi_3g(rank_3g_vi_v2, ds_rank_obs_3g, is_eager):
    """Test fit."""
    is_eager = tf.config.run_functions_eagerly(is_eager)
    model = rank_3g_vi_v2
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_3g, epochs=2)

    # Evaluate.
    model.evaluate(ds_rank_obs_3g)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_fit_vi_3g_empty_kernel_branch(
        rank_3g_vi_v2, ds_rank_obs_2g, is_eager):
    """Test fit case where third "expert" has batch_size=0.

    This will generate an obscure error if not handled correctly.
    See:
    https://github.com/matterport/Mask_RCNN/issues/521#issuecomment-646096887

    """
    tf.config.run_functions_eagerly(is_eager)
    model = rank_3g_vi_v2
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_2g, epochs=2)

    # Evaluate.
    model.evaluate(ds_rank_obs_2g)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_fit_vi_3g_v3_empty_kernel_branch(
        rank_3g_vi_v3, ds_rank_obs_2g, is_eager):
    """Test fit case where third "expert" has batch_size=0.

    This will generate an obscure error if not handled correctly.
    See:
    https://github.com/matterport/Mask_RCNN/issues/521#issuecomment-646096887

    """
    tf.config.run_functions_eagerly(is_eager)
    model = rank_3g_vi_v3
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_2g, epochs=2)

    # Evaluate.
    model.evaluate(ds_rank_obs_2g)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_fit_mle_2stimuli_2kernel(
        rank_2stim_2kern_2behav, ds_rank_obs_2g, is_eager):
    """Test fit case where third "expert" has batch_size=0.

    This will generate an obscure error if not handled correctly.
    See:
    https://github.com/matterport/Mask_RCNN/issues/521#issuecomment-646096887

    """
    tf.config.run_functions_eagerly(is_eager)
    model = rank_2stim_2kern_2behav
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_2g, epochs=2)

    # Evaluate.
    model.evaluate(ds_rank_obs_2g)


def test_save_load_rank_wtrace(
        rank_1g_mle, tmpdir, ds_rank_docket, ds_rank_obs_2g):
    """Test loading and saving of embedding model."""
    model = rank_1g_mle

    # Compile
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_2g, epochs=1)

    # Predict.
    output_0 = model.predict(ds_rank_docket)

    model_n_stimuli = model.n_stimuli
    model_n_dim = model.n_dim

    # Save the model.
    fn = tmpdir.join('embedding_test')
    model.save(fn, overwrite=True, save_traces=True)
    del model
    # Load the saved model.
    reconstructed_model = tf.keras.models.load_model(fn)

    # Predict using loaded model.
    output_1 = reconstructed_model.predict(ds_rank_docket)

    # Test for equality.
    np.testing.assert_allclose(output_0, output_1)
    assert reconstructed_model.n_stimuli == model_n_stimuli
    assert reconstructed_model.n_dim == model_n_dim

    # Continue training without recompiling.
    reconstructed_model.fit(ds_rank_obs_2g, epochs=1)


def test_save_load_rank_wotrace(
        rank_1g_mle, tmpdir, ds_rank_docket, ds_rank_obs_2g):
    """Test loading and saving of embedding model."""
    model = rank_1g_mle

    # Compile
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    # Fit one epoch.
    model.fit(ds_rank_obs_2g, epochs=1)

    # Predict.
    output_0 = model.predict(ds_rank_docket)

    model_n_stimuli = model.n_stimuli
    model_n_dim = model.n_dim

    # Save the model.
    fn = tmpdir.join('embedding_test')
    model.save(fn, overwrite=True, save_traces=False)
    del model
    # Load the saved model.
    reconstructed_model = tf.keras.models.load_model(fn)

    # Predict using loaded model.
    output_1 = reconstructed_model.predict(ds_rank_docket)

    # Test for equality.
    np.testing.assert_allclose(output_0, output_1)
    assert reconstructed_model.n_stimuli == model_n_stimuli
    assert reconstructed_model.n_dim == model_n_dim

    # Continue training without recompiling.
    reconstructed_model.fit(ds_rank_obs_2g, epochs=1)
