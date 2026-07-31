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
"""Module for testing models.py."""


from pathlib import Path

import keras
import numpy as np
import pytest

import psiz.keras.layers
from psiz.stochastic.transforms import softplus_inverse
from psiz.keras.models.stochastic_model import StochasticModel


def _map_leaves(fn, obj):
    """Apply `fn` to all non-container leaves in a nested structure."""
    if isinstance(obj, dict):
        return {k: _map_leaves(fn, v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        mapped = [_map_leaves(fn, v) for v in obj]
        return type(obj)(mapped)
    return fn(obj)


def _infer_n_sample(value):
    """Infer sample count from an array-like or dict of arrays."""
    if isinstance(value, dict):
        key = next(iter(value.keys()))
        return int(np.asarray(value[key]).shape[0])
    return int(np.asarray(value).shape[0])


class _NumpyPyDataset(keras.utils.PyDataset):
    """Simple NumPy-backed `PyDataset` used by JAX runtime tests."""

    def __init__(self, x, y, w, *, batch_size, shuffle=False, seed=None):
        self.x = x
        self.y = y
        self.w = w
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self._n_sample = _infer_n_sample(x)
        self._indices = np.arange(self._n_sample)
        super().__init__()

    def __len__(self):
        return int(np.ceil(self._n_sample / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self._indices)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self._n_sample)
        idx = self._indices[start:end]
        return (
            _map_leaves(lambda v: np.asarray(v)[idx], self.x),
            np.asarray(self.y)[idx],
            np.asarray(self.w)[idx],
        )


def _build_tf_dataset(case):
    """Build a `tf.data.Dataset` from a canonical raw case."""
    tf = pytest.importorskip("tensorflow")
    ds = tf.data.Dataset.from_tensor_slices((case["x"], case["y"], case["w"]))
    return ds.batch(case["batch_size"], drop_remainder=False)


def _build_torch_dataloader(case):
    """Build a Torch `DataLoader` from a canonical raw case."""
    torch = pytest.importorskip("torch")
    torch_utils_data = pytest.importorskip("torch.utils.data")

    n_example = case["batch_size"]

    def as_torch(value):
        return torch.as_tensor(value)

    x = _map_leaves(as_torch, case["x"])
    y = as_torch(case["y"])
    w = as_torch(case["w"])

    class CaseDataset(torch_utils_data.Dataset):
        def __len__(self):
            return n_example

        def __getitem__(self, idx):
            x_i = _map_leaves(lambda v: v[idx], x)
            return x_i, y[idx], w[idx]

    return torch_utils_data.DataLoader(
        CaseDataset(), batch_size=n_example, shuffle=False, drop_last=False
    )


def _build_numpy_pydataset(case):
    """Build a NumPy-backed `PyDataset` from a canonical raw case."""
    x = _map_leaves(np.asarray, case["x"])
    y = np.asarray(case["y"])
    w = np.asarray(case["w"])
    return _NumpyPyDataset(x, y, w, batch_size=case["batch_size"])


_DATASET_ADAPTERS = {
    "tf": _build_tf_dataset,
    "torch": _build_torch_dataloader,
    "numpy": _build_numpy_pydataset,
}


def _materialize_dataset(case, backend):
    """Materialize a backend-specific iterable from a canonical raw case."""
    return _DATASET_ADAPTERS[backend](case)


@pytest.fixture(
    scope="module",
    params=["tf", "torch", "numpy"],
    ids=["tf_dataset", "torch_dataloader", "numpy_pydataset"],
)
def data_backend(request):
    """Backend used for dataset materialization in local fixtures."""
    runtime_backend = keras.backend.backend()
    if runtime_backend == "jax":
        if request.param != "numpy":
            pytest.skip("JAX runtime tests use NumPy-backed dataset materialization.")
    elif request.param == "numpy":
        pytest.skip("NumPy-backed dataset materialization is reserved for JAX runtime tests.")
    return request.param


@keras.saving.register_keras_serializable()
class LayerA(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(LayerA, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.Constant(1.0)

    def build(self, input_shape):
        """Build."""
        if self.built:
            return
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            trainable=True,
            name="kernel",
            dtype=self.dtype,
        )

    def call(self, inputs, training=False):
        x = keras.ops.matmul(inputs, self.kernel)
        return x

    def get_config(self):
        config = super(LayerA, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable()
class LayerB(keras.layers.Layer):
    """A simple repeat layer."""

    def __init__(self, **kwargs):
        """Initialize."""
        super(LayerB, self).__init__(**kwargs)
        self.w0_initializer = keras.initializers.Constant(1.0)

    def build(self, input_shape):
        """Build."""
        if self.built:
            return
        self.w0 = self.add_weight(
            shape=[],
            initializer=self.w0_initializer,
            trainable=True,
            name="w0",
            dtype=self.dtype,
        )

    def call(self, inputs, training=None):
        """Call."""
        return self.w0 * inputs

    def get_config(self):
        return super(LayerB, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable()
class CellA(keras.layers.Layer):
    """A simple RNN cell."""

    def __init__(self, **kwargs):
        """Initialize."""
        super(CellA, self).__init__(**kwargs)
        self.layer_0 = LayerA(3)

        # Satisfy RNNCell contract.
        # NOTE: A placeholder state.
        self.state_size = [1]

    def build(self, input_shape):
        """Build."""
        if self.built:
            return
        self.layer_0.build(input_shape)
        super().build(input_shape)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial state."""
        initial_state = [keras.ops.zeros([batch_size, 1])]
        return initial_state

    def call(self, inputs, states, training=None):
        """Call."""
        outputs = self.layer_0(inputs)
        return outputs, states

    def get_config(self):
        return super(CellA, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ModelControl(keras.Model):
    """A non-stochastic model to use as a control case.

    Gates:
        None

    """

    def __init__(self, dense_layer=None, **kwargs):
        super(ModelControl, self).__init__(**kwargs)
        if dense_layer is None:
            dense_layer = keras.layers.Dense(3)
        self.dense_layer = dense_layer

    def call(self, inputs):
        x = inputs["x_a"]
        x = self.dense_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dense_layer": self.dense_layer,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["dense_layer"] = keras.layers.deserialize(config["dense_layer"])
        return cls(**config)


class ModelA(StochasticModel):
    """A stochastic model.

    Default input handling.
    No custom layers.

    Gates:
        None

    """

    def __init__(self, dense_layer=None, **kwargs):
        super(ModelA, self).__init__(**kwargs)
        if dense_layer is None:
            dense_layer = keras.layers.Dense(3)
        self.dense_layer = dense_layer

    def call(self, inputs):
        x = self.dense_layer(inputs["x_a"])
        return x

    def get_config(self):
        config = super(ModelA, self).get_config()
        config.update(
            {
                "dense_layer": self.dense_layer,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["dense_layer"] = keras.layers.deserialize(config["dense_layer"])
        return cls(**config)


class ModelB(StochasticModel):
    """A stochastic model with a custom layer.

    Custom layer.
    Assumes single tensor input as dictionary.

    Gates:
        None

    """

    def __init__(self, custom_layer=None, **kwargs):
        super(ModelB, self).__init__(**kwargs)
        if custom_layer is None:
            custom_layer = LayerA(3)
        self.custom_layer = custom_layer

    def call(self, inputs):
        x = self.custom_layer(inputs["x_a"])
        return x

    def get_config(self):
        """Return model configuration."""
        config = super(ModelB, self).get_config()
        config.update(
            {
                "custom_layer": self.custom_layer,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # config["custom_layer"] = LayerA.from_config(config["custom_layer"]) TODO use or remove
        config["custom_layer"] = keras.layers.deserialize(config["custom_layer"])
        return cls(**config)


class ModelB2(StochasticModel):
    """A stochastic model with a custom layer.

    Custom layer.
    Assumes single tensor input as tensor via methid overriding.

    Gates:
        None

    """

    def __init__(self, custom_layer=None, **kwargs):
        super(ModelB2, self).__init__(**kwargs)
        if custom_layer is None:
            custom_layer = LayerA(3)
        self.custom_layer = custom_layer

    def call(self, inputs):
        x = self.custom_layer(inputs)
        return x

    def get_config(self):
        """Return model configuration."""
        config = super(ModelB2, self).get_config()
        config.update(
            {
                "custom_layer": self.custom_layer,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["custom_layer"] = keras.layers.deserialize(config["custom_layer"])
        return cls(**config)


class ModelC(StochasticModel):
    """A stochastic model with a custom layer.

    Assumes dictionary of tensors input.

    Gates:
        None

    """

    def __init__(self, branch_0=None, branch_1=None, **kwargs):
        """Initialize."

        Args:
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super(ModelC, self).__init__(**kwargs)
        if branch_0 is None:
            branch_0 = LayerB()
        self.branch_0 = branch_0
        if branch_1 is None:
            branch_1 = LayerB()
        self.branch_1 = branch_1
        self.add_layer = keras.layers.Add()

    def build(self, input_shape):
        self._input_shape = input_shape
        self.branch_0.build(input_shape["x_a"])
        self.branch_1.build(input_shape["x_b"])
        super().build(input_shape)

    def call(self, inputs):
        """Call.

        Args:
            inputs: A dictionary of inputs.

        """
        x_a = inputs["x_a"]
        x_b = inputs["x_b"]
        x_a = self.branch_0(x_a)
        x_b = self.branch_1(x_b)
        return self.add_layer([x_a, x_b])

    def get_config(self):
        """Return model configuration."""
        config = super(ModelC, self).get_config()
        config.update(
            {
                "branch_0": self.branch_0,
                "branch_1": self.branch_1,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["branch_0"] = keras.layers.deserialize(config["branch_0"])
        config["branch_1"] = keras.layers.deserialize(config["branch_1"])
        return cls(**config)

    def get_build_config(self):
        build_config = {
            "input_shape": self._input_shape,
        }
        return build_config

    def build_from_config(self, config):
        self.build(config["input_shape"])


class ModelD(StochasticModel):
    """A stochastic model with an RNN layer.

    Assumes dictionary of tensors input.

    Gates:
        None

    """

    def __init__(self, rnn_layer=None, **kwargs):
        """Initialize."

        Args:
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super(ModelD, self).__init__(**kwargs)
        if rnn_layer is None:
            rnn_layer = keras.layers.RNN(CellA(), return_sequences=True)
        self.rnn_layer = rnn_layer
        self.add_layer = keras.layers.Add()

    def build(self, input_shape):
        self.rnn_layer.build(input_shape["x_a"])
        super().build(input_shape)

    def call(self, inputs):
        """Call.

        Args:
            inputs: A dictionary of inputs.

        """
        x_a = inputs["x_a"]
        x_b = inputs["x_b"]
        x_a = self.rnn_layer(x_a)
        return self.add_layer([x_a, x_b])

    def get_config(self):
        """Return model configuration."""
        config = super(ModelD, self).get_config()
        config.update(
            {
                "rnn_layer": self.rnn_layer,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["rnn_layer"] = keras.layers.deserialize(config["rnn_layer"])
        return cls(**config)


class RankModelA(StochasticModel):
    """A `SoftRank` model.

    A stochastic, non-VI percept layer.

    Gates:
        None

    """

    def __init__(self, percept=None, proximity=None, soft_4rank1=None, **kwargs):
        """Initialize."""
        super(RankModelA, self).__init__(**kwargs)

        self.stimuli_axis = 1

        if percept is None:
            n_stimuli = 20
            n_dim = 3
            prior_scale = 0.2
            percept = psiz.keras.layers.EmbeddingNormalDiag(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
                scale_initializer=keras.initializers.Constant(
                    keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
                ),
            )
        self.percept = percept

        if proximity is None:
            proximity = psiz.keras.layers.Minkowski(
                rho_initializer=keras.initializers.Constant(2.0),
                w_initializer=keras.initializers.Constant(1.0),
                trainable=False,
                activation=psiz.keras.layers.ExponentialSimilarity(
                    beta_initializer=keras.initializers.Constant(10.0),
                    tau_initializer=keras.initializers.Constant(1.0),
                    gamma_initializer=keras.initializers.Constant(0.001),
                    trainable=False,
                ),
            )
        self.proximity = proximity

        if soft_4rank1 is None:
            soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)
        self.soft_4rank1 = soft_4rank1

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given4rank1_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_4rank1(s)

    def get_config(self):
        config = super(RankModelA, self).get_config()
        config.update(
            {
                "percept": keras.layers.serialize(self.percept),
                "proximity": keras.layers.serialize(self.proximity),
                "soft_4rank1": keras.layers.serialize(self.soft_4rank1),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["percept"] = keras.layers.deserialize(config["percept"])
        config["proximity"] = keras.layers.deserialize(config["proximity"])
        config["soft_4rank1"] = keras.layers.deserialize(config["soft_4rank1"])
        return cls(**config)


class RankModelB(StochasticModel):
    """A `SoftRank` model.

    A variational percept layer.

    Gates:
        None

    """

    def __init__(self, percept=None, proximity=None, soft_4rank1=None, **kwargs):
        """Initialize."""
        super(RankModelB, self).__init__(**kwargs)

        self.stimuli_axis = 1

        if percept is None:
            n_stimuli = 20
            n_dim = 3
            kl_weight = 0.1
            prior_scale = 0.2
            embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
                scale_initializer=keras.initializers.Constant(
                    keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
                ),
            )
            embedding_prior = psiz.keras.layers.EmbeddingShared(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
                embedding=psiz.keras.layers.EmbeddingNormalDiag(
                    1,
                    1,
                    loc_initializer=keras.initializers.Constant(0.0),
                    scale_initializer=keras.initializers.Constant(
                        keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
                    ),
                    loc_trainable=False,
                ),
            )
            percept = psiz.keras.layers.EmbeddingVariational(
                posterior=embedding_posterior,
                prior=embedding_prior,
                kl_weight=kl_weight,
                kl_n_sample=30,
            )
        self.percept = percept

        if proximity is None:
            proximity = psiz.keras.layers.Minkowski(
                rho_initializer=keras.initializers.Constant(2.0),
                w_initializer=keras.initializers.Constant(1.0),
                activation=psiz.keras.layers.ExponentialSimilarity(
                    trainable=False,
                    beta_initializer=keras.initializers.Constant(10.0),
                    tau_initializer=keras.initializers.Constant(1.0),
                    gamma_initializer=keras.initializers.Constant(0.0),
                ),
                trainable=False,
            )
        self.proximity = proximity

        if soft_4rank1 is None:
            soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)
        self.soft_4rank1 = soft_4rank1

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given4rank1_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_4rank1(s)

    def get_config(self):
        config = super(RankModelB, self).get_config()
        config.update(
            {
                "percept": self.percept,
                "proximity": self.proximity,
                "soft_4rank1": self.soft_4rank1,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["percept"] = keras.layers.deserialize(config["percept"])
        config["proximity"] = keras.layers.deserialize(config["proximity"])
        config["soft_4rank1"] = keras.layers.deserialize(config["soft_4rank1"])
        return cls(**config)


class RankModelC(StochasticModel):
    """A `SoftRank` model.

    A variational percept layer.

    Gates:
        Percept layer (BraidGate:2) with shared prior.

    """

    def __init__(self, braid_percept=None, proximity=None, soft_4rank1=None, **kwargs):
        """Initialize."""
        super(RankModelC, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        kl_weight = 0.1
        prior_scale = 0.2
        self.stimuli_axis = 1

        if braid_percept is None:
            embedding_posterior_0 = psiz.keras.layers.EmbeddingNormalDiag(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
                scale_initializer=keras.initializers.Constant(
                    keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
                ),
            )
            embedding_posterior_1 = psiz.keras.layers.EmbeddingNormalDiag(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
                scale_initializer=keras.initializers.Constant(
                    keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
                ),
            )
            embedding_prior = psiz.keras.layers.EmbeddingShared(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
                embedding=psiz.keras.layers.EmbeddingNormalDiag(
                    1,
                    1,
                    loc_initializer=keras.initializers.Constant(0.0),
                    scale_initializer=keras.initializers.Constant(
                        keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
                    ),
                    loc_trainable=False,
                ),
            )
            percept_0 = psiz.keras.layers.EmbeddingVariational(
                posterior=embedding_posterior_0,
                prior=embedding_prior,
                kl_weight=kl_weight,
                kl_n_sample=30,
            )
            percept_1 = psiz.keras.layers.EmbeddingVariational(
                posterior=embedding_posterior_1,
                prior=embedding_prior,
                kl_weight=kl_weight,
                kl_n_sample=30,
            )
            braid_percept = psiz.keras.layers.BraidGate(
                subnets=[percept_0, percept_1], gating_index=-1
            )
        self.braid_percept = braid_percept

        if proximity is None:
            proximity = psiz.keras.layers.Minkowski(
                rho_initializer=keras.initializers.Constant(2.0),
                w_initializer=keras.initializers.Constant(1.0),
                activation=psiz.keras.layers.ExponentialSimilarity(
                    trainable=False,
                    beta_initializer=keras.initializers.Constant(10.0),
                    tau_initializer=keras.initializers.Constant(1.0),
                    gamma_initializer=keras.initializers.Constant(0.0),
                ),
                trainable=False,
            )
        self.proximity = proximity

        if soft_4rank1 is None:
            soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)
        self.soft_4rank1 = soft_4rank1

    def call(self, inputs):
        """Call."""
        z = self.braid_percept(
            [inputs["given4rank1_stimulus_set"], inputs["percept_gate_weights"]]
        )
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_4rank1(s)

    def get_config(self):
        config = super(RankModelC, self).get_config()
        config.update(
            {
                "braid_percept": self.braid_percept,
                "proximity": self.proximity,
                "soft_4rank1": self.soft_4rank1,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["braid_percept"] = keras.layers.deserialize(config["braid_percept"])
        config["proximity"] = keras.layers.deserialize(config["proximity"])
        config["soft_4rank1"] = keras.layers.deserialize(config["soft_4rank1"])
        return cls(**config)


class RankModelD(StochasticModel):
    """A `SoftRank` model.

    A variational percept layer.

    Gates:
        None

    """

    def __init__(
        self, percept=None, proximity=None, soft_2rank1=None, soft_4rank1=None, **kwargs
    ):
        """Initialize."""
        super(RankModelD, self).__init__(**kwargs)

        self.stimuli_axis = 1

        if percept is None:
            n_stimuli = 20
            n_dim = 3
            kl_weight = 0.1
            prior_scale = 0.2
            embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
                scale_initializer=keras.initializers.Constant(
                    keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
                ),
            )
            embedding_prior = psiz.keras.layers.EmbeddingShared(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
                embedding=psiz.keras.layers.EmbeddingNormalDiag(
                    1,
                    1,
                    loc_initializer=keras.initializers.Constant(0.0),
                    scale_initializer=keras.initializers.Constant(
                        keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
                    ),
                    loc_trainable=False,
                ),
            )
            percept = psiz.keras.layers.EmbeddingVariational(
                posterior=embedding_posterior,
                prior=embedding_prior,
                kl_weight=kl_weight,
                kl_n_sample=30,
            )
        self.percept = percept

        if proximity is None:
            proximity = psiz.keras.layers.Minkowski(
                rho_initializer=keras.initializers.Constant(2.0),
                w_initializer=keras.initializers.Constant(1.0),
                activation=psiz.keras.layers.ExponentialSimilarity(
                    trainable=False,
                    beta_initializer=keras.initializers.Constant(10.0),
                    tau_initializer=keras.initializers.Constant(1.0),
                    gamma_initializer=keras.initializers.Constant(0.0),
                ),
                trainable=False,
            )
        self.proximity = proximity

        if soft_2rank1 is None:
            soft_2rank1 = psiz.keras.layers.SoftRank(n_select=1)
        self.soft_2rank1 = soft_2rank1

        if soft_4rank1 is None:
            soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)
        self.soft_4rank1 = soft_4rank1

    def call(self, inputs):
        """Call."""
        # The 2-rank-1 branch.
        z = self.percept(inputs["given2rank1_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        prob_2rank1 = self.soft_2rank1(s)

        # The 4-rank-1 branch.
        z = self.percept(inputs["given4rank1_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        prob_4rank1 = self.soft_4rank1(s)

        outputs = {
            "given2rank1_outcome": prob_2rank1,
            "given4rank1_outcome": prob_4rank1,
        }
        return outputs

    def get_config(self):
        config = super(RankModelD, self).get_config()
        config.update(
            {
                "percept": keras.saving.serialize_keras_object(self.percept),
                "proximity": keras.saving.serialize_keras_object(self.proximity),
                "soft_2rank1": keras.saving.serialize_keras_object(self.soft_2rank1),
                "soft_4rank1": keras.saving.serialize_keras_object(self.soft_4rank1),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["percept"] = keras.layers.deserialize(config["percept"])
        config["proximity"] = keras.layers.deserialize(config["proximity"])
        config["soft_2rank1"] = keras.layers.deserialize(config["soft_2rank1"])
        config["soft_4rank1"] = keras.layers.deserialize(config["soft_4rank1"])
        return cls(**config)

    def build(self, input_shape):
        super().build(input_shape)
        self._input_shape = input_shape

    def get_build_config(self):
        build_config = {
            "input_shape": self._input_shape,
        }
        return build_config

    def build_from_config(self, config):
        self.build(config["input_shape"])


# finish or move out
# class RankCellModelA(StochasticModel):
#     """A VI RankSimilarityCell model.

#     Variational percept layer.

#     Gates:
#         None

#     """

#     def __init__(self, **kwargs):
#         """Initialize."""
#         super(RankCellModelA, self).__init__(**kwargs)

#         n_stimuli = 20
#         n_dim = 3
#         kl_weight = 0.1
#         prior_scale = 0.2

#         embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
#             n_stimuli + 1,
#             n_dim,
#             mask_zero=True,
#             scale_initializer=keras.initializers.Constant(
#                 keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
#             ),
#         )
#         embedding_prior = psiz.keras.layers.EmbeddingShared(
#             n_stimuli + 1,
#             n_dim,
#             mask_zero=True,
#             embedding=psiz.keras.layers.EmbeddingNormalDiag(
#                 1,
#                 1,
#                 loc_initializer=keras.initializers.Constant(0.0),
#                 scale_initializer=keras.initializers.Constant(
#                     keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
#                 ),
#                 loc_trainable=False,
#             ),
#         )
#         percept = psiz.keras.layers.EmbeddingVariational(
#             posterior=embedding_posterior,
#             prior=embedding_prior,
#             kl_weight=kl_weight,
#             kl_n_sample=30,
#         )
#         proximity = psiz.keras.layers.Minkowski(
#             rho_initializer=keras.initializers.Constant(2.0),
#             w_initializer=keras.initializers.Constant(1.0),
#             activation=psiz.keras.layers.ExponentialSimilarity(
#                 trainable=False,
#                 beta_initializer=keras.initializers.Constant(10.0),
#                 tau_initializer=keras.initializers.Constant(1.0),
#                 gamma_initializer=keras.initializers.Constant(0.0),
#             ),
#             trainable=False,
#         )
#         rank_cell = psiz.keras.layers.RankSimilarityCell(
#             n_reference=8, n_select=2, percept=percept, kernel=proximity
#         )
#         rnn = keras.layers.RNN(rank_cell, return_sequences=True)
#         self.behavior = rnn

#     def call(self, inputs):
#         """Call."""
#         return self.behavior(inputs)

#     def get_config(self):
#         return super(RankCellModelA, self).get_config()


class RateModelA(StochasticModel):
    """A `Logistic` model.

    A variatoinal percept layer.

    Gates:
        None

    """

    def __init__(self, percept=None, proximity=None, rate=None, **kwargs):
        """Initialize."""
        super(RateModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        kl_weight = 0.1
        prior_scale = 0.2
        self.stimuli_axis = 1

        if percept is None:
            embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
                scale_initializer=keras.initializers.Constant(
                    keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
                ),
            )
            embedding_prior = psiz.keras.layers.EmbeddingShared(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
                embedding=psiz.keras.layers.EmbeddingNormalDiag(
                    1,
                    1,
                    loc_initializer=keras.initializers.Constant(0.0),
                    scale_initializer=keras.initializers.Constant(
                        keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
                    ),
                    loc_trainable=False,
                ),
            )
            percept = psiz.keras.layers.EmbeddingVariational(
                posterior=embedding_posterior,
                prior=embedding_prior,
                kl_weight=kl_weight,
                kl_n_sample=30,
            )
        self.percept = percept

        if proximity is None:
            proximity = psiz.keras.layers.Minkowski(
                rho_initializer=keras.initializers.Constant(2.0),
                w_initializer=keras.initializers.Constant(1.0),
                activation=psiz.keras.layers.ExponentialSimilarity(
                    trainable=False,
                    beta_initializer=keras.initializers.Constant(10.0),
                    tau_initializer=keras.initializers.Constant(1.0),
                    gamma_initializer=keras.initializers.Constant(0.0),
                ),
                trainable=False,
            )
        self.proximity = proximity

        if rate is None:
            rate = psiz.keras.layers.Logistic()
        self.rate = rate

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["rate2_stimulus_set"])
        z_0, z_1 = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_0, z_1])
        return self.rate(s)

    def get_config(self):
        config = super(RateModelA, self).get_config()
        config.update(
            {
                "percept": self.percept,
                "proximity": self.proximity,
                "rate": self.rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["percept"] = keras.layers.deserialize(config["percept"])
        config["proximity"] = keras.layers.deserialize(config["proximity"])
        config["rate"] = keras.layers.deserialize(config["rate"])
        return cls(**config)


# TODO finish or move out
# class ALCOVEModelA(StochasticModel):
#     """An `ALCOVECell` model.

#     Gates:
#         None

#     """

#     def __init__(self, **kwargs):
#         """Initialize."""
#         super(ALCOVEModelA, self).__init__(**kwargs)

#         n_stimuli = 20
#         n_dim = 4
#         n_output = 3
#         prior_scale = 0.2

#         percept = psiz.keras.layers.EmbeddingNormalDiag(
#             n_stimuli + 1,
#             n_dim,
#             mask_zero=True,
#             scale_initializer=keras.initializers.Constant(
#                 keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
#             ),
#             trainable=False,
#         )
#         similarity = psiz.keras.layers.ExponentialSimilarity(
#             beta_initializer=keras.initializers.Constant(3.0),
#             tau_initializer=keras.initializers.Constant(1.0),
#             gamma_initializer=keras.initializers.Constant(0.0),
#             trainable=False,
#         )
#         cell = psiz.keras.layers.ALCOVECell(
#             n_output,
#             percept=percept,
#             similarity=similarity,
#             rho_initializer=keras.initializers.Constant(2.0),
#             temperature_initializer=keras.initializers.Constant(1.0),
#             lr_attention_initializer=keras.initializers.Constant(0.03),
#             lr_association_initializer=keras.initializers.Constant(0.03),
#             trainable=False,
#         )
#         rnn = keras.layers.RNN(cell, return_sequences=True, stateful=False)
#         self.behavior = rnn

#     def call(self, inputs):
#         """Call."""
#         return self.behavior(inputs)

#     def get_config(self):
#         return super(ALCOVEModelA, self).get_config()


# TODO finish or move out
# class ALCOVEModelB(StochasticModel):
#     """An `ALCOVECell` model.

#     VI percept layer.

#     Gates:
#         None

#     """

#     def __init__(self, **kwargs):
#         """Initialize."""
#         super(ALCOVEModelB, self).__init__(**kwargs)

#         n_stimuli = 20
#         n_dim = 4
#         n_output = 3
#         kl_weight = 0.1
#         prior_scale = 0.2

#         # VI percept layer
#         embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
#             n_stimuli + 1,
#             n_dim,
#             mask_zero=True,
#             scale_initializer=keras.initializers.Constant(
#                 keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
#             ),
#         )
#         embedding_prior = psiz.keras.layers.EmbeddingShared(
#             n_stimuli + 1,
#             n_dim,
#             mask_zero=True,
#             embedding=psiz.keras.layers.EmbeddingNormalDiag(
#                 1,
#                 1,
#                 loc_initializer=keras.initializers.Constant(0.0),
#                 scale_initializer=keras.initializers.Constant(
#                     keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
#                 ),
#                 loc_trainable=False,
#             ),
#         )
#         percept = psiz.keras.layers.EmbeddingVariational(
#             posterior=embedding_posterior,
#             prior=embedding_prior,
#             kl_weight=kl_weight,
#             kl_n_sample=30,
#         )
#         similarity = psiz.keras.layers.ExponentialSimilarity(
#             beta_initializer=keras.initializers.Constant(3.0),
#             tau_initializer=keras.initializers.Constant(1.0),
#             gamma_initializer=keras.initializers.Constant(0.0),
#             trainable=False,
#         )
#         cell = psiz.keras.layers.ALCOVECell(
#             n_output,
#             percept=percept,
#             similarity=similarity,
#             rho_initializer=keras.initializers.Constant(2.0),
#             temperature_initializer=keras.initializers.Constant(1.0),
#             lr_attention_initializer=keras.initializers.Constant(0.03),
#             lr_association_initializer=keras.initializers.Constant(0.03),
#             trainable=False,
#         )
#         rnn = keras.layers.RNN(cell, return_sequences=True, stateful=False)
#         self.behavior = rnn

#     def call(self, inputs):
#         """Call."""
#         return self.behavior(inputs)

#     def get_config(self):
#         return super(ALCOVEModelB, self).get_config()


def build_ranksim_subclass_a(is_eager):
    """Build subclassed `Model`.

    SoftRank, one group, stochastic (non VI).

    """
    model = RankModelA(n_sample=3)
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
        "run_eagerly": is_eager,
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_b(is_eager):
    """Build subclassed `Model`.

    SoftRank, one group, stochastic (non VI).

    """
    model = RankModelB(n_sample=3)
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_c(is_eager):
    """Build subclassed `Model`.

    SoftRank, gated VI percept.

    """
    model = RankModelC(n_sample=3)
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_d(is_eager):
    """Build subclassed `Model`.

    SoftRank, one group, two outputs, stochastic (VI).

    """
    model = RankModelD(n_sample=3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "given2rank1_outcome": keras.losses.CategoricalCrossentropy(
                name="given2rank1_loss"
            ),
            "given4rank1_outcome": keras.losses.CategoricalCrossentropy(
                name="given4rank1_loss"
            ),
        },
        weighted_metrics={
            "given2rank1_outcome": keras.metrics.CategoricalCrossentropy(
                name="given2rank1_cce"
            ),
            "given4rank1_outcome": keras.metrics.CategoricalCrossentropy(
                name="given4rank1_cce"
            ),
        },
    )
    return model


# TODO finish or move out
# def build_ranksimcell_subclass_a(is_eager):
#     """Build subclassed `Model`.

#     RankSimilarityCell, one group, stochastic (non VI).

#     """
#     model = RankCellModelA(n_sample=3)
#     compile_kwargs = {
#         "loss": keras.losses.CategoricalCrossentropy(),
#         "optimizer": keras.optimizers.Adam(learning_rate=0.001),
#         "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
#     }
#     model.compile(**compile_kwargs)
#     return model


def build_ratesim_subclass_a(is_eager):
    """Build subclassed `Model`."""
    model = RateModelA(n_sample=11)
    compile_kwargs = {
        "loss": keras.losses.MeanSquaredError(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.MeanSquaredError(name="mse")],
    }
    model.compile(**compile_kwargs)
    return model


# def build_alcove_subclass_a(is_eager):
#     """Build subclassed `Model`.

#     ALCOVECell, one group, stochastic (non VI).

#     """
#     model = ALCOVEModelA(n_sample=2)
#     compile_kwargs = {
#         "loss": keras.losses.CategoricalCrossentropy(),
#         "optimizer": keras.optimizers.Adam(learning_rate=0.001),
#         "weighted_metrics": [keras.metrics.CategoricalAccuracy(name="accuracy")],
#     }
#     model.compile(**compile_kwargs)
#     return model


# def build_alcove_subclass_b(is_eager):
#     """Build subclassed `Model`.

#     ALCOVECell, one group, VI percept layer.

#     """
#     model = ALCOVEModelB(n_sample=2)
#     compile_kwargs = {
#         "loss": keras.losses.CategoricalCrossentropy(),
#         "optimizer": keras.optimizers.Adam(learning_rate=0.001),
#         "weighted_metrics": [keras.metrics.CategoricalAccuracy(name="accuracy")],
#     }
#     model.compile(**compile_kwargs)
#     return model


@pytest.fixture(scope="module")
def ds_x2(data_backend):
    """Dataset.

    x = [rank-2]

    """
    n_example = 6
    x_a = np.array(
        [
            [0.1, 1.1, 2.1],
            [0.2, 1.2, 2.2],
            [0.3, 1.3, 2.3],
            [0.4, 1.4, 2.4],
            [0.5, 1.5, 2.5],
            [0.6, 1.6, 2.6],
        ],
        dtype="float32",
    )
    x = {"x_a": x_a}
    y = np.array(
        [
            [10.1, 11.1, 12.1],
            [10.2, 11.2, 12.2],
            [10.3, 11.3, 12.3],
            [10.4, 11.4, 12.4],
            [10.5, 11.5, 12.5],
            [10.6, 11.6, 12.6],
        ],
        dtype="float32",
    )

    w = np.array([1.0, 1.0, 0.2, 1.0, 1.0, 0.8], dtype="float32")
    input_shape = {"x_a": (x_a.shape)}
    case = {
        "x": x,
        "y": y,
        "w": w,
        "batch_size": n_example,
    }
    dataset = _materialize_dataset(case, data_backend)

    return {
        "dataset": dataset,
        "input_shape": input_shape,
        "data_backend": data_backend,
    }


@pytest.fixture(scope="module")
def ds_x2_as_tensor(data_backend):
    """Dataset.

    x = [rank-2]

    """
    n_example = 6
    x = np.array(
        [
            [0.1, 1.1, 2.1],
            [0.2, 1.2, 2.2],
            [0.3, 1.3, 2.3],
            [0.4, 1.4, 2.4],
            [0.5, 1.5, 2.5],
            [0.6, 1.6, 2.6],
        ],
        dtype="float32",
    )
    y = np.array(
        [
            [10.1, 11.1, 12.1],
            [10.2, 11.2, 12.2],
            [10.3, 11.3, 12.3],
            [10.4, 11.4, 12.4],
            [10.5, 11.5, 12.5],
            [10.6, 11.6, 12.6],
        ],
        dtype="float32",
    )

    w = np.array([1.0, 1.0, 0.2, 1.0, 1.0, 0.8], dtype="float32")
    input_shape = x.shape
    case = {
        "x": x,
        "y": y,
        "w": w,
        "batch_size": n_example,
    }
    dataset = _materialize_dataset(case, data_backend)

    return {
        "dataset": dataset,
        "input_shape": input_shape,
        "data_backend": data_backend,
    }


@pytest.fixture(scope="module")
def ds_x2_x2_x2(data_backend):
    """Dataset.

    x = [rank-2,  rank-2, rank-2]

    """
    n_example = 6
    x_a = np.array(
        [
            [0.1, 1.1, 2.1],
            [0.2, 1.2, 2.2],
            [0.3, 1.3, 2.3],
            [0.4, 1.4, 2.4],
            [0.5, 1.5, 2.5],
            [0.6, 1.6, 2.6],
        ],
        dtype="float32",
    )
    x_b = np.array(
        [
            [10.1, 11.1, 12.1],
            [10.2, 11.2, 12.2],
            [10.3, 11.3, 12.3],
            [10.4, 11.4, 12.4],
            [10.5, 11.5, 12.5],
            [10.6, 11.6, 12.6],
        ],
        dtype="float32",
    )
    x_c = np.array(
        [
            [20.1, 21.1, 22.1],
            [20.2, 21.2, 22.2],
            [20.3, 21.3, 22.3],
            [20.4, 21.4, 22.4],
            [20.5, 21.5, 22.5],
            [20.6, 21.6, 22.6],
        ],
        dtype="float32",
    )

    x = {
        "x_a": x_a,
        "x_b": x_b,
        "x_c": x_c,
    }
    y = np.array(
        [
            [10.1, 11.1, 12.1],
            [10.2, 11.2, 12.2],
            [10.3, 11.3, 12.3],
            [10.4, 11.4, 12.4],
            [10.5, 11.5, 12.5],
            [10.6, 11.6, 12.6],
        ],
        dtype="float32",
    )

    w = np.array([1.0, 1.0, 0.2, 1.0, 1.0, 0.8], dtype="float32")
    input_shape = {
        "x_a": x_a.shape,
        "x_b": x_b.shape,
        "x_c": x_c.shape,
    }
    case = {
        "x": x,
        "y": y,
        "w": w,
        "batch_size": n_example,
    }
    dataset = _materialize_dataset(case, data_backend)

    return {
        "dataset": dataset,
        "input_shape": input_shape,
        "data_backend": data_backend,
    }


@pytest.fixture(scope="module")
def ds_x3_x3(data_backend):
    """Dataset.

    x = [rank-3, rank-3]

    """
    n_example = 6
    x_a = np.array(
        [
            [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]],
            [[0.2, 1.2, 2.2], [3.2, 4.2, 5.2]],
            [[0.3, 1.3, 2.3], [3.3, 4.3, 5.3]],
            [[0.4, 1.4, 2.4], [3.4, 4.4, 5.4]],
            [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
            [[0.6, 1.6, 2.6], [3.6, 4.6, 5.6]],
        ],
        dtype="float32",
    )
    x_b = np.array(
        [
            [[10.1, 11.1, 12.1], [13.1, 14.1, 15.1]],
            [[10.2, 11.2, 12.2], [13.2, 14.2, 15.2]],
            [[10.3, 11.3, 12.3], [13.3, 14.3, 15.3]],
            [[10.4, 11.4, 12.4], [13.4, 14.4, 15.4]],
            [[10.5, 11.5, 12.5], [13.5, 14.5, 15.5]],
            [[10.6, 11.6, 12.6], [13.6, 14.6, 15.6]],
        ],
        dtype="float32",
    )

    x = {
        "x_a": x_a,
        "x_b": x_b,
    }
    y = np.array(
        [
            [[10.1, 11.1, 12.1], [10.1, 11.1, 12.1]],
            [[10.2, 11.2, 12.2], [10.2, 11.2, 12.2]],
            [[10.3, 11.3, 12.3], [10.3, 11.3, 12.3]],
            [[10.4, 11.4, 12.4], [10.4, 11.4, 12.4]],
            [[10.5, 11.5, 12.5], [10.5, 11.5, 12.5]],
            [[10.6, 11.6, 12.6], [10.6, 11.6, 12.6]],
        ],
        dtype="float32",
    )

    w = np.array(
        [[1.0, 1.0], [1.0, 1.0], [0.2, 0.2], [1.0, 1.0], [1.0, 1.0], [0.8, 0.8]],
        dtype="float32",
    )
    input_shape = {
        "x_a": x_a.shape,
        "x_b": x_b.shape,
    }
    case = {
        "x": x,
        "y": y,
        "w": w,
        "batch_size": n_example,
    }
    dataset = _materialize_dataset(case, data_backend)

    return {
        "dataset": dataset,
        "input_shape": input_shape,
        "data_backend": data_backend,
    }


def call_fit_evaluate_predict(model, dataset):
    """Simple test of call, fit, evaluate, and predict."""
    # Test isolated call.
    for data in dataset:
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        y_pred = model(x, training=False)

    # Test fit.
    model.fit(dataset, epochs=3)

    # Test evaluate.
    eval0 = model.evaluate(dataset)

    # Test predict.
    pred0 = model.predict(dataset)


class TestControl:
    """Test non-stochastic Control Model."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load(self, ds_x2, is_eager, tmpdir):
        """Test model serialization."""
        dataset = ds_x2["dataset"]

        model = ModelControl()
        compile_kwargs = {
            "loss": keras.losses.MeanSquaredError(),
            "optimizer": keras.optimizers.Adam(learning_rate=0.001),
            "run_eagerly": is_eager,
        }
        model.compile(**compile_kwargs)
        model.fit(dataset, epochs=2)
        result0 = model.evaluate(dataset)
        kernel0 = model.dense_layer.kernel
        bias0 = model.dense_layer.bias
        fp_model = Path(tmpdir) / "test_model.psiz"
        psiz.keras.save_psiz_model(model, fp_model)
        del model

        loaded = psiz.keras.load_psiz_model(
            fp_model,
            custom_objects={"ModelControl": ModelControl},
        )
        result1 = loaded.evaluate(dataset)
        kernel1 = loaded.dense_layer.kernel
        bias1 = loaded.dense_layer.bias

        # Test for model equality.
        assert result0 == result1
        np.testing.assert_allclose(kernel0, kernel1)
        np.testing.assert_allclose(bias0, bias1)


class TestModelA:
    """Test custom ModelA"""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load(self, ds_x2, is_eager, tmpdir):
        """Test model serialization."""
        dataset = ds_x2["dataset"]

        model = ModelA(n_sample=2)
        compile_kwargs = {
            "loss": keras.losses.MeanSquaredError(),
            "optimizer": keras.optimizers.Adam(learning_rate=0.001),
            "run_eagerly": is_eager,
        }
        model.compile(**compile_kwargs)
        model.fit(dataset, epochs=2)
        assert model.n_sample == 2
        results_0 = model.evaluate(dataset, return_dict=True)
        kernel0 = model.dense_layer.kernel
        bias0 = model.dense_layer.bias
        fp_model = Path(tmpdir) / "test_model.psiz"
        psiz.keras.save_psiz_model(model, fp_model)
        del model

        loaded = psiz.keras.load_psiz_model(
            fp_model, custom_objects={"ModelA": ModelA}
        )
        results_1 = loaded.evaluate(dataset, return_dict=True)
        kernel1 = loaded.dense_layer.kernel
        bias1 = loaded.dense_layer.bias

        # Test for model equality.
        assert loaded.n_sample == 2
        assert results_0["loss"] == results_1["loss"]
        np.testing.assert_allclose(kernel0, kernel1)
        np.testing.assert_allclose(bias0, bias1)


class TestModelB:
    """Test custom ModelB"""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_b1(self, ds_x2, is_eager, tmpdir):
        """Test model serialization."""
        dataset = ds_x2["dataset"]

        model = ModelB(n_sample=2)
        compile_kwargs = {
            "loss": keras.losses.MeanSquaredError(),
            "optimizer": keras.optimizers.Adam(learning_rate=0.001),
            "run_eagerly": is_eager,
        }
        model.compile(**compile_kwargs)
        model.fit(dataset, epochs=2)
        assert model.n_sample == 2
        results_0 = model.evaluate(dataset, return_dict=True)
        kernel0 = model.custom_layer.kernel
        fp_model = Path(tmpdir) / "test_model.psiz"
        psiz.keras.save_psiz_model(model, fp_model)
        del model

        loaded = psiz.keras.load_psiz_model(
            fp_model, custom_objects={"ModelB": ModelB}
        )
        results_1 = loaded.evaluate(dataset, return_dict=True)
        kernel1 = loaded.custom_layer.kernel

        # Test for model equality.
        assert loaded.n_sample == 2
        assert results_0["loss"] == results_1["loss"]
        np.testing.assert_allclose(kernel0, kernel1)

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_b2(self, ds_x2_as_tensor, is_eager, tmpdir):
        """Test model serialization."""

        dataset = ds_x2_as_tensor["dataset"]

        model = ModelB2(n_sample=2)
        compile_kwargs = {
            "loss": keras.losses.MeanSquaredError(),
            "optimizer": keras.optimizers.Adam(learning_rate=0.001),
            "run_eagerly": is_eager,
        }
        model.compile(**compile_kwargs)
        model.fit(dataset, epochs=2)
        assert model.n_sample == 2
        results_0 = model.evaluate(dataset, return_dict=True)
        kernel0 = model.custom_layer.kernel
        fp_model = Path(tmpdir) / "test_model.psiz"
        psiz.keras.save_psiz_model(model, fp_model)
        del model

        loaded = psiz.keras.load_psiz_model(
            fp_model,
            custom_objects={"ModelB2": ModelB2},
        )
        results_1 = loaded.evaluate(dataset, return_dict=True)
        kernel1 = loaded.custom_layer.kernel

        # Test for model equality.
        assert loaded.n_sample == 2
        assert results_0["loss"] == results_1["loss"]
        np.testing.assert_allclose(kernel0, kernel1)


class TestModelC:
    """Test custom ModelC"""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage(self, ds_x2_x2_x2, is_eager):
        """Test usage."""

        dataset = ds_x2_x2_x2["dataset"]
        input_shape = ds_x2_x2_x2["input_shape"]
        model = ModelC(n_sample=2)
        model.build(input_shape)
        # TODO why don't we compile?
        # compile_kwargs = {
        #     "loss": keras.losses.MeanSquaredError(),
        #     "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        #     "run_eagerly": is_eager,
        # }
        # model.compile(**compile_kwargs)

        assert model.n_sample == 2

        x0_desired = np.array(
            [
                [0.1, 1.1, 2.1],
                [0.1, 1.1, 2.1],
                [0.2, 1.2, 2.2],
                [0.2, 1.2, 2.2],
                [0.3, 1.3, 2.3],
                [0.3, 1.3, 2.3],
                [0.4, 1.4, 2.4],
                [0.4, 1.4, 2.4],
                [0.5, 1.5, 2.5],
                [0.5, 1.5, 2.5],
                [0.6, 1.6, 2.6],
                [0.6, 1.6, 2.6],
            ],
            dtype="float32",
        )
        x1_desired = np.array(
            [
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
                [10.6, 11.6, 12.6],
            ],
            dtype="float32",
        )
        x2_desired = np.array(
            [
                [20.1, 21.1, 22.1],
                [20.1, 21.1, 22.1],
                [20.2, 21.2, 22.2],
                [20.2, 21.2, 22.2],
                [20.3, 21.3, 22.3],
                [20.3, 21.3, 22.3],
                [20.4, 21.4, 22.4],
                [20.4, 21.4, 22.4],
                [20.5, 21.5, 22.5],
                [20.5, 21.5, 22.5],
                [20.6, 21.6, 22.6],
                [20.6, 21.6, 22.6],
            ],
            dtype="float32",
        )

        y_desired = np.array(
            [
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
                [10.6, 11.6, 12.6],
            ],
            dtype="float32",
        )

        sample_weight_desired = np.array(
            [1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8],
            dtype="float32",
        )

        y_pred_desired = np.array(
            [
                [10.2, 12.2, 14.2],
                [10.2, 12.2, 14.2],
                [10.4, 12.4, 14.4],
                [10.4, 12.4, 14.4],
                [10.6, 12.6, 14.6],
                [10.6, 12.6, 14.6],
                [10.8, 12.8, 14.8],
                [10.8, 12.8, 14.8],
                [11.0, 13.0, 15.0],
                [11.0, 13.0, 15.0],
                [11.2, 13.2, 15.2],
                [11.2, 13.2, 15.2],
            ],
            dtype="float32",
        )

        # Perform a `test_step`.
        for data in dataset:
            x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
            # Adjust `x`, `y` and `sample_weight` batch axis to reflect
            # multiple samples.
            x = model.repeat_samples_in_batch_axis(x, model.n_sample)
            y = model.repeat_samples_in_batch_axis(y, model.n_sample)
            sample_weight = model.repeat_samples_in_batch_axis(
                sample_weight, model.n_sample
            )

            x = {k: keras.ops.convert_to_numpy(v) for k, v in x.items()}
            y = keras.ops.convert_to_numpy(y)
            sample_weight = keras.ops.convert_to_numpy(sample_weight)

            # Assert `x`, `y` and `sample_weight` handled correctly.
            np.testing.assert_allclose(x["x_a"], x0_desired)
            np.testing.assert_allclose(x["x_b"], x1_desired)
            np.testing.assert_allclose(x["x_c"], x2_desired)
            np.testing.assert_allclose(y, y_desired)
            np.testing.assert_allclose(sample_weight, sample_weight_desired)

            y_pred = model(x, training=False)
            y_pred = keras.ops.convert_to_numpy(y_pred)
            # Assert `y_pred` handled correctly.
            np.testing.assert_allclose(y_pred, y_pred_desired, atol=1e-6)

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_nsample_change(self, ds_x2_x2_x2, is_eager):
        """Test model where number of samples changes between use."""

        dataset = ds_x2_x2_x2["dataset"]
        model = ModelC(n_sample=2)
        compile_kwargs = {
            "loss": keras.losses.MeanSquaredError(),
            "optimizer": keras.optimizers.Adam(learning_rate=0.001),
            "run_eagerly": is_eager,
        }
        model.compile(**compile_kwargs)
        model.fit(dataset)

        # Change model's `n_sample` attribute.
        model.n_sample = 5

        # When running model, we now expect the following:
        y_desired = np.array(
            [
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
            ],
            dtype="float32",
        )
        sample_weight_desired = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
            ],
            dtype="float32",
        )
        y_pred_shape_desired = [30, 3]

        # Perform a `test_step` to verify `n_sample` took effect.
        for data in dataset:
            x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
            # Adjust `x`, `y` and `sample_weight` batch axis to reflect
            # multiple samples.
            x = model.repeat_samples_in_batch_axis(x, model.n_sample)
            y = model.repeat_samples_in_batch_axis(y, model.n_sample)
            sample_weight = model.repeat_samples_in_batch_axis(
                sample_weight, model.n_sample
            )
            y = keras.ops.convert_to_numpy(y)
            sample_weight = keras.ops.convert_to_numpy(sample_weight)
            # Assert `y` and `sample_weight` handled correctly.
            # Assert `y` and `sample_weight` handled correctly.
            np.testing.assert_allclose(y, y_desired)
            np.testing.assert_allclose(sample_weight, sample_weight_desired)

            y_pred = model(x, training=False)
            # Assert `y_pred` handled correctly.
            np.testing.assert_allclose(np.shape(y_pred), y_pred_shape_desired)

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load(self, ds_x2_x2_x2, is_eager, tmpdir):
        """Test model serialization."""

        dataset = ds_x2_x2_x2["dataset"]
        model = ModelC(n_sample=7)
        compile_kwargs = {
            "loss": keras.losses.MeanSquaredError(),
            "optimizer": keras.optimizers.Adam(learning_rate=0.001),
            "run_eagerly": is_eager,
        }
        model.compile(**compile_kwargs)

        model.fit(dataset, epochs=2)
        assert model.n_sample == 7
        results_0 = model.evaluate(dataset, return_dict=True)
        branch_0_w0_0 = model.branch_0.w0
        branch_1_w0_0 = model.branch_1.w0

        # Save the model.
        fp_model = Path(tmpdir) / "test_model.psiz"
        psiz.keras.save_psiz_model(model, fp_model)
        del model

        # Load the saved model.
        loaded = psiz.keras.load_psiz_model(
            fp_model, custom_objects={"ModelC": ModelC}
        )
        results_1 = loaded.evaluate(dataset, return_dict=True)
        branch_0_w0_1 = loaded.branch_0.w0
        branch_1_w0_1 = loaded.branch_1.w0

        # Test for model equality.
        assert loaded.n_sample == 7
        assert results_0["loss"] == results_1["loss"]
        np.testing.assert_allclose(branch_0_w0_0, branch_0_w0_1)
        np.testing.assert_allclose(branch_1_w0_0, branch_1_w0_1)


class TestModelD:
    """Test using subclassed `Model` `ModelD`"""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage(self, ds_x3_x3, is_eager):
        """Test with RNN layer."""

        dataset = ds_x3_x3["dataset"]
        input_shape = ds_x3_x3["input_shape"]
        model = ModelD(n_sample=10)
        model.build(input_shape)
        # TODO why don't we compile?

        assert model.n_sample == 10

        # Do a quick test of the Tensor shapes.
        x0_shape_desired = [60, 2, 3]
        x1_shape_desired = [60, 2, 3]
        y_shape_desired = [60, 2, 3]
        w_shape_desired = [60, 2]
        y_pred_shape_desired = [60, 2, 3]

        # Perform a `test_step`.
        for data in dataset:
            x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
            # Adjust `x`, `y` and `sample_weight` batch axis to reflect
            # multiple samples.
            x = model.repeat_samples_in_batch_axis(x, model.n_sample)
            y = model.repeat_samples_in_batch_axis(y, model.n_sample)
            sample_weight = model.repeat_samples_in_batch_axis(
                sample_weight, model.n_sample
            )
            # Assert `x`, `y` and `sample_weight` handled correctly.
            np.testing.assert_allclose(x["x_a"].shape, x0_shape_desired)
            np.testing.assert_allclose(x["x_b"].shape, x1_shape_desired)
            np.testing.assert_allclose(y.shape, y_shape_desired)
            np.testing.assert_allclose(sample_weight.shape, w_shape_desired)

            y_pred = model(x, training=False)
            # Assert `y_pred` handled correctly.
            np.testing.assert_allclose(y_pred.shape, y_pred_shape_desired)

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load(self, ds_x3_x3, is_eager, tmpdir):
        """Test model serialization."""

        dataset = ds_x3_x3["dataset"]
        model = ModelD(n_sample=11)
        compile_kwargs = {
            "loss": keras.losses.MeanSquaredError(),
            "optimizer": keras.optimizers.Adam(learning_rate=0.001),
            "run_eagerly": is_eager,
        }
        model.compile(**compile_kwargs)

        model.fit(dataset, epochs=2)
        assert model.n_sample == 11
        results_0 = model.evaluate(dataset, return_dict=True)
        kernel_0 = model.rnn_layer.cell.layer_0.kernel

        # Save the model.
        fp_model = Path(tmpdir) / "test_model.psiz"
        psiz.keras.save_psiz_model(model, fp_model)
        del model

        # Load the saved model.
        loaded = psiz.keras.load_psiz_model(
            fp_model, custom_objects={"ModelD": ModelD}
        )
        results_1 = loaded.evaluate(dataset, return_dict=True)
        kernel_1 = loaded.rnn_layer.cell.layer_0.kernel

        # Test for model equality.
        assert loaded.n_sample == 11
        assert results_0["loss"] == results_1["loss"]
        np.testing.assert_allclose(kernel_0, kernel_1)


class TestRankSimilarity:
    """Test using `SoftRank` layer."""

    @pytest.mark.backend_tensorflow
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_4rank1_v0, is_eager):
        """Test subclassed `StochasticModel`."""

        ds = ds_4rank1_v0
        model = build_ranksim_subclass_a(is_eager)
        call_fit_evaluate_predict(model, ds)
        keras.backend.clear_session()

    @pytest.mark.backend_tensorflow
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a_mixed_precision(self, ds_4rank1_v0, is_eager, tmpdir):
        """Test subclassed `StochasticModel`."""
        fp_model = Path(tmpdir) / Path("checkpoint.model.keras")

        ds = ds_4rank1_v0
        keras.mixed_precision.set_global_policy("mixed_float16")
        model = build_ranksim_subclass_a(is_eager)
        call_fit_evaluate_predict(model, ds)

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=fp_model,
                monitor="loss",
                save_best_only=True,
                save_weights_only=False,
                mode="min",
            ),
        ]
        history = model.fit(
            ds,
            epochs=3,
            callbacks=callbacks,
            verbose=0,
        )

        # Check that model checkpoint was created.
        assert fp_model.exists()

        keras.mixed_precision.set_global_policy("float32")
        keras.backend.clear_session()

    @pytest.mark.backend_tensorflow
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_subclass_a(self, ds_4rank1_v0, is_eager, tmpdir):
        """Test save/load.

        We change default `n_sample` for a more comprehensive test.

        """

        ds = ds_4rank1_v0
        model = build_ranksim_subclass_a(is_eager)
        # TODO remove?
        # input_shape = {k: v.shape for k, v in ds.element_spec[0].items()}
        # model.build(input_shape)

        # Test initialization settings.
        assert model.n_sample == 3

        # Test propogation of setting `n_sample`.
        model.n_sample = 21
        assert model.n_sample == 21

        model.fit(ds, epochs=1)
        _ = model.evaluate(ds)
        percept_mean = model.percept.embeddings.mean()
        percept_variance = model.percept.embeddings.variance()

        # Test storage serialization.
        fp_model = Path(tmpdir) / "test_model.psiz"
        psiz.keras.save_psiz_model(model, fp_model)
        del model

        # Load the saved model.
        loaded = psiz.keras.load_psiz_model(
            fp_model,
            custom_objects={"RankModelA": RankModelA},
        )
        _ = loaded.evaluate(ds)
        loaded_percept_mean = loaded.percept.embeddings.mean()
        loaded_percept_variance = loaded.percept.embeddings.variance()

        # Test for model equality.
        assert loaded.n_sample == 21
        np.testing.assert_allclose(percept_mean, loaded_percept_mean)
        np.testing.assert_allclose(percept_variance, loaded_percept_variance)
        # NOTE: Don't check loss for equivalence because of
        # stochasticity.

        keras.backend.clear_session()

    @pytest.mark.backend_tensorflow
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_b(self, ds_4rank1_v0, is_eager):
        """Test subclassed `StochasticModel`."""

        ds = ds_4rank1_v0
        model = build_ranksim_subclass_b(is_eager)
        call_fit_evaluate_predict(model, ds)
        keras.backend.clear_session()

    @pytest.mark.backend_tensorflow
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_subclass_b(self, ds_4rank1_v0, is_eager, tmpdir):
        """Test save/load."""

        ds = ds_4rank1_v0
        model = build_ranksim_subclass_b(is_eager)
        # TODO remove?
        # input_shape = {k: v.shape for k, v in ds.element_spec[0].items()}
        # model.build(input_shape)

        # Test initialization settings.
        assert model.n_sample == 3

        # Test propogation of setting `n_sample`.
        model.n_sample = 21
        assert model.n_sample == 21

        model.fit(ds, epochs=1)
        _ = model.evaluate(ds)
        percept_mean = model.percept.embeddings.mean()
        percept_variance = model.percept.embeddings.variance()

        # Test storage serialization.
        fp_model = Path(tmpdir) / "test_model.psiz"
        psiz.keras.save_psiz_model(model, fp_model)
        del model

        # Load the saved model.
        loaded = psiz.keras.load_psiz_model(
            fp_model,
            custom_objects={"RankModelB": RankModelB},
        )
        _ = loaded.evaluate(ds)
        loaded_percept_mean = loaded.percept.embeddings.mean()
        loaded_percept_variance = loaded.percept.embeddings.variance()

        # Test for model equality.
        assert loaded.n_sample == 21
        np.testing.assert_allclose(percept_mean, loaded_percept_mean)
        np.testing.assert_allclose(percept_variance, loaded_percept_variance)
        # NOTE: Don't check loss for equivalence because of
        # stochasticity.

        keras.backend.clear_session()

    @pytest.mark.backend_tensorflow
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_c(self, ds_4rank1_v2, is_eager):
        """Test subclassed `StochasticModel`."""

        ds = ds_4rank1_v2
        model = build_ranksim_subclass_c(is_eager)
        call_fit_evaluate_predict(model, ds)
        keras.backend.clear_session()

    @pytest.mark.backend_tensorflow
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_d(self, ds_2rank1_4rank1_v0, is_eager):
        """Test subclassed `StochasticModel`."""

        ds = ds_2rank1_4rank1_v0
        model = build_ranksim_subclass_d(is_eager)
        call_fit_evaluate_predict(model, ds)
        keras.backend.clear_session()

    @pytest.mark.backend_tensorflow
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_agent_subclass_a(self, ds_4rank1_v0, is_eager):
        """Test usage in 'agent mode'."""

        ds = ds_4rank1_v0
        model = build_ranksim_subclass_a(is_eager)

        def simulate_agent(x):
            depth = 4
            n_sample = 3
            x = model.repeat_samples_in_batch_axis(x, n_sample)
            outcome_probs = model(x)
            outcome_probs = model.average_repeated_samples(outcome_probs, n_sample)
            outcome_idx = keras.random.categorical(
                keras.ops.log(outcome_probs), num_samples=1
            )
            outcome_idx = keras.ops.squeeze(outcome_idx, axis=-1)
            outcome_one_hot = keras.ops.one_hot(outcome_idx, depth)
            return outcome_one_hot

        _ = ds.map(lambda x, y, w: (x, simulate_agent(x), w))

        keras.backend.clear_session()


# TODO finish or move out
# class TestRankSimilarityCell:
#     """Test using `RankSimilarityCell` layer."""

#     @pytest.mark.parametrize(
#         "is_eager",
#         [
#             True,
#             pytest.param(
#                 False,
#                 marks=pytest.mark.xfail(
#                     reason="'add_loss' does not work inside RNN cell."
#                 ),
#             ),
#         ],
#     )
#     def test_usage_subclass_a(self, ds_time_8rank2_v0, is_eager):
#         """Test subclassed `StochasticModel`."""
#

#         ds = ds_time_8rank2_v0
#         model = build_ranksimcell_subclass_a()
#         call_fit_evaluate_predict(model, ds)
#         keras.backend.clear_session()

#     @pytest.mark.xfail(reason="'add_loss' does not work inside RNN cell.")
#     @pytest.mark.parametrize("is_eager", [True, False])
#
#     def test_save_load_subclass_a(
#         self, ds_time_8rank2_v0, is_eager, tmpdir
#     ):
#         """Test save/load."""
#

#         ds = ds_time_8rank2_v0
#         model = build_ranksimcell_subclass_a(is_eager)
#         input_shape = {k: v.shape for k, v in ds.element_spec[0].items()}
#         model.build(input_shape)

#         # Test initialization settings.
#         assert model.n_sample == 3

#         # Test propogation of setting `n_sample`.
#         model.n_sample = 21
#         assert model.n_sample == 21

#         model.fit(ds, epochs=1)
#         percept_mean = model.cell.percept.embeddings.mean()
#         _ = model.evaluate(ds)

#         # Test storage serialization.
#         fp_model = Path(tmpdir) / "test_model.psiz"
#         psiz.keras.save_psiz_model(model, fp_model)
#         del model

#         # Load the saved model.
#         loaded = psiz.keras.load_psiz_model(
#             fp_model, custom_objects={"RankModelB": RankModelB}
#         )
#         loaded_percept_mean = loaded.cell.percept.embeddings.mean()
#         _ = loaded.evaluate(ds)

#         # Test for model equality.
#         assert loaded.n_sample == 21

#         # Check `percept` posterior mean the same.
#         np.testing.assert_allclose(percept_mean, loaded_percept_mean)

#         keras.backend.clear_session()


class TestRateSimilarity:
    """Test using `Logistic` layer."""

    @pytest.mark.backend_tensorflow
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_rate2_v0, is_eager):
        """Test subclassed `StochasticModel`."""

        ds = ds_rate2_v0
        model = build_ratesim_subclass_a(is_eager)
        call_fit_evaluate_predict(model, ds)
        keras.backend.clear_session()

    @pytest.mark.backend_tensorflow
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_subclass_a(self, ds_rate2_v0, is_eager, tmpdir):
        """Test save/load."""

        ds = ds_rate2_v0
        model = build_ratesim_subclass_a(is_eager)
        model.fit(ds, epochs=1)

        # Test stochastic attributes.
        assert model.n_sample == 11

        _ = model.evaluate(ds)
        percept_mean = model.percept.embeddings.mean()

        # Test storage serialization.
        fp_model = Path(tmpdir) / "test_model.psiz"
        psiz.keras.save_psiz_model(model, fp_model)
        del model

        # Load the saved model.
        loaded = psiz.keras.load_psiz_model(
            fp_model,
            custom_objects={"RateModelA": RateModelA},
        )
        _ = loaded.evaluate(ds)

        # Test for model equality.
        assert loaded.n_sample == 11

        # Check `percept` posterior mean the same.
        np.testing.assert_allclose(percept_mean, loaded.percept.embeddings.mean())

        keras.backend.clear_session()


# TODO finish or move out
# class TestALCOVECell:
#     """Test using `ALCOVECell` layer."""

#     @pytest.mark.parametrize("is_eager", [True, False])
#     def test_usage_subclass_a(self, ds_time_categorize_v0, is_eager):
#         """Test subclassed model, one group."""
#

#         ds = ds_time_categorize_v0
#         model = build_alcove_subclass_a(is_eager)
#         call_fit_evaluate_predict(model, ds)
#         keras.backend.clear_session()

#     @pytest.mark.parametrize("is_eager", [True, False])
#     def test_save_load_subclass_a(self, ds_time_categorize_v0, is_eager, tmpdir):
#         """Test save/load."""
#

#         ds = ds_time_categorize_v0
#         model = build_alcove_subclass_a(is_eager)
#         model.fit(ds, epochs=1)

#         # Test initialization settings.
#         assert model.n_sample == 2

#         # Update `n_sample`.
#         model.n_sample = 11
#         _ = model.evaluate(ds)
#         percept_mean = model.behavior.cell.percept.embeddings.mean()

#         # Test storage.
#         fp_model = Path(tmpdir) / "test_model.psiz"
#         psiz.keras.save_psiz_model(model, fp_model)
#         del model
#         # Load the saved model.
#         loaded = psiz.keras.load_psiz_model(
#             fp_model,
#             custom_objects={"ALCOVEModelA": ALCOVEModelA},
#         )
#         _ = loaded.evaluate(ds)

#         # Test for model equality.
#         loaded_percept_mean = loaded.behavior.cell.percept.embeddings.mean()
#         assert loaded.n_sample == 11

#         # Check `percept` posterior mean the same.
#         np.testing.assert_allclose(percept_mean, loaded_percept_mean)

#         keras.backend.clear_session()

#     @pytest.mark.parametrize(
#         "is_eager",
#         [
#             True,
#             pytest.param(
#                 False,
#                 marks=pytest.mark.xfail(
#                     reason="'add_loss' does not work inside RNN cell."
#                 ),
#             ),
#         ],
#     )
#     def test_usage_subclass_b(self, ds_time_categorize_v0, is_eager):
#         """Test subclassed model, one group."""
#

#         ds = ds_time_categorize_v0
#         model = build_alcove_subclass_b(is_eager)
#         call_fit_evaluate_predict(model, ds)
#         keras.backend.clear_session()

#     @pytest.mark.xfail(reason="'add_loss' does not work inside RNN cell.")
#     @pytest.mark.parametrize("is_eager", [True, False])
#     def test_save_load_subclass_b(self, ds_time_categorize_v0, is_eager, tmpdir):
#         """Test save/load."""
#

#         ds = ds_time_categorize_v0
#         model = build_alcove_subclass_b(is_eager)
#         model.fit(ds, epochs=1)

#         # Test initialization settings.
#         assert model.n_sample == 2

#         # Increase `n_sample` to get more consistent evaluations
#         model.n_sample = 11
#         _ = model.evaluate(ds)
#         percept_mean = model.behavior.cell.percept.embeddings.mean()

#         # Test storage.
#         fp_model = Path(tmpdir) / "test_model.psiz"
#         psiz.keras.save_psiz_model(model, fp_model)
#         del model
#         # Load the saved model.
#         loaded = psiz.keras.load_psiz_model(
#             fp_model,
#             custom_objects={"ALCOVEModelA": ALCOVEModelA},
#         )
#         _ = loaded.evaluate(ds)

#         # Test for model equality.
#         assert loaded.n_sample == 11

#         # Check `percept` posterior mean the same.
#         np.testing.assert_allclose(
#             percept_mean, loaded.behavior.cell.percept.embeddings.mean()
#         )

#         keras.backend.clear_session()
