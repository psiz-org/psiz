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
# ==============================================================================

"""Module of custom TensorFlow Layers.

Classes:
    QueryReference:
    Embedding:
    WeightedDistance:
    SeparateAttention:
    Attention:
    InverseKernel:
    ExponentialKernel:
    HeavyTailedKernel:
    StudentsTKernel:

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import layers


class QueryReference(layers.Layer):
    """Model of query reference similarity judgments."""

    def __init__(
            self, coordinate, attention, kernel,
            config_list):
        """Initialize.

        Arguments:
            tf_theta:
            tf_phi:
            tf_z:
            tf_similarity:
            config_list: It is assumed that the indices that will be
                passed in later as inputs will correspond to the
                indices in this data structure.

        """
        super(QueryReference, self).__init__()

        self.coordinate = coordinate
        self.attention = attention
        self.kernel = kernel

        self.n_config = tf.constant(len(config_list))
        self.config_n_select = tf.constant(config_list.n_select.values)
        self.config_is_ranked = tf.constant(config_list.is_ranked.values)
        # TODO REMOVE
        # self.max_n_reference = tf.constant(
        #     np.max(config_list.n_reference.values)
        # )

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A list of inputs:
                stimulus_set: Containing the integers [0, n_stimuli[
                config_idx: Containing the integers [0, n_config[
                group_id: Containing the integers [0, n_group[

        """
        # Inputs.
        obs_stimulus_set = inputs[0]
        obs_config_idx = inputs[1]
        obs_group_id = inputs[2]
        is_present = inputs[4]

        # Expand attention weights.
        attention = self.attention(obs_group_id)

        # Inflate cooridnates.
        z_stimulus_set = self.coordinate(obs_stimulus_set)
        max_n_reference = tf.shape(z_stimulus_set)[2] - 1
        z_q, z_r = tf.split(z_stimulus_set, [1, max_n_reference], 2)

        # Compute similarity between query and references.
        sim_qr = self.kernel([z_q, z_r, attention])

        # Zero out similarities involving placeholder.
        sim_qr = sim_qr * tf.cast(is_present[:, 1:], dtype=K.floatx())

        # Pre-allocate likelihood tensor.
        n_trial = tf.shape(obs_stimulus_set)[0]
        likelihood = tf.zeros([n_trial], dtype=K.floatx())

        # Compute the probability of observations for different trial
        # configurations.
        for i_config in tf.range(self.n_config):
            n_select = self.config_n_select[i_config]
            is_ranked = self.config_is_ranked[i_config]

            # Identify trials belonging to current trial configuration.
            locs = tf.equal(obs_config_idx, i_config)
            trial_idx = tf.squeeze(tf.where(locs))

            # Grab similarities belonging to current trial configuration.
            sim_qr_config = tf.gather(sim_qr, trial_idx)

            # Compute probability of behavior.
            prob_config = _tf_ranked_sequence_probability(
                sim_qr_config, n_select
            )

            # Update master results.
            likelihood = tf.tensor_scatter_nd_update(
                likelihood, tf.expand_dims(trial_idx, axis=1), prob_config
            )

        return likelihood

    def reset_weights(self):
        """Reset trainable variables."""
        pass


class Embedding(layers.Layer):
    """Embedding coordinates.

    Handles a placeholder stimulus using stimulus ID -1.

    """

    # TODO
    # embeddings_initializer='uniform',
    # embeddings_regularizer=None,
    # embeddings_constraint=None,
    def __init__(
            self, n_stimuli, n_dim, fit_z=True, z_min=None,
            z_max=None, **kwargs):
        """Initialize a coordinate layer.

        With no constraints, the coordinates are initialized using a
            using a multivariate Gaussian.

        Arguments:
            n_stimuli:
            n_dim:
            fit_z (optional): Boolean
            embeddings_initializer (optional): Initializer for the
                `z` matrix.
            embeddings_regularizer: Regularizer function applied to
                the `z` matrix.
            embeddings_constraint: Constraint function applied to
                the `z` matrix.
            z_min (optional):
            z_max (optional):

        """
        super(Embedding, self).__init__(**kwargs)

        self.n_stimuli = n_stimuli
        self.n_dim = n_dim

        # self.embeddings_initializer = initializers.get(embeddings_initializer)
        # self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        # self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.z_min = z_min
        self.z_max = z_max

        if z_min is not None and z_max is None:
            z_constraint = GreaterEqualThan(min_value=z_min)
        elif z_min is None and z_max is not None:
            z_constraint = LessEqualThan(max_value=z_max)
        elif z_min is not None and z_max is not None:
            z_constraint = MinMax(min_value, max_value)
        else:
            z_constraint = ProjectZ()

        self.fit_z = fit_z
        self.z = tf.Variable(
            initial_value=self.random_z(), trainable=fit_z,
            name="z", dtype=K.floatx(),
            constraint=z_constraint
        )

    # TODO
    # def build(self, input_shape):

    def call(self, inputs):
        """Call."""
        stimulus_set = inputs + 1  # Add one for placeholder stimulus.  TODO tf.constant(1, dtype=tf.int32)
        z_pad = tf.concat(
            [
                tf.zeros([1, self.z.shape[1]], dtype=K.floatx()),
                self.z
            ], axis=0
        )
        z_stimulus_set = self._tf_inflate_points(stimulus_set, z_pad)
        return z_stimulus_set

    def _tf_inflate_points(self, stimulus_set, z):
        """Inflate stimulus set into embedding points.

        Note: This method will not gracefully handle the masking
        placeholder stimulus ID (i.e., -1). The stimulus IDs and
        coordinates must already have been adjusted for the masking
        placeholder.

        """
        n_trial = tf.shape(stimulus_set)[0]
        input_length = tf.shape(stimulus_set)[1]
        n_dim = tf.shape(z)[1]

        # Flatten stimulus_set and inflate all indices at once.
        flat_idx = tf.reshape(stimulus_set, [-1])
        z_set = tf.gather(z, flat_idx)

        # Reshape and permute dimensions.
        z_set = tf.reshape(z_set, [n_trial, input_length, n_dim])
        z_set = tf.transpose(z_set, perm=[0, 2, 1])
        return z_set

    def reset_weights(self):
        """Reset trainable variables."""
        if self.fit_z:
            self.z.assign(self.random_z())

    def random_z(self):
        """Random z."""
        # TODO Factor out as initializer
        z = RandomEmbedding(
            mean=tf.zeros([self.n_dim], dtype=K.floatx()),
            stdev=tf.ones([self.n_dim], dtype=K.floatx()),
            minval=tf.constant(-3., dtype=K.floatx()),
            maxval=tf.constant(0., dtype=K.floatx()),
            dtype=K.floatx()
        )(shape=[self.n_stimuli, self.n_dim])
        return z

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'n_stimuli': self.n_stimuli, 'n_dim': self.n_dim,
            'fit_z': self.fit_z, 'z_min': self.z_min, 'z_max': self.z_max
        })
        return config


# TODO call or __call__ for Initializer
class RandomEmbedding(initializers.Initializer):
    """Initializer that generates tensors with a normal distribution.

    Arguments:
        mean: A python scalar or a scalar tensor. Mean of the random
            values to generate.
        minval: Minimum value of a uniform random sampler for each
            dimension.
        maxval: Maximum value of a uniform random sampler for each
            dimension.
        seed: A Python integer. Used to create random seeds. See
        `tf.set_random_seed` for behavior.
        dtype: The data type. Only floating point types are supported.

    """

    def __init__(
            self, mean=0.0, stdev=1.0, minval=-3.0, maxval=0.0, seed=None,
            dtype=K.floatx()):
        """Initialize."""
        self.mean = mean
        self.stdev = stdev
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        """Call."""
        # TODO is partition info necessary?
        if dtype is None:
            dtype = self.dtype
        scale = tf.pow(
            tf.constant(10., dtype=dtype),
            tf.random.uniform(
                [1],
                minval=self.minval,
                maxval=self.maxval,
                dtype=dtype,
                seed=self.seed,
                name=None
            )
        )
        stdev = scale * self.stdev
        return tf.random.normal(
            shape, self.mean, stdev, dtype, seed=self.seed)

    def get_config(self):
        """Return configuration."""
        # TODO is this the correct pattern for initializers?
        config = super().get_config()
        config.update({
            "mean": self.mean,
            "stdev": self.stdev,
            "min": self.minval,
            "max": self.maxval,
            "seed": self.seed,
            "dtype": self.dtype.name
        })
        return config


class WeightedDistance(layers.Layer):
    """Weighted Minkowski distance."""

    def __init__(self, fit_rho=True, **kwargs):
        """Initialize.

        Arguments:
            fit_rho (optional): Boolean

        """
        super(WeightedDistance, self).__init__(**kwargs)
        self.fit_rho = fit_rho
        self.rho = tf.Variable(
            initial_value=self.random_rho(),
            trainable=self.fit_rho, name="rho", dtype=K.floatx(),
            constraint=GreaterThan(min_value=1.0)
        )

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: List of inputs.

        """
        z_q = inputs[0]  # Query.
        z_r = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        # Weighted Minkowski distance.
        d_qr = tf.pow(tf.abs(z_q - z_r), self.rho)
        d_qr = tf.multiply(d_qr, w)
        d_qr = tf.pow(tf.reduce_sum(d_qr, axis=1), 1. / self.rho)

        return d_qr

    def reset_weights(self):
        """Reset trainable variables."""
        if self.fit_rho:
            self.rho.assign(self.random_rho())

    def random_rho(self):
        """Random rho."""
        return tf.random_uniform_initializer(1.01, 3.)(shape=[])

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({'fit_rho': self.fit_rho})
        return config


class SeparateAttention(layers.Layer):
    """Attention Layer."""

    def __init__(self, n_dim, n_group=1, fit_group=None, **kwargs):
        """Initialize.

        Arguments:
            n_dim: Integer
            n_group: Integer
            fit_group: Boolean Array
                shape=(n_group,)

        """
        super(SeparateAttention, self).__init__(**kwargs)

        self.n_dim = n_dim
        self.n_group = n_group

        if fit_group is None:
            if self.n_group == 1:
                fit_group = [False]
            else:
                fit_group = np.ones(n_group, dtype=bool)
        self.fit_group = fit_group

        w_list = []
        for i_group in range(self.n_group):
            w_i_name = "w_{0}".format(i_group)
            if self.n_group == 1:
                initial_value = np.ones([1, self.n_dim])
            else:
                initial_value = self.random_w()

            w_i = tf.Variable(
                initial_value=initial_value,
                trainable=fit_group[i_group], name=w_i_name, dtype=K.floatx(),
                constraint=ProjectAttention()
            )
            setattr(self, w_i_name, w_i)
            w_list.append(w_i)
        self.w_list = w_list
        self.concat_layer = tf.keras.layers.Concatenate(axis=0)

    def call(self, inputs):
        """Call.

        Inflate weights by `group_id`.

        Arguments:
            inputs: group_id

        """
        w_all = self.concat_layer(self.w_list)
        w_expand = tf.gather(w_all, inputs)
        w_expand = tf.expand_dims(w_expand, axis=2)
        return w_expand

    def reset_weights(self):
        """Reset trainable variables."""
        w_list = []
        for i_group in range(self.n_group):
            w_i_name = "w_{0}".format(i_group)
            w_i = getattr(self, w_i_name)
            if self.fit_group[i_group]:
                w_i.assign(self.random_w())
            w_list.append(w_i)
        self.w_list = w_list

    def random_w(self):
        """Random w."""
        scale = tf.constant(self.n_dim, dtype=K.floatx())
        alpha = tf.constant(np.ones((self.n_dim)), dtype=K.floatx())
        return RandomAttention(
            alpha, scale, dtype=K.floatx()
        )(shape=[1, self.n_dim])


class Attention(layers.Layer):
    """Attention Layer."""

    def __init__(self, n_dim=None, n_group=1, fit_group=None, **kwargs):
        """Initialize.

        Arguments:
            n_dim: Integer
            n_group (optional): Integer
            fit_group: Boolean Array
                shape=(n_group,)

        """
        super(Attention, self).__init__(**kwargs)

        self.n_dim = n_dim
        self.n_group = n_group

        if fit_group is None:
            if self.n_group == 1:
                fit_group = False
            else:
                fit_group = True
        self.fit_group = fit_group

        if self.n_group == 1:
            initial_value = np.ones([1, self.n_dim])
        else:
            initial_value = self.random_w()

        self.w = tf.Variable(
            initial_value=initial_value,
            trainable=fit_group, name='w', dtype=K.floatx(),
            constraint=ProjectAttention()
        )

    def call(self, inputs):
        """Call.

        Inflate weights by `group_id`.

        Arguments:
            inputs: group_id

        """
        w_expand = tf.gather(self.w, inputs)
        w_expand = tf.expand_dims(w_expand, axis=2)
        return w_expand

    def reset_weights(self):
        """Reset trainable variables."""
        if self.fit_group:
            self.w.assign(self.random_w())

    def random_w(self):
        """Random w."""
        scale = tf.constant(self.n_dim, dtype=K.floatx())
        alpha = tf.constant(np.ones((self.n_dim)), dtype=K.floatx())
        return RandomAttention(
            alpha, scale, dtype=K.floatx()
        )(shape=[self.n_group, self.n_dim])

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'n_dim': self.n_dim, 'n_group': self.n_group,
            'fit_group': self.fit_group
        })
        return config


class InverseKernel(layers.Layer):
    """Inverse-distance similarity kernel.

    This embedding technique uses the following similarity kernel:
        s(x,y) = 1 / norm(x - y, rho)**tau,
    where x and y are n-dimensional vectors. The similarity kernel has
    three free parameters: rho, tau, and mu.

    """

    def __init__(self, fit_rho=True, fit_tau=True, fit_mu=True, **kwargs):
        """Initialize.

        Arguments:
            fit_tau (optional): Boolean
            fit_gamme (optional): Boolean
            fit_beta (optional): Boolean

        """
        super(InverseKernel, self).__init__(**kwargs)
        self.distance_layer = WeightedDistance(fit_rho=fit_rho)
        self.rho = self.distance_layer.rho

        self.fit_tau = fit_tau
        self.tau = tf.Variable(
            initial_value=self.random_tau(),
            trainable=fit_tau, name="tau", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=1.0)
        )

        self.fit_mu = fit_mu
        self.mu = tf.Variable(
            initial_value=self.random_mu(),
            trainable=fit_mu, name="mu", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=2.2204e-16)
        )

        self.theta = {
            'rho': self.distance_layer.rho,
            'tau': self.tau,
            'mu': self.mu
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs:

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        z_q = inputs[0]  # Query.
        z_r = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        d_qr = self.distance_layer([z_q, z_r, w])

        # Exponential family similarity function.
        sim_qr = 1 / (tf.pow(d_qr, self.tau) + self.mu)
        return sim_qr

    def reset_weights(self):
        """Reset trainable variables."""
        self.distance_layer.reset_weights()

        if self.fit_tau:
            self.tau.assign(self.random_tau())

        if self.fit_mu:
            self.mu.assign(self.random_mu())

    def random_tau(self):
        """Random tau."""
        return tf.random_uniform_initializer(1., 2.)(shape=[])

    def random_mu(self):
        """Random mu."""
        return tf.random_uniform_initializer(0.0000000001, .001)(shape=[])

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_rho': self.distance_layer.fit_rho, 'fit_tau': self.fit_tau,
            'fit_mu': self.fit_mu
        })
        return config


class ExponentialKernel(layers.Layer):
    """Exponential family similarity kernel.

    This embedding technique uses the following similarity kernel:
        s(x,y) = exp(-beta .* norm(x - y, rho).^tau) + gamma,
    where x and y are n-dimensional vectors. The similarity kernel has
    four free parameters: rho, tau, gamma, and beta. The exponential
    family is obtained by integrating across various psychological
    theories [1,2,3,4].

    References:
        [1] Jones, M., Love, B. C., & Maddox, W. T. (2006). Recency
            effects as a window to generalization: Separating
            decisional and perceptual sequential effects in category
            learning. Journal of Experimental Psychology: Learning,
            Memory, & Cognition, 32 , 316-332.
        [2] Jones, M., Maddox, W. T., & Love, B. C. (2006). The role of
            similarity in generalization. In Proceedings of the 28th
            annual meeting of the cognitive science society (pp. 405-
            410).
        [3] Nosofsky, R. M. (1986). Attention, similarity, and the
            identification-categorization relationship. Journal of
            Experimental Psychology: General, 115, 39-57.
        [4] Shepard, R. N. (1987). Toward a universal law of
            generalization for psychological science. Science, 237,
            1317-1323.

    """

    def __init__(
            self, fit_rho=True, fit_tau=True, fit_gamma=True, fit_beta=True,
            **kwargs):
        """Initialize.

        Arguments:
            fit_rho (optional): Boolean
            fit_tau (optional): Boolean
            fit_gamme (optional): Boolean
            fit_beta (optional): Boolean

        """
        super(ExponentialKernel, self).__init__(**kwargs)
        self.distance_layer = WeightedDistance(fit_rho=fit_rho)
        self.rho = self.distance_layer.rho

        self.fit_tau = fit_tau
        self.tau = tf.Variable(
            initial_value=self.random_tau(),
            trainable=self.fit_tau, name="tau", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=1.0)
        )

        self.fit_gamma = fit_gamma
        self.gamma = tf.Variable(
            initial_value=self.random_gamma(),
            trainable=self.fit_gamma, name="gamma", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=0.0)
        )

        self.fit_beta = fit_beta
        self.beta = tf.Variable(
            initial_value=self.random_beta(),
            trainable=self.fit_beta, name="beta", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=1.0)
        )

        self.theta = {
            'rho': self.distance_layer.rho,
            'tau': self.tau,
            'gamma': self.gamma,
            'beta': self.beta
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs:
                z_q: A set of embedding points.
                    shape = (n_trial, n_dim [, n_sample])
                z_r: A set of embedding points.
                    shape = (n_trial, n_dim [, n_sample])
                attention: The weights allocated to each dimension
                    in a weighted minkowski metric.
                    shape = (n_trial, n_dim [, n_sample])

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        z_q = inputs[0]  # Query.
        z_r = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        d_qr = self.distance_layer([z_q, z_r, w])

        # Exponential family similarity function.
        sim_qr = tf.exp(
            tf.negative(self.beta) * tf.pow(d_qr, self.tau)
        ) + self.gamma
        return sim_qr

    def reset_weights(self):
        """Reset trainable variables."""
        self.distance_layer.reset_weights()

        if self.fit_tau:
            self.tau.assign(self.random_tau())

        if self.fit_gamma:
            self.gamma.assign(self.random_gamma())

        if self.fit_beta:
            self.beta.assign(self.random_beta())

    def random_tau(self):
        """Random tau."""
        return tf.random_uniform_initializer(1., 2.)(shape=[])

    def random_gamma(self):
        """Random gamma."""
        return tf.random_uniform_initializer(0., .001)(shape=[])

    def random_beta(self):
        """Random beta."""
        return tf.random_uniform_initializer(1., 30.)(shape=[])

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_rho': self.distance_layer.fit_rho, 'fit_tau': self.fit_tau,
            'fit_gamma': self.fit_gamma, 'fit_beta': self.fit_beta
        })
        return config


class HeavyTailedKernel(layers.Layer):
    """Heavy-tailed family similarity kernel.

    This embedding technique uses the following similarity kernel:
        s(x,y) = (kappa + (norm(x-y, rho).^tau)).^(-alpha),
    where x and y are n-dimensional vectors. The similarity kernel has
    four free parameters: rho, tau, kappa, and alpha. The
    heavy-tailed family is a generalization of the Student-t family.

    """

    def __init__(
            self, fit_rho=True, fit_tau=True, fit_kappa=True, fit_alpha=True,
            **kwargs):
        """Initialize.

        Arguments:
            fit_rho (optional): Boolean
            fit_tau (optional): Boolean
            fit_kappa (optional): Boolean
            fit_alpha (optional): Boolean

        """
        super(HeavyTailedKernel, self).__init__(**kwargs)
        self.distance_layer = WeightedDistance(fit_rho=fit_rho)
        self.rho = self.distance_layer.rho

        self.fit_tau = fit_tau
        self.tau = tf.Variable(
            initial_value=self.random_tau(),
            trainable=fit_tau, name="tau", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=1.0)
        )

        self.fit_kappa = fit_kappa
        self.kappa = tf.Variable(
            initial_value=self.random_kappa(),
            trainable=fit_kappa, name="kappa", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=0.0)
        )

        self.fit_alpha = fit_alpha
        self.alpha = tf.Variable(
            initial_value=self.random_alpha(),
            trainable=fit_alpha, name="alpha", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=0.0)
        )

        self.theta = {
            'rho': self.distance_layer.rho,
            'tau': self.tau,
            'kappa': self.kappa,
            'alpha': self.alpha
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs:

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        z_q = inputs[0]  # Query.
        z_r = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        d_qr = self.distance_layer([z_q, z_r, w])

        # Heavy-tailed family similarity function.
        sim_qr = tf.pow(
            self.kappa + tf.pow(d_qr, self.tau), (tf.negative(self.alpha))
        )
        return sim_qr

    def reset_weights(self):
        """Reset trainable variables."""
        self.distance_layer.reset_weights()

        if self.fit_tau:
            self.tau.assign(self.random_tau())

        if self.fit_kappa:
            self.kappa.assign(self.random_kappa())

        if self.fit_alpha:
            self.alpha.assign(self.random_alpha())

    def random_tau(self):
        """Random tau."""
        return tf.random_uniform_initializer(1., 2.)(shape=[])

    def random_kappa(self):
        """Random kappa."""
        return tf.random_uniform_initializer(1., 11.)(shape=[])

    def random_alpha(self):
        """Random alpha."""
        return tf.random_uniform_initializer(10., 60.)(shape=[])

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_rho': self.distance_layer.fit_rho, 'fit_tau': self.fit_tau,
            'fit_kappa': self.fit_kappa, 'fit_alpha': self.fit_alpha
        })
        return config


class StudentsTKernel(layers.Layer):
    """Student's t-distribution similarity kernel.

    The embedding technique uses the following similarity kernel:
        s(x,y) = (1 + (((norm(x-y, rho)^tau)/alpha))^(-(alpha + 1)/2),
    where x and y are n-dimensional vectors. The similarity kernel has
    three free parameters: rho, tau, and alpha. The original Student-t
    kernel proposed by van der Maaten [1] uses the parameter settings
    rho=2, tau=2, and alpha=n_dim-1. By default, all variables are fit
    to the data. This behavior can be changed by setting the
    appropriate fit_<var_name>=False.

    References:
    [1] van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic
        triplet embedding. In Machine learning for signal processing
        (MLSP), 2012 IEEE international workshop on (p. 1-6).
        doi:10.1109/MLSP.2012.6349720

    """

    def __init__(
            self, n_dim=None, fit_rho=True, fit_tau=True, fit_alpha=True, **kwargs):
        """Initialize.

        Arguments:
            n_dim:  Integer indicating the dimensionality of the embedding.
            fit_rho (optional): Boolean
            fit_tau (optional): Boolean
            fit_alpha (optional): Boolean

        """
        super(StudentsTKernel, self).__init__(**kwargs)
        self.distance_layer = WeightedDistance(fit_rho=fit_rho)
        self.rho = self.distance_layer.rho

        self.n_dim = n_dim

        self.fit_tau = fit_tau
        self.tau = tf.Variable(
            initial_value=self.random_tau(),
            trainable=fit_tau, name="tau", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=1.0)
        )

        self.fit_alpha = fit_alpha
        self.alpha = tf.Variable(
            initial_value=self.random_alpha(),
            trainable=fit_alpha, name="alpha", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=0.000001)
        )

        self.theta = {
            'rho': self.distance_layer.rho,
            'tau': self.tau,
            'alpha': self.alpha
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs:

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        z_q = inputs[0]  # Query.
        z_r = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        d_qr = self.distance_layer([z_q, z_r, w])

        # Student-t family similarity kernel.
        sim_qr = tf.pow(
            1 + (tf.pow(d_qr, self.tau) / self.alpha), tf.negative(self.alpha + 1)/2
        )
        return sim_qr

    def reset_weights(self):
        """Reset trainable variables."""
        self.distance_layer.reset_weights()

        if self.fit_tau:
            self.tau.assign(self.random_tau())

        if self.fit_alpha:
            self.alpha.assign(self.random_alpha())

    def random_tau(self):
        """Random tau."""
        return tf.random_uniform_initializer(1., 2.)(shape=[])

    def random_alpha(self):
        """Random alpha."""
        alpha_min = np.max((1, self.n_dim - 2.))
        alpha_max = self.n_dim + 2.
        return tf.random_uniform_initializer(alpha_min, alpha_max)(shape=[])

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_rho': self.distance_layer.fit_rho, 'fit_tau': self.fit_tau,
            'fit_alpha': self.fit_alpha
        })
        return config


class RandomAttention(initializers.Initializer):
    """Initializer that generates tensors for attention weights.

    Arguments:
        concentration: An array indicating the concentration
            parameters (i.e., alpha values) governing a Dirichlet
            distribution.
        scale: Scalar indicating how the Dirichlet sample should be scaled.
        dtype: The data type. Only floating point types are supported.

    """

    def __init__(self, concentration, scale=1.0, dtype=K.floatx()):
        """Initialize."""
        self.concentration = concentration
        self.scale = scale
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        """Call."""
        if dtype is None:
            dtype = self.dtype
        dist = tfp.distributions.Dirichlet(self.concentration)
        return self.scale * dist.sample([shape[0]])

    def get_config(self):
        """Return configuration."""
        return {
            "concentration": self.concentration,
            "dtype": self.dtype.name
        }


# TODO call or __call__ for constraint
class GreaterThan(constraints.Constraint):
    """Constrains the weights to be greater than a value."""

    def __init__(self, min_value=0.):
        """Initialize."""
        self.min_value = min_value

    def __call__(self, w):
        """Call."""
        w = w - self.min_value
        w = w * tf.cast(tf.math.greater(w, 0.), K.floatx())
        w = w + self.min_value
        return w


class LessThan(constraints.Constraint):
    """Constrains the weights to be less than a value."""

    def __init__(self, max_value=0.):
        """Initialize."""
        self.max_value = max_value

    def __call__(self, w):
        """Call."""
        w = w - self.max_value
        w = w * tf.cast(tf.math.greater(0., w), K.floatx())
        w = w + self.max_value
        return w


class GreaterEqualThan(constraints.Constraint):
    """Constrains the weights to be greater/equal than a value."""

    def __init__(self, min_value=0.):
        """Initialize."""
        self.min_value = min_value

    def __call__(self, w):
        """Call."""
        w = w - self.min_value
        w = w * tf.cast(tf.math.greater_equal(w, 0.), K.floatx())
        w = w + self.min_value
        return w


class LessEqualThan(constraints.Constraint):
    """Constrains the weights to be greater/equal than a value."""

    def __init__(self, max_value=0.):
        """Initialize."""
        self.max_value = max_value

    def __call__(self, w):
        """Call."""
        w = w - self.max_value
        w = w * tf.cast(tf.math.greater_equal(0., w), K.floatx())
        w = w + self.max_value
        return w


class MinMax(constraints.Constraint):
    """Constrains the weights to be between/equal values."""

    def __init__(self, min_value, max_value):
        """Initialize."""
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        """Call."""
        w = w - self.min_value
        w = w * tf.cast(tf.math.greater_equal(w, 0.), K.floatx())
        w = w + self.min_value

        w = w - self.max_value
        w = w * tf.cast(tf.math.greater_equal(0., w), K.floatx())
        w = w + self.max_value

        return w


class ProjectZ(constraints.Constraint):
    """Constrains the embedding to be zero-centered.

    Constraint is used to improve numerical stability.
    """

    def __init__(self):
        """Initialize."""

    def __call__(self, tf_z):
        """Call."""
        tf_mean = tf.reduce_mean(tf_z, axis=0, keepdims=True)
        tf_z_centered = tf_z - tf_mean
        return tf_z_centered


class ProjectAttention(constraints.Constraint):
    """Return projection of attention weights."""

    def __init__(self):
        """Initialize."""

    def __call__(self, tf_attention_0):
        """Call."""
        n_dim = tf.shape(tf_attention_0, out_type=K.floatx())[1]
        tf_attention_1 = tf.divide(
            tf.reduce_sum(tf_attention_0, axis=1, keepdims=True), n_dim
        )
        tf_attention_proj = tf.divide(
            tf_attention_0, tf_attention_1
        )
        return tf_attention_proj


def _tf_ranked_sequence_probability(sim_qr, n_select):
    """Return probability of a ranked selection sequence.

    See: _ranked_sequence_probability

    Arguments:
        sim_qr: A tensor containing the precomputed similarities
            between the query stimuli and corresponding reference
            stimuli.
            shape = (n_trial, n_reference)
        n_select: A scalar indicating the number of selections
            made for each trial.

    """
    # Initialize.
    n_trial = tf.shape(sim_qr)[0]
    seq_prob = tf.ones([n_trial], dtype=K.floatx())
    selected_idx = n_select - 1
    denom = tf.reduce_sum(sim_qr[:, selected_idx:], axis=1)

    for i_selected in tf.range(selected_idx, -1, -1):
        # Compute selection probability.
        prob = tf.divide(sim_qr[:, i_selected], denom)
        # Update sequence probability.
        seq_prob = tf.multiply(seq_prob, prob)
        # Update denominator in preparation for computing the probability
        # of the previous selection in the sequence.
        if i_selected > tf.constant(0, dtype=tf.int32):
            denom = tf.add(denom, sim_qr[:, i_selected - 1])
        seq_prob.set_shape([None])
    return seq_prob
