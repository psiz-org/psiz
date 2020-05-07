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

"""Module of custom TensorFlow layers.

Classes:
    EmbeddingRe: An Embedding layer.
    WeightedMinkowski: A weighted distance layer.
    Attention: A simple attention layer.
    InverseKernel: A parameterized inverse similarity layer.
    ExponentialKernel: A parameterized exponential similarity layer.
    HeavyTailedKernel: A parameterized heavy-tailed similarity layer.
    StudentsTKernel: A parameterized Student's t-distribution
        similarity layer.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import psiz.keras.constraints as pk_constraints
import psiz.keras.initializers as pk_initializers
import psiz.keras.regularizers


class EmbeddingRe(tf.keras.layers.Layer):
    """Embedding coordinates.

    The embeddings are stored in the variable `z` that has
    shape=(input_dim, output_dim). Handles a placeholder stimulus using
    stimulus ID "-1".

    Arguments:
        input_dim: An integer indicating the total number of unique
            stimuli that will be embedded. This must be equal to or
            greater than three.
        output_dim: An integer indicating the dimensionality of the
            embeddings. Must be equal to or greater than one.
        embeddings_initializer (optional): Initializer for the `z`
            matrix. By default, the coordinates are intialized using a
            multivariate Gaussian at various scales.
        embeddings_regularizer (optional): Regularizer function applied
            to the `z` matrix.
        embeddings_constraint (optional): Constraint function applied
            to the `z` matrix. By default, a constraint will be used
            that zero-centers the centroid of the embedding to promote
            numerical stability.
        kwargs: See tf.keras.layers.Layer.

    Raises:
        ValueError: If `input_dim` or `output_dim` arguments are
            invalid.

    """

    def __init__(
            self, input_dim, output_dim, embeddings_initializer=None,
            embeddings_regularizer=None, embeddings_constraint=None,
            **kwargs):
        """Initialize."""
        super(EmbeddingRe, self).__init__(**kwargs)

        if (input_dim < 3):
            raise ValueError("There must be at least three stimuli.")
        self.input_dim = input_dim

        if (output_dim < 1):
            raise ValueError(
                "The output dimensionality (`output_dim`) must be an integer "
                "greater than 0."
            )
        self.output_dim = output_dim

        # Handle initializer.
        if embeddings_initializer is None:
            embeddings_initializer = pk_initializers.RandomScaleMVN(
                minval=-4., maxval=-2.
            )
        self.embeddings_initializer = tf.keras.initializers.get(
            embeddings_initializer
        )

        # Handle regularizer.
        self.embeddings_regularizer = tf.keras.regularizers.get(
            embeddings_regularizer
        )

        # Handle constraints.
        if embeddings_constraint is None:
            embeddings_constraint = pk_constraints.Center(axis=0)
        self.embeddings_constraint = tf.keras.constraints.get(
            embeddings_constraint
        )

        # TODO not sure if trainable should be set this way or let
        # superclass logic handle it?
        self.z = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            trainable=self.trainable, name='z', dtype=K.floatx(),
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint
        )

    def call(self, inputs):
        """Call."""
        # Add one for placeholder stimulus.
        stimulus_set = inputs + tf.constant(1, dtype=inputs.dtype)
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
        batch_size = tf.shape(stimulus_set)[0]
        input_length = tf.shape(stimulus_set)[1]
        output_dim = tf.shape(z)[1]

        # Flatten stimulus_set and inflate all indices at once.
        flat_idx = tf.reshape(stimulus_set, [-1])
        z_set = tf.gather(z, flat_idx)

        # Reshape and permute dimensions.
        z_set = tf.reshape(z_set, [batch_size, input_length, output_dim])
        z_set = tf.transpose(z_set, perm=[0, 2, 1])
        return z_set

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer':
                tf.keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer':
                tf.keras.regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint':
                tf.keras.constraints.serialize(self.embeddings_constraint)
        })
        config = _updated_config(self, config)
        return config


class WeightedMinkowski(tf.keras.layers.Layer):
    """Weighted Minkowski distance.

    Arguments:
        fit_rho (optional): Boolean indicating if variable is
            trainable.
        rho_init (optional): Initializer for rho.

    """

    def __init__(self, fit_rho=True, rho_init=None, **kwargs):
        """Initialize."""
        super(WeightedMinkowski, self).__init__(**kwargs)
        self.fit_rho = fit_rho

        if rho_init is None:
            rho_init = tf.random_uniform_initializer(1.01, 3.)
        self.rho_init = tf.keras.initializers.get(rho_init)
        self.rho = self.add_weight(
            shape=[], initializer=self.rho_init,
            trainable=self.fit_rho, name="rho", dtype=K.floatx(),
            constraint=pk_constraints.GreaterThan(min_value=1.0)
        )

        self.theta = {'rho': self.rho}

    def call(self, inputs):
        """Call.

        Arguments:
            inputs:
                z_q: A set of embedding points.
                    shape = (batch_size, n_dim [, n_sample])
                z_r: A set of embedding points.
                    shape = (batch_size, n_dim [, n_sample])
                w: The weights allocated to each dimension
                    in a weighted minkowski metric.
                    shape = (batch_size, n_dim [, n_sample])

        """
        z_q = inputs[0]  # Query.
        z_r = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        # Weighted Minkowski distance.
        d_qr = tf.pow(tf.abs(z_q - z_r), self.rho)
        d_qr = tf.multiply(d_qr, w)
        d_qr = tf.pow(tf.reduce_sum(d_qr, axis=1), 1. / self.rho)

        return d_qr

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_rho': self.fit_rho,
            'rho_init': tf.keras.initializers.serialize(self.rho_init)
        })
        config = _updated_config(self, config)
        return config


class Attention(tf.keras.layers.Layer):
    """Attention Layer.

    Arguments:
        n_dim: An integer indicating the dimensionality of the
            embeddings. Must be equal to or greater than one.
        n_group (optional): An integer indicating the number of
            different population groups in the embedding. A separate
            set of attention weights will be inferred for each group.
            Must be equal to or greater than one.
        fit_group: Boolean indicating if variable is trainable.
            shape=(n_group,)

    Raises:
        ValueError: If `n_dim` or `n_group` arguments are invalid.

    """

    def __init__(
            self, n_group=1, n_dim=None, fit_group=None,
            embeddings_initializer=None, embeddings_regularizer=None,
            embeddings_constraint=None, **kwargs):
        """Initialize."""
        super(Attention, self).__init__(**kwargs)

        if (n_group < 1):
            raise ValueError(
                "The number of groups (`n_group`) must be an integer greater "
                "than 0."
            )
        self.n_group = n_group

        if (n_dim < 1):
            raise ValueError(
                "The dimensionality (`n_dim`) must be an integer "
                "greater than 0."
            )
        self.n_dim = n_dim

        # Handle initializer.
        if embeddings_initializer is None:
            if self.n_group == 1:
                embeddings_initializer = tf.keras.initializers.Ones()
            else:
                scale = self.n_dim
                alpha = np.ones((self.n_dim))
                embeddings_initializer = pk_initializers.RandomAttention(
                    alpha, scale
                )
        self.embeddings_initializer = tf.keras.initializers.get(
            embeddings_initializer
        )

        # Handle regularizer.
        self.embeddings_regularizer = tf.keras.regularizers.get(
            embeddings_regularizer
        )

        # Handle constraints.
        if embeddings_constraint is None:
            embeddings_constraint = pk_constraints.NonNegNorm(
                scale=self.n_dim
            )
        self.embeddings_constraint = tf.keras.constraints.get(
            embeddings_constraint
        )

        if fit_group is None:
            if self.n_group == 1:
                fit_group = False
            else:
                fit_group = True
        self.fit_group = fit_group

        self.w = self.add_weight(
            shape=(self.n_group, self.n_dim),
            initializer=self.embeddings_initializer,
            trainable=fit_group, name='w', dtype=K.floatx(),
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint
        )

    def call(self, inputs):
        """Call.

        Inflate weights by `group_id`.

        Arguments:
            inputs: A Tensor denoting `group_id`.

        """
        return tf.gather(self.w, inputs)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'n_group': self.n_group,
            'n_dim': self.n_dim,
            'fit_group': self.fit_group,
            'embeddings_initializer':
                tf.keras.initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer':
                tf.keras.regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint':
                tf.keras.constraints.serialize(self.embeddings_constraint)
        })
        config = _updated_config(self, config)
        return config


class InverseKernel(tf.keras.layers.Layer):
    """Inverse-distance similarity function.

    The inverse-distance similarity function is parameterized as:
        s(x,y) = 1 / (d(x,y)**tau + mu),
    where x and y are n-dimensional vectors.

    Arguments:
        fit_tau (optional): Boolean indicating if variable is
            trainable.
        fit_gamma (optional): Boolean indicating if variable is
            trainable.
        fit_beta (optional): Boolean indicating if variable is
            trainable.

    """

    def __init__(
            self, fit_tau=True, fit_mu=True, tau_init=None, mu_init=None,
            **kwargs):
        """Initialize."""
        super(InverseKernel, self).__init__(**kwargs)

        self.fit_tau = fit_tau
        if tau_init is None:
            tau_init = tf.random_uniform_initializer(1., 2.)
        self.tau_init = tf.keras.initializers.get(tau_init)
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_init, trainable=self.fit_tau,
            name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_mu = fit_mu
        if mu_init is None:
            mu_init = tf.random_uniform_initializer(0.0000000001, .001)
        self.mu_init = tf.keras.initializers.get(tau_init)
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_int, trainable=self.fit_mu,
            name="mu", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=2.2204e-16)
        )

        self.theta = {
            'tau': self.tau,
            'mu': self.mu
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A tensor of distances.

        Returns:
            A tensor of similarities.

        """
        return 1 / (tf.pow(inputs, self.tau) + self.mu)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_tau': self.fit_tau,
            'fit_mu': self.fit_mu,
            'tau_init': tf.keras.initializers.serialize(self.tau_init),
            'mu_init': tf.keras.initializers.serialize(self.mu_init),
        })
        config = _updated_config(self, config)
        return config


class ExponentialKernel(tf.keras.layers.Layer):
    """Exponential family similarity function.

    This exponential-family similarity function is parameterized as:
        s(x,y) = exp(-beta .* d(x,y).^tau) + gamma,
    where x and y are n-dimensional vectors. The exponential family
    function is obtained by integrating across various psychological
    theories [1,2,3,4].

    By default beta=10. and is not trainable to prevent redundancy with
    trainable embeddings and to prevent short-circuiting any
    regularizers placed on the embeddings.

    Arguments:
        fit_tau (optional): Boolean indicating if variable is
            trainable.
        fit_gamma (optional): Boolean indicating if variable is
            trainable.
        fit_beta (optional): Boolean indicating if variable is
            trainable.

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
            self, fit_tau=True, fit_gamma=True, fit_beta=False, tau_init=None,
            gamma_init=None, beta_init=None, **kwargs):
        """Initialize."""
        super(ExponentialKernel, self).__init__(**kwargs)

        self.fit_tau = fit_tau
        if tau_init is None:
            tau_init = tf.random_uniform_initializer(1., 2.)
        self.tau_init = tf.keras.initializers.get(tau_init)
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_init, trainable=self.fit_tau,
            name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_gamma = fit_gamma
        if gamma_init is None:
            gamma_init = tf.random_uniform_initializer(0., .001)
        self.gamma_init = tf.keras.initializers.get(gamma_init)
        self.gamma = self.add_weight(
            shape=[], initializer=self.gamma_init, trainable=self.fit_gamma,
            name="gamma", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
        )

        self.fit_beta = fit_beta
        if beta_init is None:
            if fit_beta:
                beta_init = tf.random_uniform_initializer(1., 30.)
            else:
                beta_init = tf.keras.initializers.Constant(value=10.)
        self.beta_init = tf.keras.initializers.get(beta_init)
        self.beta = self.add_weight(
            shape=[], initializer=self.beta_init, trainable=self.fit_beta,
            name="beta", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)

        )

        self.theta = {
            'tau': self.tau,
            'gamma': self.gamma,
            'beta': self.beta
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A tensor of distances.

        Returns:
            A tensor of similarities.

        """
        return tf.exp(
            tf.negative(self.beta) * tf.pow(inputs, self.tau)
        ) + self.gamma

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_tau': self.fit_tau,
            'fit_gamma': self.fit_gamma,
            'fit_beta': self.fit_beta,
            'tau_init': tf.keras.initializers.serialize(self.tau_init),
            'gamma_init': tf.keras.initializers.serialize(self.gamma_init),
            'beta_init': tf.keras.initializers.serialize(self.beta_init),
        })
        config = _updated_config(self, config)
        return config


class HeavyTailedKernel(tf.keras.layers.Layer):
    """Heavy-tailed family similarity function.

    The heavy-tailed similarity function is parameterized as:
        s(x,y) = (kappa + (d(x,y).^tau)).^(-alpha),
    where x and y are n-dimensional vectors. The heavy-tailed family is
    a generalization of the Student-t family.

    Arguments:
        fit_tau (optional): Boolean indicating if variable is
            trainable.
        fit_kappa (optional): Boolean indicating if variable is
            trainable.
        fit_alpha (optional): Boolean indicating if variable is
            trainable.

    """

    def __init__(
            self, fit_tau=True, fit_kappa=True, fit_alpha=True, tau_init=None,
            kappa_init=None, alpha_init=None,
            **kwargs):
        """Initialize."""
        super(HeavyTailedKernel, self).__init__(**kwargs)

        self.fit_tau = fit_tau
        if tau_init is None:
            tau_init = tf.random_uniform_initializer(1., 2.)
        self.tau_init = tf.keras.initializers.get(tau_init)
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_init, trainable=self.fit_tau,
            name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_kappa = fit_kappa
        if kappa_init is None:
            kappa_init = tf.random_uniform_initializer(1., 11.)
        self.kappa_init = tf.keras.initializers.get(kappa_init)
        self.kappa = self.add_weight(
            shape=[], initializer=self.kappa_init, trainable=self.fit_kappa,
            name="kappa", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
        )

        self.fit_alpha = fit_alpha
        if alpha_init is None:
            alpha_init = tf.random_uniform_initializer(10., 60.)
        self.alpha_init = tf.keras.initializers.get(alpha_init)
        self.alpha = self.add_weight(
            shape=[], initializer=self.alpha_init, trainable=self.fit_alpha,
            name="alpha", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
        )

        self.theta = {
            'tau': self.tau,
            'kappa': self.kappa,
            'alpha': self.alpha
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A tensor of distances.

        Returns:
            A tensor of similarities.

        """
        return tf.pow(
            self.kappa + tf.pow(inputs, self.tau), (tf.negative(self.alpha))
        )

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_tau': self.fit_tau,
            'fit_kappa': self.fit_kappa,
            'fit_alpha': self.fit_alpha,
            'tau_init': tf.keras.initializers.serialize(self.tau_init),
            'kappa_init': tf.keras.initializers.serialize(self.kappa_init),
            'alpha_init': tf.keras.initializers.serialize(self.alpha_init),
        })
        config = _updated_config(self, config)
        return config


class StudentsTKernel(tf.keras.layers.Layer):
    """Student's t-distribution similarity function.

    The Student's t-distribution similarity function is parameterized
    as:
        s(x,y) = (1 + (((d(x,y)^tau)/alpha))^(-(alpha + 1)/2),
    where x and y are n-dimensional vectors. The original Student-t
    kernel proposed by van der Maaten [1] uses a L2 distane, tau=2, and
    alpha=n_dim-1. By default, all variables are fit to the data.

    Arguments:
        n_dim:  Integer indicating the dimensionality of the embedding.
        fit_tau (optional): Boolean indicating if variable is
            trainable.
        fit_alpha (optional): Boolean indicating if variable is
            trainable.

    References:
    [1] van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic
        triplet embedding. In Machine learning for signal processing
        (MLSP), 2012 IEEE international workshop on (p. 1-6).
        doi:10.1109/MLSP.2012.6349720

    """

    def __init__(
            self, n_dim=None, fit_tau=True, fit_alpha=True, tau_init=None,
            alpha_init=None, **kwargs):
        """Initialize."""
        super(StudentsTKernel, self).__init__(**kwargs)

        self.n_dim = n_dim

        self.fit_tau = fit_tau
        if tau_init is None:
            tau_init = tf.random_uniform_initializer(1., 2.)
        self.tau_init = tf.keras.initializers.get(tau_init)
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_init, trainable=self.fit_tau,
            name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_alpha = fit_alpha
        if alpha_init is None:
            alpha_init = tf.random_uniform_initializer(
                np.max((1, self.n_dim - 2.)), self.n_dim + 2.
            )
        self.alpha_init = tf.keras.initializers.get(alpha_init)
        self.alpha = self.add_weight(
            shape=[], initializer=self.alpha_init, trainable=fit_alpha,
            name="alpha", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=0.000001)
        )

        self.theta = {
            'tau': self.tau,
            'alpha': self.alpha
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A tensor of distances.

        Returns:
            A tensor of similarities.

        """
        return tf.pow(
            1 + (tf.pow(inputs, self.tau) / self.alpha),
            tf.negative(self.alpha + 1)/2
        )

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_tau': self.fit_tau,
            'fit_alpha': self.fit_alpha,
            'tau_init': tf.keras.initializers.serialize(self.tau_init),
            'alpha_init': tf.keras.initializers.serialize(self.alpha_init),
        })
        config = _updated_config(self, config)
        return config


def _updated_config(self, config):
    """Return updated config."""
    return {
        'class_name': self.__class__.__name__,
        'config': config
    }
