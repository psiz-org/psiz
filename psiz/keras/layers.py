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
    WeightedMinkowski: A weighted distance layer.
    Attention: A simple attention layer.
    InverseSimilarity: A parameterized inverse similarity layer.
    ExponentialSimilarity: A parameterized exponential similarity
        layer.
    HeavyTailedSimilarity: A parameterized heavy-tailed similarity
        layer.
    StudentsTSimilarity: A parameterized Student's t-distribution
        similarity layer.
    Rank: A rank behavior layer.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import psiz.keras.constraints as pk_constraints
import psiz.keras.initializers as pk_initializers
import psiz.keras.regularizers
import psiz.ops


class WeightedMinkowski(tf.keras.layers.Layer):
    """Weighted Minkowski distance."""

    def __init__(self, fit_rho=True, rho_initializer=None, **kwargs):
        """Initialize.

        Arguments:
            fit_rho (optional): Boolean indicating if variable is
                trainable.
            rho_initializer (optional): Initializer for rho.

        """
        super(WeightedMinkowski, self).__init__(**kwargs)
        self.fit_rho = fit_rho

        if rho_initializer is None:
            rho_initializer = tf.random_uniform_initializer(1.01, 3.)
        self.rho_initializer = tf.keras.initializers.get(rho_initializer)
        self.rho = self.add_weight(
            shape=[], initializer=self.rho_initializer,
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
        x = z_q - z_r
        d_qr = psiz.ops.wpnorm(x, w, self.rho)[:, 0]
        return d_qr

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_rho': self.fit_rho,
            'rho_initializer': tf.keras.initializers.serialize(
                self.rho_initializer
            )
        })
        return config


class Attention(tf.keras.layers.Layer):
    """Attention Layer."""

    def __init__(
            self, n_group=1, n_dim=None, fit_group=None,
            embeddings_initializer=None, embeddings_regularizer=None,
            embeddings_constraint=None, **kwargs):
        """Initialize.

        Arguments:
            n_dim: An integer indicating the dimensionality of the
                embeddings. Must be equal to or greater than one.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group. Must be equal to or greater than one.
            fit_group: Boolean indicating if variable is trainable.
                shape=(n_group,)

        Raises:
            ValueError: If `n_dim` or `n_group` arguments are invalid.
        
        """
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
        return config


class InverseSimilarity(tf.keras.layers.Layer):
    """Inverse-distance similarity function.

    The inverse-distance similarity function is parameterized as:
        s(x,y) = 1 / (d(x,y)**tau + mu),
    where x and y are n-dimensional vectors.

    """

    def __init__(
            self, fit_tau=True, fit_mu=True, tau_initializer=None,
            mu_initializer=None, **kwargs):
        """Initialize.

        Arguments:
            fit_tau (optional): Boolean indicating if variable is
                trainable.
            fit_gamma (optional): Boolean indicating if variable is
                trainable.
            fit_beta (optional): Boolean indicating if variable is
                trainable.

        """
        super(InverseSimilarity, self).__init__(**kwargs)

        self.fit_tau = fit_tau
        if tau_initializer is None:
            tau_initializer = tf.random_uniform_initializer(1., 2.)
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_initializer, trainable=self.fit_tau,
            name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_mu = fit_mu
        if mu_initializer is None:
            mu_initializer = tf.random_uniform_initializer(0.0000000001, .001)
        self.mu_initializer = tf.keras.initializers.get(tau_initializer)
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
            'tau_initializer': tf.keras.initializers.serialize(
                self.tau_initializer
            ),
            'mu_initializer': tf.keras.initializers.serialize(
                self.mu_initializer
            ),
        })
        return config


class ExponentialSimilarity(tf.keras.layers.Layer):
    """Exponential family similarity function.

    This exponential-family similarity function is parameterized as:
        s(x,y) = exp(-beta .* d(x,y).^tau) + gamma,
    where x and y are n-dimensional vectors. The exponential family
    function is obtained by integrating across various psychological
    theories [1,2,3,4].

    By default beta=10. and is not trainable to prevent redundancy with
    trainable embeddings and to prevent short-circuiting any
    regularizers placed on the embeddings.

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
            self, fit_tau=True, fit_gamma=True, fit_beta=False,
            tau_initializer=None, gamma_initializer=None,
            beta_initializer=None, **kwargs):
        """Initialize.

        Arguments:
            fit_tau (optional): Boolean indicating if variable is
                trainable.
            fit_gamma (optional): Boolean indicating if variable is
                trainable.
            fit_beta (optional): Boolean indicating if variable is
                trainable.

        """
        super(ExponentialSimilarity, self).__init__(**kwargs)

        self.fit_tau = fit_tau
        if tau_initializer is None:
            tau_initializer = tf.random_uniform_initializer(1., 2.)
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_initializer, trainable=self.fit_tau,
            name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_gamma = fit_gamma
        if gamma_initializer is None:
            gamma_initializer = tf.random_uniform_initializer(0., .001)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.gamma = self.add_weight(
            shape=[], initializer=self.gamma_initializer,
            trainable=self.fit_gamma, name="gamma", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
        )

        self.fit_beta = fit_beta
        if beta_initializer is None:
            if fit_beta:
                beta_initializer = tf.random_uniform_initializer(1., 30.)
            else:
                beta_initializer = tf.keras.initializers.Constant(value=10.)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.beta = self.add_weight(
            shape=[], initializer=self.beta_initializer,
            trainable=self.fit_beta, name="beta", dtype=K.floatx(),
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
            'tau_initializer': tf.keras.initializers.serialize(
                self.tau_initializer
            ),
            'gamma_initializer': tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            'beta_initializer': tf.keras.initializers.serialize(
                self.beta_initializer
            ),
        })
        return config


class HeavyTailedSimilarity(tf.keras.layers.Layer):
    """Heavy-tailed family similarity function.

    The heavy-tailed similarity function is parameterized as:
        s(x,y) = (kappa + (d(x,y).^tau)).^(-alpha),
    where x and y are n-dimensional vectors. The heavy-tailed family is
    a generalization of the Student-t family.

    """

    def __init__(
            self, fit_tau=True, fit_kappa=True, fit_alpha=True,
            tau_initializer=None, kappa_initializer=None,
            alpha_initializer=None, **kwargs):
        """Initialize.

        Arguments:
            fit_tau (optional): Boolean indicating if variable is
                trainable.
            fit_kappa (optional): Boolean indicating if variable is
                trainable.
            fit_alpha (optional): Boolean indicating if variable is
                trainable.

        """
        super(HeavyTailedSimilarity, self).__init__(**kwargs)

        self.fit_tau = fit_tau
        if tau_initializer is None:
            tau_initializer = tf.random_uniform_initializer(1., 2.)
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_initializer, trainable=self.fit_tau,
            name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_kappa = fit_kappa
        if kappa_initializer is None:
            kappa_initializer = tf.random_uniform_initializer(1., 11.)
        self.kappa_initializer = tf.keras.initializers.get(kappa_initializer)
        self.kappa = self.add_weight(
            shape=[], initializer=self.kappa_initializer,
            trainable=self.fit_kappa, name="kappa", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
        )

        self.fit_alpha = fit_alpha
        if alpha_initializer is None:
            alpha_initializer = tf.random_uniform_initializer(10., 60.)
        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
        self.alpha = self.add_weight(
            shape=[], initializer=self.alpha_initializer,
            trainable=self.fit_alpha, name="alpha", dtype=K.floatx(),
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
            'tau_initializer': tf.keras.initializers.serialize(
                self.tau_initializer
            ),
            'kappa_initializer': tf.keras.initializers.serialize(
                self.kappa_initializer
            ),
            'alpha_initializer': tf.keras.initializers.serialize(
                self.alpha_initializer
            ),
        })
        return config


class StudentsTSimilarity(tf.keras.layers.Layer):
    """Student's t-distribution similarity function.

    The Student's t-distribution similarity function is parameterized
    as:
        s(x,y) = (1 + (((d(x,y)^tau)/alpha))^(-(alpha + 1)/2),
    where x and y are n-dimensional vectors. The original Student-t
    kernel proposed by van der Maaten [1] uses a L2 distane, tau=2, and
    alpha=n_dim-1. By default, all variables are fit to the data.

    References:
    [1] van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic
        triplet embedding. In Machine learning for signal processing
        (MLSP), 2012 IEEE international workshop on (p. 1-6).
        doi:10.1109/MLSP.2012.6349720

    """

    def __init__(
            self, n_dim=None, fit_tau=True, fit_alpha=True,
            tau_initializer=None, alpha_initializer=None, **kwargs):
        """Initialize.

        Arguments:
            n_dim:  Integer indicating the dimensionality of the
                embedding.
            fit_tau (optional): Boolean indicating if variable is
                trainable.
            fit_alpha (optional): Boolean indicating if variable is
                trainable.

        """
        super(StudentsTSimilarity, self).__init__(**kwargs)

        self.n_dim = n_dim

        self.fit_tau = fit_tau
        if tau_initializer is None:
            tau_initializer = tf.random_uniform_initializer(1., 2.)
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_initializer, trainable=self.fit_tau,
            name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_alpha = fit_alpha
        if alpha_initializer is None:
            alpha_initializer = tf.random_uniform_initializer(
                np.max((1, self.n_dim - 2.)), self.n_dim + 2.
            )
        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
        self.alpha = self.add_weight(
            shape=[], initializer=self.alpha_initializer, trainable=fit_alpha,
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
            'tau_initializer': tf.keras.initializers.serialize(
                self.tau_initializer
            ),
            'alpha_initializer': tf.keras.initializers.serialize(
                self.alpha_initializer
            ),
        })
        return config


class Rank(tf.keras.layers.Layer):
    """A rank behavior layer."""

    def __init__(self, **kwargs):
        """Initialize.

        Arguments:
            kwargs (optional): Additional keyword arguments.

        """
        super(Rank, self).__init__(**kwargs)

    def call(self, inputs):
        """Return probability of a ranked selection sequence.

        See: _ranked_sequence_probability for NumPy implementation.

        Arguments:
            inputs:
                sim_qr: A tensor containing the precomputed
                    similarities between the query stimuli and
                    corresponding reference stimuli.
                    shape = (batch_size, n_max_reference, n_outcome)
                is_select: A Boolean tensor indicating if a reference
                    was selected.
                    shape = (batch_size, n_max_reference, n_outcome)

        """
        sim_qr = inputs[0]
        is_select = inputs[1]
        is_outcome = inputs[2]

        # Initialize sequence log-probability. Note that log(prob=1)=1.
        batch_size = tf.shape(sim_qr)[0]
        n_outcome = tf.shape(sim_qr)[2]
        seq_log_prob = tf.zeros([batch_size, n_outcome], dtype=K.floatx())

        # Compute denominator based on formulation of Luce's choice rule.
        denom = tf.cumsum(sim_qr, axis=1, reverse=True)

        # Compute log-probability of each selection, assuming all selections
        # occurred. Add fuzz factor to avoid log(0)
        sim_qr = tf.maximum(sim_qr, tf.keras.backend.epsilon())
        denom = tf.maximum(denom, tf.keras.backend.epsilon())
        log_prob = tf.math.log(sim_qr) - tf.math.log(denom)

        # Mask non-existent selections.
        log_prob = is_select * log_prob

        # Compute sequence log-probability
        seq_log_prob = tf.reduce_sum(log_prob, axis=1)
        seq_prob = tf.math.exp(seq_log_prob)
        return is_outcome * seq_prob
