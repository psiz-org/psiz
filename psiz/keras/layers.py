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
    GroupAttention: A simple group-specific attention layer.
    InverseSimilarity: A parameterized inverse similarity layer.
    ExponentialSimilarity: A parameterized exponential similarity
        layer.
    HeavyTailedSimilarity: A parameterized heavy-tailed similarity
        layer.
    StudentsTSimilarity: A parameterized Student's t-distribution
        similarity layer.
    Kernel: A kernel that allows the user to separately specify a
        distance and similarity function.
    AttentionKernel: A kernel that uses group-specific attention
        weights and allows the user to separately specify a distance
        and similarity function.
    RankBehavior: A rank behavior layer.
    EmbeddingNormalDiag: A normal distribution embedding layer.
    EmbeddingLogNormalDiag: A log-normal distribution embedding layer.
    EmbeddingLogitNormalDiag: A logit-normal distribution embedding
        layer.
    EmbeddingVariational: A variational embedding layer.
    GroupAttentionVariational: A variation group attention layer.

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow_probability.python.distributions import Normal, LogNormal, LogitNormal
from tensorflow_probability.python.layers import util as tfp_layers_util

import psiz.keras.constraints as pk_constraints
import psiz.keras.initializers as pk_initializers
import psiz.keras.regularizers
import psiz.ops


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='WeightedMinkowski'
)
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

        # Expand rho.
        batch_size = tf.shape(z_q)[0]
        n_compare = tf.shape(z_r)[2]
        n_outcome = tf.shape(z_q)[3]
        rho = self.rho * tf.ones([batch_size, 1, n_compare, n_outcome])

        # Weighted Minkowski distance.
        x = z_q - z_r
        d_qr = psiz.ops.wpnorm(x, w, rho)[:, 0]
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


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='GroupAttention'
)
class GroupAttention(tf.keras.layers.Layer):
    """Group-specific attention weights."""

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
        super(GroupAttention, self).__init__(**kwargs)

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
        group_id = inputs[:, 0]
        return tf.gather(self.w, group_id)

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


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='InverseSimilarity'
)
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


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='ExponentialSimilarity'
)
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


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='HeavyTailedSimilarity'
)
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


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='StudentsTSimilarity'
)
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


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='Kernel'
)
class Kernel(tf.keras.layers.Layer):
    """A basic population-wide kernel."""

    def __init__(self, distance=None, similarity=None, **kwargs):
        """Initialize."""
        super(Kernel, self).__init__(**kwargs)

        if distance is None:
            distance = psiz.keras.layers.WeightedMinkowski()
        self.distance = distance

        if similarity is None:
            similarity = psiz.keras.layers.ExponentialSimilarity()
        self.similarity = similarity

        # Gather all pointers to theta-associated variables.
        theta = self.distance.theta
        theta.update(self.similarity.theta)
        self.theta = theta

    def call(self, inputs):
        """Call.

        Compute k(z_0, z_1), where `k` is the similarity kernel.

        Note: Broadcasting rules are used to compute similarity between
            `z_0` and `z_1`.

        Arguments:
            inputs:
                z_0:
                z_1:
                membership: (unused)

        """
        z_0 = inputs[0]
        z_1 = inputs[1]

        # Create identity attention weights.
        batch_size = tf.shape(z_0)[0]
        n_dim = tf.shape(z_0)[1]
        # NOTE: We must fill in the `dimensionality` dimension in order to
        # keep shapes compatible between op input and calculated input
        # gradient.
        # TODO can we always assume 4D?
        attention = tf.ones([batch_size, n_dim, 1, 1])

        # Compute distance between query and references.
        dist_qr = self.distance([z_0, z_1, attention])
        # Compute similarity.
        sim_qr = self.similarity(dist_qr)
        return sim_qr

    @property
    def n_group(self):
        """Getter method for n_group."""
        return 1

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'distance': tf.keras.utils.serialize_keras_object(self.distance),
            'similarity': tf.keras.utils.serialize_keras_object(
                self.similarity
            ),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create from configuration."""
        config['distance'] = tf.keras.layers.deserialize(config['distance'])
        config['similarity'] = tf.keras.layers.deserialize(
            config['similarity']
        )
        return cls(**config)


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='AttentionKernel'
)
class AttentionKernel(tf.keras.layers.Layer):
    """Attention kernel container."""

    def __init__(
            self, n_dim=None, attention=None, distance=None, similarity=None,
            **kwargs):
        """Initialize.

        Arguments:
            n_dim: The dimensionality of the attention weights. This
                should match the dimensionality of the embedding.
            attention: A attention layer. If this is specified, the
                argument `n_dim` is ignored.
            distance: A distance layer.
            similarity: A similarity layer.

        """
        super(AttentionKernel, self).__init__(**kwargs)

        if attention is None:
            attention = psiz.keras.layers.GroupAttention(
                n_dim=n_dim, n_group=1
            )
        self.attention = attention

        if distance is None:
            distance = psiz.keras.layers.WeightedMinkowski()
        self.distance = distance

        if similarity is None:
            similarity = psiz.keras.layers.ExponentialSimilarity()
        self.similarity = similarity

        # Gather all pointers to theta-associated variables.
        theta = self.distance.theta
        theta.update(self.similarity.theta)
        self.theta = theta

    def call(self, inputs):
        """Call.

        Compute k(z_0, z_1), where `k` is the similarity kernel.

        Note: Broadcasting rules are used to compute similarity between
            `z_0` and `z_1`.

        Arguments:
            inputs:
                z_0:
                z_1:
                membership:

        """
        z_0 = inputs[0]
        z_1 = inputs[1]
        membership = inputs[2]

        # Expand attention weights.
        attention = self.attention(membership)
        # Add singleton dimensions for n_reference and n_outcome axis.
        attention = tf.expand_dims(attention, axis=2)
        attention = tf.expand_dims(attention, axis=3)

        # Compute distance between query and references.
        dist_qr = self.distance([z_0, z_1, attention])
        # Compute similarity.
        sim_qr = self.similarity(dist_qr)
        return sim_qr

    @property
    def n_dim(self):
        """Getter method for n_dim."""
        return self.attention.n_dim

    @property
    def n_group(self):
        """Getter method for n_group."""
        return self.attention.n_group

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'n_dim': self.n_dim,
            'attention': tf.keras.utils.serialize_keras_object(self.attention),
            'distance': tf.keras.utils.serialize_keras_object(self.distance),
            'similarity': tf.keras.utils.serialize_keras_object(
                self.similarity
            ),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create from configuration."""
        config['attention'] = tf.keras.layers.deserialize(config['attention'])
        config['distance'] = tf.keras.layers.deserialize(config['distance'])
        config['similarity'] = tf.keras.layers.deserialize(
            config['similarity']
        )
        return cls(**config)


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='RankBehavior'
)
class RankBehavior(tf.keras.layers.Layer):
    """A rank behavior layer.

    Embodies a `_tf_ranked_sequence_probability` call.

    """

    def __init__(self, **kwargs):
        """Initialize.

        Arguments:
            kwargs (optional): Additional keyword arguments.

        """
        super(RankBehavior, self).__init__(**kwargs)

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

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        return config


class _EmbeddingLocScale(tf.keras.layers.Layer):
    """A private base class for a location-scale embedding.

    Each embedding point is characterized by a location-scale
    distribution.

    """
    def __init__(
            self, input_dim, output_dim, mask_zero=False, input_length=None,
            loc_initializer=None, scale_initializer=None, loc_regularizer=None,
            scale_regularizer=None, loc_constraint=None, scale_constraint=None,
            loc_trainable=True, scale_trainable=True, **kwargs):
        """Initialize.

        Arguments:
            input_dim:
            output_dim:
            mask_zero (optional):
            input_length (optional):
            loc_initializer (optional):
            scale_initializer (optional):
            loc_regularizer (optional):
            scale_regularizer (optional):
            loc_constraint (optional):
            scale_constraint (optional):
            loc_trainable (optional):
            scale_trainable (optional):
            kwargs: Additional key-word arguments.

        Notes:
            The trinability of a particular variable is determined by a
            logical and between `self.trainable` (the
            layer-wise attribute) and `self.x_trainable` (the
            attribute that specifically controls the variable `x`).

        """
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        dtype = kwargs.pop('dtype', K.floatx())
        # We set autocast to False, as we do not want to cast floating-
        # point inputs to self.dtype. In call(), we cast to int32, and
        # casting to self.dtype before casting to int32 might cause the
        # int32 values to be different due to a loss of precision.
        kwargs['autocast'] = False
        super(_EmbeddingLocScale, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length
        self._supports_ragged_inputs = True

        # Handle initializer.
        if loc_initializer is None:
            loc_initializer = tf.keras.initializers.RandomNormal()
        self.loc_initializer = tf.keras.initializers.get(loc_initializer)
        if scale_initializer is None:
            scale_initializer = (
                tf.keras.initializers.RandomNormal(
                    mean=tfp.math.softplus_inverse(1.), stddev=.001
                )
            )
        self.scale_initializer = tf.keras.initializers.get(
            scale_initializer
        )

        # Handle regularizer.
        self.loc_regularizer = tf.keras.regularizers.get(
            loc_regularizer
        )
        self.scale_regularizer = tf.keras.regularizers.get(
            scale_regularizer
        )

        # Handle constraints.
        self.loc_constraint = tf.keras.constraints.get(
            loc_constraint
        )
        self.scale_constraint = tf.keras.constraints.get(
            scale_constraint
        )

        self.loc_trainable = loc_trainable
        self.scale_trainable = scale_trainable

        # If self.dtype is None, build weights using the default dtype.
        dtype = tf.as_dtype(self.dtype or K.floatx())

        # Note: most sparse optimizers do not have GPU kernels defined.
        # When building graphs, the placement algorithm is able to
        # place variables on CPU since it knows all kernels using the
        # variable only exist on CPU. When eager execution is enabled,
        # the placement decision has to be made right now. Checking for
        # the presence of GPUs to avoid complicating the TPU codepaths
        # which can handle sparse optimizers.
        if context.executing_eagerly() and context.context().num_gpus():
            with tf.python.framework.ops.device('cpu:0'):
                self.embeddings = self._build_embeddings_distribution(dtype)
        else:
            self.embeddings = self._build_embeddings_distribution(dtype)

    # @tf_utils.shape_type_conversion
    # def build(self, input_shape):
    #     """Build."""
    #     # If self.dtype is None, build weights using the default dtype.
    #     dtype = tf.as_dtype(self.dtype or K.floatx())

    #     # Note: most sparse optimizers do not have GPU kernels defined.
    #     # When building graphs, the placement algorithm is able to
    #     # place variables on CPU since it knows all kernels using the
    #     # variable only exist on CPU. When eager execution is enabled,
    #     # the placement decision has to be made right now. Checking for
    #     # the presence of GPUs to avoid complicating the TPU codepaths
    #     # which can handle sparse optimizers.
    #     if context.executing_eagerly() and context.context().num_gpus():
    #         with tf.python.framework.ops.device('cpu:0'):
    #             self.embeddings = self._build_embeddings_distribution(dtype)
    #     else:
    #         self.embeddings = self._build_embeddings_distribution(dtype)
    #     self.built = True

    def call(self, inputs):
        """Call."""
        dtype = K.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')

        # Delay reification until end of subclass call in order to
        # generate independent samples for each instance in batch_size.
        inputs_loc = embedding_ops.embedding_lookup(
            self.embeddings.distribution.loc, inputs
        )
        inputs_scale = embedding_ops.embedding_lookup(
            self.embeddings.distribution.scale, inputs
        )
        return [inputs_loc, inputs_scale]

    def get_config(self):
        """Return layer configuration."""
        config = super(_EmbeddingLocScale, self).get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'mask_zero': self.mask_zero,
            'input_length': self.input_length,
            'loc_initializer':
                tf.keras.initializers.serialize(self.loc_initializer),
            'scale_initializer':
                tf.keras.initializers.serialize(self.scale_initializer),
            'loc_regularizer':
                tf.keras.regularizers.serialize(self.loc_regularizer),
            'scale_regularizer':
                tf.keras.regularizers.serialize(self.scale_regularizer),
            'loc_constraint':
                tf.keras.constraints.serialize(self.loc_constraint),
            'scale_constraint':
                tf.keras.constraints.serialize(self.scale_constraint),
            'loc_trainable': self.loc_trainable,
            'scale_trainable': self.scale_trainable,
        })
        return config

    @property
    def embeddings_mode(self):
        """Getter method for mode of `embeddings`."""
        return self.embeddings.mode()


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingNormalDiag'
)
class EmbeddingNormalDiag(_EmbeddingLocScale):
    """A distribution-based embedding.

    Each embedding point is characterized by a Normal distribution with
    a diagonal scale matrix.

    """
    def __init__(self, input_dim, output_dim, **kwargs):
        """Initialize."""
        super(EmbeddingNormalDiag, self).__init__(
            input_dim, output_dim, **kwargs
        )

    def _build_embeddings_distribution(self, dtype):
        """Build embeddings distribution."""
        # Handle location variables.
        loc_trainable = self.trainable and self.loc_trainable
        loc = self.add_weight(
            name='loc', shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.loc_initializer, regularizer=self.loc_regularizer,
            trainable=loc_trainable, constraint=self.loc_constraint
        )

        # Handle scale variables.
        scale_trainable = self.trainable and self.scale_trainable
        untransformed_scale = self.add_weight(
            name='untransformed_scale',
            shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.scale_initializer,
            regularizer=self.scale_regularizer, trainable=scale_trainable,
            constraint=self.scale_constraint
        )
        scale = tfp.util.DeferredTensor(
            untransformed_scale,
            lambda x: (K.epsilon() + tf.nn.softplus(x))
        )

        dist = tfp.distributions.Normal(loc=loc, scale=scale)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call."""
        [inputs_loc, inputs_scale] = super().call(inputs)
        # Use reparameterization trick.
        dist_batch = Normal(loc=inputs_loc, scale=inputs_scale)
        # Reify output using samples.
        return dist_batch.sample()


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingLogNormalDiag'
)
class EmbeddingLogNormalDiag(_EmbeddingLocScale):
    """A distribution-based embedding.

    Each embedding point is characterized by a Log-Normal distribution
    with a diagonal scale matrix.

    """
    def __init__(self, input_dim, output_dim, **kwargs):
        """Initialize."""
        super(EmbeddingLogNormalDiag, self).__init__(
            input_dim, output_dim, **kwargs
        )

    def _build_embeddings_distribution(self, dtype):
        """Build embeddings distribution."""
        # Handle location variables.
        loc_trainable = self.trainable and self.loc_trainable
        loc = self.add_weight(
            name='loc', shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.loc_initializer, regularizer=self.loc_regularizer,
            trainable=loc_trainable, constraint=self.loc_constraint
        )

        # Handle scale variables.
        scale_trainable = self.trainable and self.scale_trainable
        untransformed_scale = self.add_weight(
            name='untransformed_scale',
            shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.scale_initializer,
            regularizer=self.scale_regularizer, trainable=scale_trainable,
            constraint=self.scale_constraint
        )
        scale = tfp.util.DeferredTensor(
            untransformed_scale,
            lambda x: (K.epsilon() + tf.nn.softplus(x))
        )

        dist = tfp.distributions.LogNormal(loc=loc, scale=scale)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call."""
        [inputs_loc, inputs_scale] = super().call(inputs)
        # Use reparameterization trick.
        dist_batch = LogNormal(loc=inputs_loc, scale=inputs_scale)
        # Reify output using samples.
        return dist_batch.sample()


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingLogitNormalDiag'
)
class EmbeddingLogitNormalDiag(_EmbeddingLocScale):
    """A distribution-based embedding.

    Each embedding point is characterized by LogitNormal distribution
    with a diagonal scale matrix.

    """
    def __init__(self, input_dim, output_dim, **kwargs):
        """Initialize."""
        # Overide default scale initializer.
        scale_initializer = kwargs.pop('scale_initializer', None)
        if scale_initializer is None:
            scale_initializer = tf.keras.initializers.RandomNormal(0.3, .01)

        super(EmbeddingLogitNormalDiag, self).__init__(
            input_dim, output_dim, scale_initializer=scale_initializer,
            **kwargs
        )

    def _build_embeddings_distribution(self, dtype):
        """Build embeddings distribution."""
        # Handle location variables.
        loc_trainable = self.trainable and self.loc_trainable
        untransformed_loc = self.add_weight(
            name='untransformed_loc', shape=[self.input_dim, self.output_dim],
            dtype=dtype, initializer=self.loc_initializer,
            regularizer=self.loc_regularizer, trainable=loc_trainable,
            constraint=self.loc_constraint
        )

        # Handle scale variables.
        scale_trainable = self.trainable and self.scale_trainable
        untransformed_scale = self.add_weight(
            name='untransformed_scale',
            shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.scale_initializer,
            regularizer=self.scale_regularizer, trainable=scale_trainable,
            constraint=self.scale_constraint
        )
        scale = tfp.util.DeferredTensor(
            untransformed_scale, lambda x: (K.epsilon() + tf.math.exp(x))
        )

        dist = tfp.distributions.LogitNormal(
            loc=untransformed_loc, scale=scale
        )
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call."""
        [inputs_loc, inputs_scale] = super().call(inputs)
        # Use reparameterization trick.
        dist_batch = LogitNormal(loc=inputs_loc, scale=inputs_scale)
        # Reify output using samples.
        return dist_batch.sample()

    @property
    def embeddings_mode(self):
        """Getter method for mode of `embeddings`."""
        # Use median as approximation of mode. For logit-normal distribution,
        # `median = logistic(loc)`.
        return tf.math.sigmoid(self.embeddings.distribution.loc)


class _Variational(tf.keras.layers.Layer):
    """Private base class for variational layer.

    NOTE: Assumes that the KL divergence between the posterior and
    prior is registered.

    """

    def __init__(self, posterior=None, prior=None, kl_weight=1., **kwargs):
        """Initialize.

        Arguments:
            posterior: A layer embodying the posterior.
            prior: A layer embodying the prior.
            kl_weight (optional): A scalar applied to the KL
                divergence computation. This value should be 1 divided
                by the total number of training examples.
            kwargs: Additional key-word arguments.

        """
        super(_Variational, self).__init__(**kwargs)
        self.posterior = posterior
        self.prior = prior
        self.kl_weight = kl_weight

    def call(self, inputs):
        """Call."""
        # Run forward pass through variational posterior layer.
        outputs = self.posterior(inputs)

        # Apply KL divergence between posterior and prior.
        self.apply_kl(self.posterior, self.prior)

        return outputs
     
    def _add_kl_loss(self, posterior_dist, prior_dist):
        """Add KL divergence loss.""" 
        self.add_loss(
            lambda: kl_lib.kl_divergence(
                posterior_dist, prior_dist
            ) * self.kl_weight
        )

    def get_config(self):
        """Return configuration."""
        config = super(_Variational, self).get_config()
        config.update({
            'posterior': tf.keras.utils.serialize_keras_object(self.posterior),
            'prior': tf.keras.utils.serialize_keras_object(self.prior),
            'kl_weight': self.kl_weight
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration.

        This method is the reverse of `get_config`, capable of
        instantiating the same layer from the config dictionary.

        Args:
            config: A Python dictionary, typically the output of
                `get_config`.

        Returns:
            layer: A layer instance.

        """
        config['posterior'] = tf.keras.layers.deserialize(config['posterior'])
        config['prior'] = tf.keras.layers.deserialize(config['prior'])
        return cls(**config)


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingVariational'
)
class EmbeddingVariational(_Variational):
    """Variational analog of Embedding layer."""

    def __init__(self, **kwargs):
        """Initialize.

        Arguments:
            kwargs: Additional key-word arguments.

        """
        super(EmbeddingVariational, self).__init__(**kwargs)
    
    def apply_kl(self, posterior, prior):
        """Apply KL divergence."""
        self._add_kl_loss(posterior.embeddings, prior.embeddings)

    @property
    def output_dim(self):
        """Getter method for embeddings posterior mode."""
        return self.posterior.embeddings.distribution.loc.shape[1]

    @property
    def embeddings(self):
        """Getter method for embeddings posterior mode."""
        return self.posterior.embeddings_mode


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='GroupAttentionVariational'
)
class GroupAttentionVariational(_Variational):
    """Variational analog of group-specific attention weights."""

    def __init__(self, **kwargs):
        """Initialize.

        Arguments:
            kwargs: Additional key-word arguments.

        """
        super(GroupAttentionVariational, self).__init__(**kwargs)

    def call(self, inputs):
        """Call.

        Inflate weights by `group_id`.

        Arguments:
            inputs: A Tensor denoting `group_id`.

        """
        group_id = inputs[:, 0]
        outputs = super().call(group_id)
        return outputs

    def apply_kl(self, posterior, prior):
        """Apply KL divergence."""
        self._add_kl_loss(posterior.embeddings, prior.embeddings)

    @property
    def n_group(self):
        """Getter method for `n_group`"""
        return self.posterior.embeddings.distribution.loc.shape[0]

    @property
    def n_dim(self):
        """Getter method for `n_group`"""
        return self.posterior.embeddings.distribution.loc.shape[1]

    @property
    def w(self):
        """Getter method for embeddings posterior mode."""
        return self.posterior.embeddings_mode
