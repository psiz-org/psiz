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
"""Module of TensorFlow kernel layers.

Classes:
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
    GroupAttentionVariational: A variation group attention layer.

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints
import psiz.keras.initializers as pk_initializers
from psiz.keras.layers.variational import Variational
from psiz.keras.layers.distances.minkowski import WeightedMinkowski
from psiz.models.base import GroupLevel


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
                fit_group = False  # TODO default should always be train
            else:
                fit_group = True
        self.fit_group = fit_group

        self.embeddings = self.add_weight(
            shape=(self.n_group, self.n_dim),
            initializer=self.embeddings_initializer,
            trainable=fit_group, name='w', dtype=K.floatx(),
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint
        )
        self.mask_zero = False

    def call(self, inputs):
        """Call.

        Inflate weights by `group_id`.

        Arguments:
            inputs: A Tensor denoting `group_id`.

        """
        output = tf.gather(self.embeddings, inputs)
        # Add singleton dimension for sample_size.
        output = tf.expand_dims(output, axis=0)
        return output

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'n_group': int(self.n_group),
            'n_dim': int(self.n_dim),
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
        tau_trainable = self.trainable and self.fit_tau
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_initializer,
            trainable=tau_trainable, name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_mu = fit_mu
        if mu_initializer is None:
            mu_initializer = tf.random_uniform_initializer(0.0000000001, .001)
        self.mu_initializer = tf.keras.initializers.get(tau_initializer)
        mu_trainable = self.trainable and self.fit_mu
        self.mu = self.add_weight(
            shape=[], initializer=self.tau_int, trainable=mu_trainable,
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
        tau_trainable = self.trainable and self.fit_tau
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_initializer,
            trainable=tau_trainable, name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_gamma = fit_gamma
        if gamma_initializer is None:
            gamma_initializer = tf.random_uniform_initializer(0., .001)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        gamma_trainable = self.trainable and self.fit_gamma
        self.gamma = self.add_weight(
            shape=[], initializer=self.gamma_initializer,
            trainable=gamma_trainable, name="gamma", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
        )

        self.fit_beta = fit_beta
        if beta_initializer is None:
            if fit_beta:
                beta_initializer = tf.random_uniform_initializer(1., 30.)
            else:
                beta_initializer = tf.keras.initializers.Constant(value=10.)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        beta_trainable = self.trainable and self.fit_beta
        self.beta = self.add_weight(
            shape=[], initializer=self.beta_initializer,
            trainable=beta_trainable, name="beta", dtype=K.floatx(),
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
        tau_trainable = self.trainable and self.fit_tau
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_initializer,
            trainable=tau_trainable, name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_kappa = fit_kappa
        if kappa_initializer is None:
            kappa_initializer = tf.random_uniform_initializer(1., 11.)
        self.kappa_initializer = tf.keras.initializers.get(kappa_initializer)
        kappa_trainable = self.trainable and self.fit_kappa
        self.kappa = self.add_weight(
            shape=[], initializer=self.kappa_initializer,
            trainable=kappa_trainable, name="kappa", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
        )

        self.fit_alpha = fit_alpha
        if alpha_initializer is None:
            alpha_initializer = tf.random_uniform_initializer(10., 60.)
        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
        alpha_trainable = self.trainable and self.fit_alpha
        self.alpha = self.add_weight(
            shape=[], initializer=self.alpha_initializer,
            trainable=alpha_trainable, name="alpha", dtype=K.floatx(),
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
        tau_trainable = self.trainable and self.fit_tau
        self.tau = self.add_weight(
            shape=[], initializer=self.tau_initializer,
            trainable=tau_trainable, name="tau", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
        )

        self.fit_alpha = fit_alpha
        if alpha_initializer is None:
            alpha_initializer = tf.random_uniform_initializer(
                np.max((1, self.n_dim - 2.)), self.n_dim + 2.
            )
        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
        alpha_trainable = self.trainable and self.fit_alpha
        self.alpha = self.add_weight(
            shape=[], initializer=self.alpha_initializer,
            trainable=alpha_trainable, name="alpha", dtype=K.floatx(),
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
class Kernel(GroupLevel):
    """A basic population-wide kernel."""

    def __init__(self, distance=None, similarity=None, **kwargs):
        """Initialize."""
        super(Kernel, self).__init__(**kwargs)

        if distance is None:
            distance = WeightedMinkowski()
        self.distance = distance

        if similarity is None:
            similarity = ExponentialSimilarity()
        self.similarity = similarity

        # Gather all pointers to theta-associated variables.
        theta = self.distance.theta
        theta.update(self.similarity.theta)
        self.theta = theta

        self._n_sample = ()
        self._kl_weight = 0

    @property
    def n_sample(self):
        return self._n_sample

    @n_sample.setter
    def n_sample(self, n_sample):
        self._n_sample = n_sample
        self.distance.n_sample = n_sample
        self.similarity.n_sample = n_sample

    @property
    def kl_weight(self):
        return self._kl_weight

    @kl_weight.setter
    def kl_weight(self, kl_weight):
        self._kl_weight = kl_weight
        # Set kl_weight of constituent layers. # TODO MAYBE use `_layers`?
        self.distance.kl_weight = kl_weight
        self.similarity.kl_weight = kl_weight

    def call(self, inputs):
        """Call.

        Compute k(z_0, z_1), where `k` is the similarity kernel.

        Note: Broadcasting rules are used to compute similarity between
            `z_0` and `z_1`.

        Arguments:
            inputs:
                z_0: A tf.Tensor denoting a set of vectors.
                    shape = (batch_size, [n, m, ...] n_dim)
                z_1: A tf.Tensor denoting a set of vectors.
                    shape = (batch_size, [n, m, ...] n_dim)

        """
        z_0 = inputs[0]
        z_1 = inputs[1]
        # group = inputs[-1][:, self.group_level]

        # Create identity attention weights.
        attention = tf.ones_like(z_0)

        # Compute distance between query and references.
        dist_qr = self.distance([z_0, z_1, attention])
        # Compute similarity.
        sim_qr = self.similarity(dist_qr)
        return sim_qr

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
class AttentionKernel(GroupLevel):
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
            attention = GroupAttention(n_dim=n_dim, n_group=1)
        self.attention = attention

        if distance is None:
            distance = WeightedMinkowski()
        self.distance = distance

        if similarity is None:
            similarity = ExponentialSimilarity()
        self.similarity = similarity

        # Gather all pointers to theta-associated variables.
        theta = self.distance.theta
        theta.update(self.similarity.theta)
        self.theta = theta

        self._n_sample = ()
        self._kl_weight = 0

    def call(self, inputs):
        """Call.

        Compute k(z_0, z_1), where `k` is the similarity kernel.

        Note: Broadcasting rules are used to compute similarity between
            `z_0` and `z_1`.

        Arguments:
            inputs:
                z_0: A tf.Tensor denoting a set of vectors.
                    shape = (batch_size, [n, m, ...] n_dim)
                z_1: A tf.Tensor denoting a set of vectors.
                    shape = (batch_size, [n, m, ...] n_dim)
                group: A tf.Tensor denoting group assignments.
                    shape = (batch_size, k)

        """
        z_0 = inputs[0]
        z_1 = inputs[1]
        group = inputs[-1]

        # Expand attention weights.
        attention = self.attention(group[:, self.group_level])

        # Add singleton inner dimensions that are not related to sample_size,
        # batch_size or vector dimensionality.
        attention_shape = tf.shape(attention)
        sample_size = tf.expand_dims(attention_shape[0], axis=0)
        batch_size = tf.expand_dims(attention_shape[1], axis=0)
        dim_size = tf.expand_dims(attention_shape[-1], axis=0)

        n_expand = tf.rank(z_0) - tf.rank(attention)
        shape_exp = tf.ones(n_expand, dtype=attention_shape[0].dtype)
        shape_exp = tf.concat(
            (sample_size, batch_size, shape_exp, dim_size), axis=0
        )
        attention = tf.reshape(attention, shape_exp)

        # Compute distance between query and references.
        dist_qr = self.distance([z_0, z_1, attention])
        # Compute similarity.
        sim_qr = self.similarity(dist_qr)
        return sim_qr

    # @property
    # def n_dim(self):
    #     """Getter method for n_dim."""
    #     return self.attention.n_dim

    @property
    def n_sample(self):
        return self._n_sample

    @n_sample.setter
    def n_sample(self, n_sample):
        self._n_sample = n_sample
        self.attention.n_sample = n_sample
        self.distance.n_sample = n_sample
        self.similarity.n_sample = n_sample

    @property
    def kl_weight(self):
        return self._kl_weight

    @kl_weight.setter
    def kl_weight(self, kl_weight):
        self._kl_weight = kl_weight
        # Set kl_weight of constituent layers. # TODO MAYBE use `_layers`?
        self.attention.kl_weight = kl_weight
        self.distance.kl_weight = kl_weight
        self.similarity.kl_weight = kl_weight

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            # 'n_dim': int(self.n_dim),
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
    package='psiz.keras.layers', name='GroupAttentionVariational'
)
class GroupAttentionVariational(Variational):
    """Variational analog of group-specific attention weights."""

    def __init__(self, **kwargs):
        """Initialize.

        Arguments:
            kwargs: Additional key-word arguments.

        """
        super(GroupAttentionVariational, self).__init__(**kwargs)

    def call(self, inputs):
        """Call.

        Grab `group_id` only.

        Arguments:
            inputs: A Tensor denoting a trial's group membership.

        """
        # Run forward pass through variational posterior layer.
        outputs = self.posterior(inputs)

        # Apply KL divergence between posterior and prior.
        self.add_kl_loss(self.posterior.embeddings, self.prior.embeddings)

        return outputs

    @property
    def n_group(self):
        """Getter method for `n_group`"""
        # TODO need better decoupling, not all distributions will have loc.
        return self.posterior.embeddings.distribution.loc.shape[0]

    @property
    def n_dim(self):
        """Getter method for `n_group`"""
        # TODO need better decoupling, not all distributions will have loc.
        return self.posterior.embeddings.distribution.loc.shape[1]

    @property
    def mask_zero(self):
        """Getter method for embeddings `mask_zero`."""
        return self.posterior.mask_zero

    @property
    def embeddings(self):
        """Getter method for embeddings posterior mode."""
        return self.posterior.embeddings
