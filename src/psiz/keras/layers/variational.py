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
"""Module of abstract TensorFlow variational layer.

Classes:
    Variational: An abstract base class for variational inference
        layers.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib


class Variational(tf.keras.layers.Layer):
    """An abstract base class for variational layers.

    This class can take advantage of a registered KL divergence
    between the posterior and prior is registered.

    Attributes:
        kl_weight: The weighting of the kl term. Should be 1/n_train.
        kl_use_exact: Boolean indicating if a registered KL divergence
            should be used.
        kl_anneal: An annealing weight that can be accessed using a
            callback. Iniitalized to one so it has no effect if not
            used in a callback.

    Notes:
        This layer is not registered as serializable because it is
        intended to be subclassed. Subclasses must implement `call`,
        which should sample from the posterior and call
        `add_kl_loss`.

    """

    def __init__(
            self, posterior=None, prior=None, kl_weight=0., kl_use_exact=False,
            kl_n_sample=100, **kwargs):
        """Initialize.

        Arguments:
            posterior: A layer embodying the posterior.
            prior: A layer embodying the prior.
            kl_weight (optional): A scalar applied to the KL
                divergence computation. This value is determined
                theoretically. See NOTES in the `fit_step` method of
                `PscyhologicalEmbedding`. By default, no KL loss is
                added.
            kl_use_exact (optional): Boolean indicating if analytical
                KL divergence should be used rather than a Monte Carlo
                approximation.
            kl_n_sample (optional): The number of samples to use if
                approximating KL.
            kwargs: Additional key-word arguments.

        """
        super().__init__(**kwargs)
        self.posterior = posterior
        self.prior = prior
        self.kl_weight = kl_weight
        self.kl_use_exact = kl_use_exact
        self.kl_n_sample = kl_n_sample
        self.kl_anneal = self.add_weight(
            name='kl_anneal', shape=[], dtype=K.floatx(),
            initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        )

    def build(self, input_shape):
        """Build."""
        self.prior.build(input_shape)
        self.posterior.build(input_shape)
        super().build(input_shape)

    def add_kl_loss(self, posterior_dist, prior_dist):
        """Add KL divergence loss."""
        if self.kl_use_exact:
            self.add_loss(
                kl_lib.kl_divergence(
                    posterior_dist, prior_dist
                ) * self.kl_weight * self.kl_anneal
            )
        else:
            self.add_loss(
                self._kl_approximation(
                    posterior_dist, prior_dist
                ) * self.kl_weight * self.kl_anneal
            )

    def _kl_approximation(self, posterior_dist, prior_dist):
        """Sample-based KL approximation."""
        posterior_samples = posterior_dist.sample(self.kl_n_sample)
        return tf.reduce_mean(
            posterior_dist.log_prob(posterior_samples)
            - prior_dist.log_prob(posterior_samples)
        )

    def get_config(self):
        """Return configuration."""
        config = super().get_config()
        config.update({
            'posterior': tf.keras.utils.serialize_keras_object(self.posterior),
            'prior': tf.keras.utils.serialize_keras_object(self.prior),
            'kl_weight': float(self.kl_weight),
            'kl_use_exact': self.kl_use_exact,
            'kl_n_sample': int(self.kl_n_sample),
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
