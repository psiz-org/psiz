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
"""Module of TensorFlow distance layers.

Classes:
    MinkowskiVariational: A variational Minkowski layer.

"""

import tensorflow as tf

from psiz.keras.layers.variational import Variational


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='MinkowskiVariational'
)
class MinkowskiVariational(Variational):
    """Variational analog of Embedding layer."""

    def __init__(self, **kwargs):
        """Initialize.

        Arguments:
            kwargs: Additional key-word arguments.

        """
        super(MinkowskiVariational, self).__init__(**kwargs)

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A list of two tf.Tensor's denoting a the set of
                vectors to compute pairwise distance. Each tensor is
                assumed to have the same shape and be at least rank-2.
                Any additional tensors in the list are ignored.
                shape = (batch_size, [n, m, ...] n_dim)

        Returns:
            shape=(batch_size, [n, m, ...])

        """
        # Run forward pass through variational posterior layer.
        outputs = self.posterior(inputs)

        # Call prior in case it is a variational layer as well.
        _ = self.prior(inputs)

        # Apply KL divergence between posterior and prior.
        self.add_kl_loss(self.posterior.w, self.prior.w)
        self.add_kl_loss(self.posterior.rho, self.prior.rho)
        return outputs

    @property
    def w(self):
        """Getter method for (posterior) w."""
        return self.posterior.w

    @property
    def rho(self):
        """Getter method for (posterior) rho."""
        return self.posterior.rho
