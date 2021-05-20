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
    GroupAttentionVariational: A variational group attention layer.

"""

import tensorflow as tf

from psiz.keras.layers.variational import Variational


# DEPRECATED
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

        Grab `groups` only.

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
