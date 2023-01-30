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
"""Module of TensorFlow embedding layers.

Classes:
    EmbeddingVariational: A variational embedding layer.

"""

import tensorflow as tf

from psiz.keras.layers.variational import Variational


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="EmbeddingVariational"
)
class EmbeddingVariational(Variational):
    """Variational analog of Embedding layer."""

    def __init__(self, **kwargs):
        """Initialize.

        Args:
            kwargs: Additional key-word arguments.

        """
        super(EmbeddingVariational, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        """Call."""
        # Run forward pass through variational posterior layer.
        outputs = self.posterior(inputs)

        # Also call prior in case prior is a variational layer itself that
        # needs to have it's VI losses registered.
        _ = self.prior(inputs)

        # Apply KL divergence between posterior and prior.
        self.add_kl_loss(self.posterior.embeddings, self.prior.embeddings)
        return outputs

    @property
    def input_dim(self):
        """Getter method for embeddings `input_dim`."""
        return self.posterior.input_dim

    @property
    def output_dim(self):
        """Getter method for embeddings `output_dim`."""
        return self.posterior.output_dim

    @property
    def mask_zero(self):
        """Getter method for embeddings `mask_zero`."""
        return self.posterior.mask_zero

    @property
    def embeddings(self):
        """Getter method for (posterior) embeddings."""
        return self.posterior.embeddings
