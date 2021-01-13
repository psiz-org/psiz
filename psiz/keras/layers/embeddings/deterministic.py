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
    EmbeddingDeterministic: An deterministic embedding layer that implements
        a sample dimension via repeats.

"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingDeterministic'
)
class EmbeddingDeterministic(tf.keras.layers.Embedding):
    """A class for adding a sample dimension."""
    def __init__(self, input_dim, output_dim, n_sample=1, **kwargs):
        """Initialize.

        Arguments:
            input_dim: See tf.keras.layers.Embedding.
            output_dim: See tf.keras.layers.Embedding.
            n_sample: The number of samples to take. Since the
                embedding is deterministic, this corresponds to the
                number of repeats.
            kwargs: See tf.keras.layers.Embedding.

        """
        super().__init__(input_dim, output_dim, **kwargs)
        self.n_sample = n_sample

    def call(self, inputs):
        outputs = super().call(inputs)
        outputs = tf.expand_dims(outputs, axis=0)
        outputs = tf.repeat(outputs, self.n_sample, axis=0)
        return outputs
