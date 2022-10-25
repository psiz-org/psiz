# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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
"""Module of TensorFlow behavior layers.

Classes:
    RateSimilarity: A (stateless) layer for rate-similarity judgments.

"""

import copy

import tensorflow as tf

from psiz.keras.layers.behaviors.rate_similarity_base import RateSimilarityBase


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='RateSimilarity'
)
class RateSimilarity(RateSimilarityBase):
    """A rate similarity behavior layer."""
    def __init__(self, **kwargs):
        """Initialize.

        Args:
            See `RateSimilarityBase`

        """
        super(RateSimilarity, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        """Return predicted rating of a trial.

        Args:
            inputs: A dictionary containing the following information:
                rate_similarity_stimulus_set: A tensor containing
                    indices that define the stimuli used in each trial.
                    shape=(batch_size, [n_sample,] n_stimuli_per_trial)
                gate_weights (optional): Tensor(s) containing gate
                    weights. The actual key value(s) will depend on how
                    the user initialized the layer.

        Returns:
            probs: The probabilites as determined by a parameterized
                logistic function.

        """
        # NOTE: The inputs are copied, because modifying the original `inputs`
        # is bad practice in TF. For example, it creates issues when saving
        # a model.
        inputs_copied = copy.copy(inputs)

        stimulus_set = inputs_copied['rate_similarity_stimulus_set']

        # TODO delete
        # Expand `sample_axis` of `stimulus_set` for stochastic
        # functionality (e.g., variational inference).
        # stimulus_set = tf.repeat(
        #     stimulus_set, self.n_sample, axis=self.sample_axis
        # )

        # Embed stimuli indices in n-dimensional space.
        inputs_copied.update({
            'rate_similarity_stimset_samples': stimulus_set
        })
        z = self._percept_adapter(inputs_copied)
        # TensorShape=(batch_size, [n_sample,] 2, n_dim])

        # Prepare retrieved embeddings point for kernel and then compute
        # similarity.
        z_q, z_r = self._split_stimulus_set(z)
        inputs_copied.update({
            'rate_similarity_z_q': z_q,
            'rate_similarity_z_r': z_r
        })
        sim_qr = self._kernel_adapter(inputs_copied)

        prob = self.lower + tf.math.divide(
            self.upper - self.lower,
            1 + tf.math.exp(-self.rate * (sim_qr - self.midpoint))
        )

        return prob
