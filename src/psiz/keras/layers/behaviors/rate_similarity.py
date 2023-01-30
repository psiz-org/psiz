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
    package="psiz.keras.layers", name="RateSimilarity"
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
            inputs["<data_scope>_stimulus_set"]: A tensor containing
                indices that define the stimuli used in each trial.
                shape=(batch_size, n_stimuli_per_trial)

        Returns:
            rating: The ratings (on a 0-1 scale) as determined by a
                parameterized logistic function.

        """
        # NOTE: The inputs are copied, because modifying the original `inputs`
        # is bad practice in TF. For example, it creates issues when saving
        # a model.
        inputs_copied = copy.copy(inputs)

        sim_qr = self._pairwise_similarity(inputs_copied)

        rating = self.lower + tf.math.divide(
            self.upper - self.lower,
            1 + tf.math.exp(-self.rate * (sim_qr - self.midpoint)),
        )

        return rating
