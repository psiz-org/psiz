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
    RateSimilarityCell: A rate similarity layer.

"""

import copy

import tensorflow as tf

from psiz.keras.layers.behaviors.rate_similarity_base import RateSimilarityBase


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="RateSimilarityCell"
)
class RateSimilarityCell(RateSimilarityBase):
    """A stateful rate behavior layer.

    Similarities are converted to probabilities using a parameterized
    logistic function,

    p(x) = lower + ((upper - lower) / (1 + exp(-rate*(x - midpoint))))

    with the following variable meanings:
    `lower`: The lower asymptote of the function's range.
    `upper`: The upper asymptote of the function's range.
    `midpoint`: The midpoint of the function's domain and point of
    maximum growth.
    `rate`: The growth rate of the logistic function.

    """

    def __init__(self, **kwargs):
        """Initialize.

        Args:
            See `RateSimilarityBase`

        """
        super(RateSimilarityCell, self).__init__(**kwargs)

        # Satisfy RNNCell contract.
        # NOTE: A placeholder state.
        self.state_size = [tf.TensorShape([1])]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial state."""
        initial_state = [tf.zeros([batch_size, 1], name="rate_cell_initial_state")]
        return initial_state

    def call(self, inputs, states, training=None):
        """Return predicted rating of a trial.

        Args:
            inputs["<data_scope>_stimulus_set"]: A tensor containing
                indices that define the stimuli used in each trial.
                shape=(batch_size, n_sample, n_stimuli_per_trial)

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

        return rating, states
