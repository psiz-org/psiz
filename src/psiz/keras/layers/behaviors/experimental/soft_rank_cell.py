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
    SoftRankCell: An RNN cell layer that performs soft ranking.

"""

import tensorflow as tf
from tensorflow.keras import backend

from psiz.keras.layers.behaviors.soft_rank_base import SoftRankBase
from psiz.keras.constraints.min_max import MinMax


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="SoftRankCell"
)
class SoftRankCell(SoftRankBase):
    """A stateful soft rank layer.

    A stateful layer that outputs a soft rank of items based on
    incoming 'strength' associated with each option. The outcome
    probabilities at the current timestep are biased by past outcome
    probabilities.

    The mixing with past outcomes is determined by the learnable
    parameter `inertia`. When `inertia` is 0.0, there is no bias from
    past outcomes. As `inertia` approaches 1.0, the current
    outcome probabililites are increasingly dominated by the
    probabilies associated with past outcomes.

    p_{t} = (1- inertia) * outcomes_{t} + inertia * p_{t-1}

    # TODO remove or clean up
    s.t. x + y = 1.0
    s.t. u = uniform (or a)

    a   | ya + xu
    b   | yb + xya + xxu
    c   | yc + xyb + xxya + xxxu
    d   | yd + xyc + xxyb + xxxya + xxxxu
    e   | ye + xyd + xxyc + xxxyb + xxxxya + xxxxxu
    f   | yf + xye + xxyd + xxxyc + xxxxyb + xxxxxya + xxxxxxu

    """

    def __init__(self, inertia_initializer=None, inertia_constraint=None, **kwargs):
        """Initialize.

        Args:
            inertia_initializer (optional): Initializer for `inertia` scalar.
                The default initializer is 0.0.
            inertia_constraint (optional): Constraint function applied to
                the `inertia` scalar. The default constraint is
                0 <= inertia < 1
            kwargs: Additional keyword arguments. See `SoftRankBase`

        """
        super(SoftRankCell, self).__init__(**kwargs)

        if inertia_initializer is None:
            inertia_initializer = tf.keras.initializers.Constant(0.0)
        self.inertia_initializer = tf.keras.initializers.get(inertia_initializer)
        if inertia_constraint is None:
            inertia_constraint = MinMax(0.0, 0.99)
        self.inertia_constraint = tf.keras.constraints.get(inertia_constraint)
        self.inertia = self.add_weight(
            shape=[],
            initializer=self.inertia_initializer,
            trainable=self.trainable,
            name="inertia",
            dtype=backend.floatx(),
            constraint=inertia_constraint,
        )

        # Satisfy RNNCell contract.
        # NOTE: A placeholder state.
        self.state_size = [
            tf.TensorShape([1])
        ]  # TODO need this to be determined on first call

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial state."""
        # initial_state = self._compute_outcome_probability(inputs)
        initial_state = [tf.ones([batch_size, 1], name="rank_cell_initial_state")]
        return initial_state

    def call(self, inputs, states, training=None):
        """Return probability of a ranked selection sequence.

        Args:
            inputs: A tensor indicating the strengths associated with
                each option. It is assumed that the last axis indicates
                the different options.

        Returns:
            outcome_prob: Probability of different behavioral outcomes.

        """
        outcome_prob = self._compute_outcome_probability(inputs)
        outcome_prob_t = (1 - self.inertia) * outcome_prob + (self.inertia) * states[0]
        states_tplus1 = outcome_prob_t
        return outcome_prob_t, states_tplus1
