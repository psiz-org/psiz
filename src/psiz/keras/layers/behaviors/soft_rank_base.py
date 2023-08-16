# -*- coding: utf-8 -*-
# Copyright 2023 The PsiZ Authors. All Rights Reserved.
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
    SoftRankBase

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend

import psiz.keras.constraints as pk_constraints
from psiz.utils.m_prefer_n import m_prefer_n


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="SoftRankBase"
)
class SoftRankBase(tf.keras.layers.Layer):
    """A base layer for ranking options.

    A base layer that outputs a soft rank of items based on incoming
    'strength' associated with each option.

    Inputs are assumed to be the 'strength' **in favor** of an option.
    The probability of selecting a given option is proportional to its
    strength. The `temperature` parameter adjust the determinism of
    the ranking.

    The number of options are inferred when the layer is built. Once,
    built, the soft rank layer can only be used for inputs with the
    specified number of options.

    """

    def __init__(
        self,
        n_select=None,
        temperature_initializer=None,
        temperature_constraint=None,
        temperature_regularizer=None,
        **kwargs
    ):
        """Initialize.

        Args:
            n_select: An integer indicating the number of options
                selected and ranked (for each trial), must be less than
                the number of options, where the number of options is
                implied by the last axis of the input.
            temperature_initializer (optional): Initializer for
                `temperature` scalar parameter.
            temperature_constraint (optional): Constraint function
                applied to `temperature` scalar parameter.
            temperature_regularizer (optional): Regularizer function
                applied to `temperature` scalar parameter.

        """
        super(SoftRankBase, self).__init__(**kwargs)
        self.n_select = int(n_select)

        if temperature_initializer is None:
            temperature_initializer = tf.keras.initializers.Constant(value=1.0)
        self.temperature_initializer = tf.keras.initializers.get(
            temperature_initializer
        )

        if temperature_constraint is None:
            temperature_constraint = pk_constraints.GreaterThan(min_value=0.0)
        self.temperature_constraint = tf.keras.constraints.get(temperature_constraint)

        self.temperature_regularizer = tf.keras.regularizers.get(
            temperature_regularizer
        )

        with tf.name_scope(self.name):
            self.temperature = self.add_weight(
                shape=[],
                initializer=self.temperature_initializer,
                trainable=self.trainable,
                name="temperature",
                dtype=backend.floatx(),
                constraint=self.temperature_constraint,
                regularizer=self.temperature_regularizer,
            )

    def build(self, input_shape):
        """Build.

        Args:
            input_shape: Shape of `strength` tensor.
                shape=(batch_size, [m, n, ...] n_option).

        """
        # We assume axes semantics based on relative position from last axis.
        option_axis = -1  # i.e., the different options.
        outcome_axis = 0  # i.e., the different judgment outcomes (this axis is added during the call).
        # Convert from *relative* axis index to *absolute* axis index.
        n_axis = len(input_shape)
        self._option_axis = n_axis + option_axis
        self._option_axis_tensor = tf.constant(self._option_axis)
        self._outcome_axis = n_axis + outcome_axis
        self._outcome_axis_tensor = tf.constant(self._outcome_axis)

        self.n_option = input_shape[self._option_axis]
        if self.n_select >= self.n_option:
            raise ValueError(
                "The argument `n_select` must be less than the "
                "number of options implied by the inputs."
            )
        # Prebuild "outcome indices" that indicate all the possible
        # n-rank-m behavioral outcomes.
        self._outcome_idx, self._n_outcome = self._possible_outcomes()

        # Prebuild a "selection mask" which will be used to mask probabilities
        # associated with non-selection events.
        selection_mask = np.zeros([self.n_option])
        for i_select in range(self.n_select):
            selection_mask[i_select] = 1.0
        selection_mask = tf.constant(selection_mask, dtype=backend.floatx())
        # Add any necessary leading axes before stimulus axis.
        if self._option_axis > 0:
            for i_axis in range(self._option_axis):
                selection_mask = tf.expand_dims(selection_mask, 0)
        # Add outcome axis.
        selection_mask = tf.expand_dims(selection_mask, self._outcome_axis)
        self._selection_mask = selection_mask

    def _possible_outcomes(self):
        """Return the possible outcomes of a rank similarity trial.

        The possible outcomes depends on `n_option` and `n_select`.

        Returns:
            An 2D Tensor indicating all possible outcomes where the
                values indicate indices of the options. Since
                the Tensor will be used in `tf.gather`, each column
                (not row) corresponds to one outcome. Note the indices
                are zero-indexed relative to the options and the
                unpermuted index is returned first.

        """
        n_option = self.n_option
        n_select = self.n_select
        outcome_idx = m_prefer_n(n_option, n_select)
        n_outcome = outcome_idx.shape[0]
        # Transpose `outcome_idx` to make more efficient when used inside
        # `call` method.
        outcome_idx = tf.transpose(tf.constant(outcome_idx))
        n_outcome = tf.constant(n_outcome, dtype=backend.floatx())
        return outcome_idx, n_outcome

    def _compute_outcome_probability(self, strength):
        """Compute outcome probability.

        Args:
            strength:
                shape=(batch_size, [m, n, ...] n_option)

        NOTE: This computation takes advantage of log-probability
            space, exploiting the fact that log(prob=1)=1 to make
            vectorization cleaner.

        """
        # NOTE: Keeping `is_option_present` explicit for now in case refactor
        # is necessary later. If keeping, remove commented casting line.
        is_option_present = tf.ones_like(strength)
        # is_option_present = tf.cast(is_option_present, backend.floatx())

        # Zero out "non-present" strengths.
        # NOTE: `is_option_present` only relevant for placeholder trials
        # since all trials will have the same number of options.
        strength = tf.math.multiply(
            strength, is_option_present, name="rank_sim_zero_out_nonpresent"
        )

        # Create and populate "outcome" axis to `strength` that reflects all
        # possible outcomes.
        strength = tf.gather(strength, self._outcome_idx, axis=self._option_axis_tensor)
        # Add singleton outcome axis to `is_option_present` to keep tensor shapes
        # consistent.
        is_option_present = tf.expand_dims(is_option_present, self._outcome_axis)

        # Determine if outcome is legitimate by checking if at least one
        # option is present. This is important because some trials are
        # placeholders. Analogous to: `is_outcome = is_option_present[:, 0]`
        is_outcome = tf.gather(
            is_option_present, indices=tf.constant(0), axis=self._option_axis_tensor
        )

        # Compute denominator based on formulation of Luce's choice rule by
        # summing over the different options present in a trial. Note that
        # the similarity for placeholder options will be zero since they
        # were zeroed out by the multiply op with `is_option_present` above.
        denom = tf.cumsum(strength, axis=self._option_axis, reverse=True)

        # Compute log-probability of each selection, assuming all selections
        # occurred. Add fuzz factor to avoid log(0)
        strength = tf.maximum(strength, tf.keras.backend.epsilon())
        denom = tf.maximum(denom, tf.keras.backend.epsilon())
        event_logit = tf.math.log(strength) - tf.math.log(denom)

        # Mask non-existent selection events (i.e, non-existent option
        # selections).
        event_logit = self._selection_mask * event_logit

        # Compute log-probability of outcome (i.e., a sequence of events).
        outcome_logit = tf.reduce_sum(event_logit, axis=self._option_axis)

        # Prepare for softmax op by converting back to probility space.
        outcome_prob = tf.math.exp(outcome_logit)
        outcome_prob = is_outcome * outcome_prob

        # Some trials will be placeholders, so we adjust the output
        # probability to be uniform so that downstream loss computation
        # does not generate nan's.
        # NOTE: The `reduce_sum` op above means that the outcome axis has been
        # shifted by one, so the next `resuce_sum` op uses `self._outcome_axis - 1`.
        total_outcome_prob = tf.reduce_sum(
            outcome_prob, axis=(self._outcome_axis - 1), keepdims=True
        )
        prob_placeholder = tf.cast(
            tf.math.equal(total_outcome_prob, 0.0), backend.floatx()
        )
        outcome_prob = outcome_prob + (prob_placeholder / self._n_outcome)

        # Compute softmax using (optional) temperature parameter.
        outcome_prob = tf.nn.softmax(
            tf.math.divide(tf.math.log(outcome_prob), self.temperature)
        )
        return outcome_prob

    def get_config(self):
        """Return layer configuration."""
        config = super(SoftRankBase, self).get_config()
        config.update(
            {
                "n_select": self.n_select,
                "temperature_initializer": tf.keras.initializers.serialize(
                    self.temperature_initializer
                ),
                "temperature_constraint": tf.keras.constraints.serialize(
                    self.temperature_constraint
                ),
                "temperature_regularizer": tf.keras.regularizers.serialize(
                    self.temperature_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)
