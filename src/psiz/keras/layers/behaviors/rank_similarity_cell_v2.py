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
    RankSimilarityCellV2: A rank behavior layer.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

from psiz.keras.layers.behaviors.behavior import Behavior
from psiz.keras.layers.gates.gate_adapter import GateAdapter


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='RankSimilarityCellV2'
)
class RankSimilarityCellV2(Behavior):
    """A rank similarity behavior layer."""
    def __init__(
        self,
        percept=None,
        kernel=None,
        percept_gate_weights_keys=None,
        kernel_gate_weights_keys=None,
        **kwargs
    ):
        """Initialize.

        Args:
            percept: A Keras Layer for computing perceptual embeddings.
            kernel: A Keras Layer for computing kernel similarity.
            percept_gate_weights_keys (optional): A list of dictionary
                keys pointing to gate weights that should be passed to
                the `percept` layer. Since the `percept` layer assumes
                a tuple is passed to the `call` method, the weights are
                appended at the end of the "standard" Tensors, in the
                same order specified by the user.
            kernel_gate_weights_keys (optional): A list of dictionary
                keys pointing to gate weights that should be passed to
                the `kernel` layer. Since the `kernel` layer assumes a
                tuple is passed to the `call` method, the weights are
                appended at the end of the "standard" Tensors, in the
                same order specified by the user.

        """
        super(RankSimilarityCellV2, self).__init__(**kwargs)
        self.percept = percept
        self.kernel = kernel

        # Set up adapters based on provided gate weights.
        self._percept_adapter = GateAdapter(
            subnet=percept,
            input_keys=['rank_similarity_stimset_samples'],
            gate_weights_keys=percept_gate_weights_keys,
            format_inputs_as_tuple=True
        )
        self._kernel_adapter = GateAdapter(
            subnet=kernel,
            input_keys=['rank_similarity_z_q', 'rank_similarity_z_q'],
            gate_weights_keys=kernel_gate_weights_keys,
            format_inputs_as_tuple=True
        )

        # Satisfy RNNCell contract.
        # NOTE: A placeholder state.
        self.state_size = [
            tf.TensorShape([1])
        ]

    def _split_stimulus_set(self, z):
        """Split stimulus set into query and reference.

        Args:
            z: A tensor of embeddings.
                shape=TensorShape(
                    [batch_size, n_sample, n_ref + 1, n_outcome, n_dim]
                )

        Returns:
            z_q: A tensor of embeddings for the query.
                shape=TensorShape(
                    [batch_size, n_sample, 1, n_outcome, n_dim]
                )
            z_r: A tensor of embeddings for the references.
                shape=TensorShape(
                    [batch_size, n_sample, n_ref, n_outcome, n_dim]
                )

        """
        # Define some useful variables before manipulating inputs.
        max_n_reference = tf.shape(z)[-3] - 1

        # Split query and reference embeddings:
        # z_q: TensorShape([batch_size, sample_size, 1, n_outcome, n_dim]
        # z_r: TensorShape([batch_size, sample_size, n_ref, n_outcome, n_dim]
        z_q, z_r = tf.split(z, [1, max_n_reference], -3)
        # The tf.split op does not infer split dimension shape. We know that
        # z_q will always have shape=1, but we don't know `max_n_reference`
        # ahead of time.
        z_q.set_shape([None, self.n_sample, 1, None, None])  # TODO necessary?

        return z_q, z_r

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial state."""
        initial_state = [
            tf.zeros([batch_size, 1], name='rank_cell_initial_state')
        ]
        return initial_state

    def get_mask(self, inputs):
        """Return appropriate mask."""
        mask = tf.not_equal(inputs['rank_similarity_stimulus_set'], 0)
        return mask[:, :, 0, 0, 0]

    def call(self, inputs, states, training=None):
        """Return probability of a ranked selection sequence.

        Args:
            inputs: A dictionary containing the following information:
                rank_similarity_stimulus_set: A tensor containing
                    indices that define the stimuli used in each trial.
                    shape=(batch_size, 1, max_reference + 1, n_outcome)
                rank_similarity_is_select: A float tensor indicating if
                    a reference was selected, which corresponds to a
                    "true" probabilistic event.
                    shape = (batch_size, 1, n_max_reference + 1, 1)
                gate_weights (optional): Tensor(s) containing gate
                    weights. The actual key value(s) will depend on how
                    the user initialized the layer.

        Returns:
            outcome_prob: Probability of different behavioral outcomes.

        NOTE: This computation takes advantage of log-probability
            space, exploiting the fact that log(prob=1)=1 to make
            vectorization cleaner.

        """
        stimulus_set = inputs['rank_similarity_stimulus_set']
        # NOTE: We drop the "query" position in `is_select`.
        is_select = inputs['rank_similarity_is_select'][:, :, 1:, :]

        # Expand `sample_axis` of `stimulus_set` for stochastic
        # functionality (e.g., variational inference).
        stimulus_set = tf.repeat(
            stimulus_set, self.n_sample, axis=self.sample_axis_in_cell
        )

        # Embed stimuli indices in n-dimensional space.
        inputs.update({
            'rank_similarity_stimset_samples': stimulus_set
        })
        z = self._percept_adapter(inputs)
        # TensorShape=(batch_size, n_sample, n, [m, ...] n_dim])

        # Prepare retrieved embeddings points for kernel and then compute
        # similarity.
        z_q, z_r = self._split_stimulus_set(z)
        inputs.update({
            'rank_similarity_z_q': z_q,
            'rank_similarity_z_r': z_r
        })
        sim_qr = self._kernel_adapter(inputs)

        # Zero out similarities involving placeholder IDs by creating
        # a mask based on reference indices. We drop the query indices
        # because they have effectively been "consumed" by the similarity
        # operation.
        is_present = tf.cast(
            tf.math.not_equal(stimulus_set[:, :, 1:], 0), K.floatx()
        )
        sim_qr = tf.math.multiply(
            sim_qr, is_present, name='rank_zero_nonpresent'
        )

        # Determine if outcome is legitimate by checking if at least one
        # reference is present. This is important because not all trials have
        # the same number of possible outcomes and we need to infer the
        # "zero-padding" of the outcome axis.
        is_outcome = is_present[:, :, 0, :]

        # Compute denominator based on formulation of Luce's choice rule by
        # summing over the different references present in a trial. Note that
        # the similarity for placeholder references will be zero since they
        # were zeroed out by the caller.
        denom = tf.cumsum(sim_qr, axis=2, reverse=True)

        # Compute log-probability of each selection, assuming all selections
        # occurred. Add fuzz factor to avoid log(0)
        sim_qr = tf.maximum(sim_qr, tf.keras.backend.epsilon())
        denom = tf.maximum(denom, tf.keras.backend.epsilon())
        event_logprob = tf.math.log(sim_qr) - tf.math.log(denom)

        # Mask non-existent events (i.e, reference selections).
        is_select = tf.cast(is_select, K.floatx())
        event_logprob = is_select * event_logprob

        # Compute log-probability of outcome (i.e., a sequence of events).
        outcome_logprob = tf.reduce_sum(event_logprob, axis=2)
        outcome_prob = tf.math.exp(outcome_logprob)
        outcome_prob = is_outcome * outcome_prob

        # Clean up numerical errors in probabilities.
        total_outcome_prob = tf.reduce_sum(outcome_prob, axis=2, keepdims=True)
        outcome_prob = outcome_prob / total_outcome_prob

        states_tplus1 = [states[0] + 1]
        return outcome_prob, states_tplus1

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'percept': tf.keras.utils.serialize_keras_object(self.percept),
            'kernel': tf.keras.utils.serialize_keras_object(self.kernel),
            'percept_gate_weights_keys': (
                self._percept_adapter.gate_weights_keys
            ),
            'kernel_gate_weights_keys': self._kernel_adapter.gate_weights_keys,
        })
        return config

    @classmethod
    def from_config(cls, config):
        percept_serial = config['percept']
        kernel_serial = config['kernel']
        config['percept'] = tf.keras.layers.deserialize(percept_serial)
        config['kernel'] = tf.keras.layers.deserialize(kernel_serial)
        return super().from_config(config)
