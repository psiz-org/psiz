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
    RankSimilarityBase: A base layer for rank-similarity judgments.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

from psiz.keras.mixins.stochastic_mixin import StochasticMixin
from psiz.keras.layers.gates.gate_adapter import GateAdapter


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='RankSimilarityBase'
)
class RankSimilarityBase(StochasticMixin, tf.keras.layers.Layer):
    """A base layer for rank similarity behavior."""
    def __init__(
        self,
        percept=None,
        kernel=None,
        percept_adapter=None,
        kernel_adapter=None,
        **kwargs
    ):
        """Initialize.

        Args:
            percept: A Keras Layer for computing perceptual embeddings.
            kernel: A Keras Layer for computing kernel similarity.
            percept_adapter (optional): A layer for adapting inputs
                to match the assumptions of the provided `percept`
                layer.
            kernel_adapter (optional): A layer for adapting inputs
                to match the assumptions of the provided `kernel`
                layer.

        """
        super(RankSimilarityBase, self).__init__(**kwargs)
        self.percept = percept
        self.kernel = kernel

        # Configure percept adapter.
        if percept_adapter is None:
            # Default adapter has no gating keys.
            percept_adapter = GateAdapter(
                format_inputs_as_tuple=True
            )
        self.percept_adapter = percept_adapter
        # Set required input keys.
        self.percept_adapter.input_keys = ['rank_similarity_stimulus_set']

        # Configure kernel adapter.
        if kernel_adapter is None:
            # Default adapter has not gating keys.
            kernel_adapter = GateAdapter(
                format_inputs_as_tuple=True
            )
        self.kernel_adapter = kernel_adapter
        self.kernel_adapter.input_keys = [
            'rank_similarity_z_q', 'rank_similarity_z_r'
        ]

    def build(self, input_shape):
        """Build.

        Expect:
        rank_similarity_stimulus_set:
        shape=(batch_size, [1,] max_reference + 1, n_outcome)

        rank_similarity_is_select:
        shape = (batch_size, [1,] n_max_reference + 1, 1)

        """
        # We assume axes semantics based on relative position from last axis.
        stimuli_axis = -2  # i.e., query and reference indices.
        outcome_axis = -1  # i.e., the different judgment outcomes.
        # Convert from *relative* axis index to *absolute* axis index.
        n_axis = len(input_shape['rank_similarity_stimulus_set'])
        self._stimuli_axis = tf.constant(n_axis + stimuli_axis)
        self._outcome_axis = tf.constant(n_axis + outcome_axis)

        # Determine the maximum number of references and precompute an index
        # Tensor to grab references only (i.e., drop query index).
        max_n_reference = (
            input_shape['rank_similarity_stimulus_set'][self._stimuli_axis]
        ) - 1
        self._max_n_reference = tf.constant(max_n_reference)
        self._reference_indices = tf.range(tf.constant(1), max_n_reference + 1)

        # Determine what the shape of `z_q` `z_r` for the "stimulus axis".
        # NOTE: We use `n_axis + 1` in anticipation of the added "embedding
        # dimension axis".
        z_q_shape = [None] * (n_axis + 1)
        z_q_shape[self._stimuli_axis] = 1
        self._z_q_shape = z_q_shape
        z_r_shape = [None] * (n_axis + 1)
        z_r_shape[self._stimuli_axis] = max_n_reference
        self._z_r_shape = z_r_shape

    def _split_stimulus_set(self, z):
        """Split embedded stimulus set into query and reference.

        Args:
            z: A tensor of embeddings.
                shape=TensorShape(
                    [batch_size, [n_sample,] n_ref + 1, n_outcome, n_dim]
                )

        Returns:
            z_q: A tensor of embeddings for the query.
                shape=TensorShape(
                    [batch_size, [n_sample,] 1, n_outcome, n_dim]
                )
            z_r: A tensor of embeddings for the references.
                shape=TensorShape(
                    [batch_size, [n_sample,] n_ref, n_outcome, n_dim]
                )

        """
        # Split query and reference embeddings:
        z_q, z_r = tf.split(z, [1, self._max_n_reference], self._stimuli_axis)

        # The `tf.split` op does not infer split dimension shape.
        # TODO Is this still necessary given `_max_n_reference` defined in
        # `build` method?
        z_q.set_shape(self._z_q_shape)
        z_r.set_shape(self._z_r_shape)

        return z_q, z_r

    def _is_reference_present(self, stimulus_set):
        """Determine if reference stimulus is present in set.

        Args:
            stimulus_set

        Returns:
            Boolean Tensor indicating if reference is present.

        """
        # NOTE: When sample axis is present, equivalent to:
        #     is_reference_present = stimulus_set[:, :, 1:]
        reference_stimulus_set = tf.gather(
            stimulus_set,
            indices=self._reference_indices,
            axis=self._stimuli_axis
        )
        # NOTE: Assumes `mask_zero=True`.
        return tf.math.not_equal(reference_stimulus_set, 0)

    def _pairwise_similarity(self, inputs_copied):
        """Compute pairwise similarity."""
        # Embed stimuli indices in n-dimensional space.
        inputs_percept = self.percept_adapter(inputs_copied)
        z = self.percept(inputs_percept)
        # TensorShape=(batch_size, [n_sample,] n, [m, ...] n_dim])

        # Prepare retrieved embeddings points for kernel and then compute
        # similarity.
        z_q, z_r = self._split_stimulus_set(z)
        inputs_copied.update({
            'rank_similarity_z_q': z_q,
            'rank_similarity_z_r': z_r
        })
        inputs_kernel = self.kernel_adapter(inputs_copied)
        sim_qr = self.kernel(inputs_kernel)
        return sim_qr

    def _compute_outcome_probability(
        self, is_reference_present, is_select, sim_qr
    ):
        """Compute outcome probability.

        NOTE: This computation takes advantage of log-probability
            space, exploiting the fact that log(prob=1)=1 to make
            vectorization cleaner.

        """
        # Zero out similarities involving placeholder/mask IDs using a
        # mask based on reference indices.
        # NOTE: Using presence of references only, since query indices have
        # effectively been "consumed" by the similarity operation.
        is_reference_present = tf.cast(is_reference_present, K.floatx())
        # Zero out non-present similarities.
        sim_qr = tf.math.multiply(
            sim_qr, is_reference_present, name='rank_sim_zero_out_nonpresent'
        )

        # Determine if outcome is legitimate by checking if at least one
        # reference is present. This is important because not all trials have
        # the same number of possible outcomes and we need to infer the
        # "zero-padding" of the outcome axis.
        # NOTE: When sample axis present, equivalent to:
        #     is_outcome = is_reference_present[:, :, 0]
        is_outcome = tf.gather(
            is_reference_present,
            indices=tf.constant(0),
            axis=self._stimuli_axis
        )

        # Compute denominator based on formulation of Luce's choice rule by
        # summing over the different references present in a trial. Note that
        # the similarity for placeholder references will be zero since they
        # were zeroed out by the multiply op with `is_reference_present` above.
        denom = tf.cumsum(sim_qr, axis=self._stimuli_axis, reverse=True)

        # Compute log-probability of each selection, assuming all selections
        # occurred. Add fuzz factor to avoid log(0)
        sim_qr = tf.maximum(sim_qr, tf.keras.backend.epsilon())
        denom = tf.maximum(denom, tf.keras.backend.epsilon())
        event_logprob = tf.math.log(sim_qr) - tf.math.log(denom)

        # Mask non-existent events (i.e, reference selections).
        is_select = tf.cast(is_select, K.floatx())
        event_logprob = is_select * event_logprob

        # Compute log-probability of outcome (i.e., a sequence of events).
        outcome_logprob = tf.reduce_sum(event_logprob, axis=self._stimuli_axis)
        outcome_prob = tf.math.exp(outcome_logprob)
        outcome_prob = is_outcome * outcome_prob

        # Clean up numerical errors in probabilities.
        # NOTE: The `reduce_sum` op above means that the outcome axis has been
        # shifted by one, so the next op uses `self._outcome_axis - 1`.
        total_outcome_prob = tf.reduce_sum(
            outcome_prob, axis=(self._outcome_axis - 1), keepdims=True
        )
        # NOTE: we use `divide_no_nan` because sometimes the
        # `total_outcome_prob` will be zero. This happens because there can
        # be placeholder examples/trials.
        outcome_prob = tf.math.divide_no_nan(
            outcome_prob, total_outcome_prob, name=None
        )
        return outcome_prob

    def get_config(self):
        """Return layer configuration."""
        config = super(RankSimilarityBase, self).get_config()
        config.update({
            'percept': tf.keras.utils.serialize_keras_object(self.percept),
            'kernel': tf.keras.utils.serialize_keras_object(self.kernel),
            'percept_adapter': tf.keras.utils.serialize_keras_object(
                self.percept_adapter
            ),
            'kernel_adapter': tf.keras.utils.serialize_keras_object(
                self.kernel_adapter
            ),
        })
        return config

    @classmethod
    def from_config(cls, config):
        percept_serial = config['percept']
        kernel_serial = config['kernel']
        percept_adapter_serial = config['percept_adapter']
        kernel_adapter_serial = config['kernel_adapter']
        config['percept'] = tf.keras.layers.deserialize(percept_serial)
        config['kernel'] = tf.keras.layers.deserialize(kernel_serial)
        config['percept_adapter'] = tf.keras.layers.deserialize(
            percept_adapter_serial
        )
        config['kernel_adapter'] = tf.keras.layers.deserialize(
            kernel_adapter_serial
        )
        return super().from_config(config)
