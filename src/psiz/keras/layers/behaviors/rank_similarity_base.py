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

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from psiz.keras.layers.gates.gate_adapter import GateAdapter
from psiz.utils.m_prefer_n import m_prefer_n


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="RankSimilarityBase"
)
class RankSimilarityBase(tf.keras.layers.Layer):
    """A base layer for rank similarity behavior."""

    def __init__(
        self,
        n_reference=None,
        n_select=None,
        percept=None,
        kernel=None,
        percept_adapter=None,
        kernel_adapter=None,
        data_scope=None,
        **kwargs
    ):
        """Initialize.

        Args:
            n_reference: An integer indicating the number of references
                used for each trial.
            n_select: An integer indicating the number of references
                selected for each trial.
            percept: A Keras Layer for computing perceptual embeddings.
            kernel: A Keras Layer for computing kernel similarity.
            percept_adapter (optional): A layer for adapting inputs
                to match the assumptions of the provided `percept`
                layer.
            kernel_adapter (optional): A layer for adapting inputs
                to match the assumptions of the provided `kernel`
                layer.
            data_scope (optional): String indicating the behavioral
                data that should be used for the layer.

        """
        super(RankSimilarityBase, self).__init__(**kwargs)
        self.n_reference = int(n_reference)
        self.n_select = int(n_select)
        self.percept = percept
        self.kernel = kernel

        # Derive data scope from configuration.
        if data_scope is None:
            data_scope = "given{0}rank{1}".format(n_reference, n_select)
        self.data_scope = data_scope

        # Configure percept adapter.
        if percept_adapter is None:
            # Default adapter has no gating keys.
            percept_adapter = GateAdapter(format_inputs_as_tuple=True)
        self.percept_adapter = percept_adapter
        # Set required input keys.
        self.percept_adapter.input_keys = [data_scope + "_stimulus_set"]

        # Configure kernel adapter.
        if kernel_adapter is None:
            # Default adapter has not gating keys.
            kernel_adapter = GateAdapter(format_inputs_as_tuple=True)
        self.kernel_adapter = kernel_adapter
        self.kernel_adapter.input_keys = [data_scope + "_z_q", data_scope + "_z_r"]

    def build(self, input_shape):
        """Build.

        Args:
            input_shape: Dictionary that should include a key like
                `*_stimulus_set` with
                shape=(batch_size, max_reference + 1).

        """
        # We assume axes semantics based on relative position from last axis.
        stimuli_axis = -1  # i.e., query and reference indices.
        outcome_axis = 0  # i.e., the different judgment outcomes.
        # Convert from *relative* axis index to *absolute* axis index.
        n_axis = len(input_shape[self.data_scope + "_stimulus_set"])
        self._stimuli_axis = n_axis + stimuli_axis
        self._stimuli_axis_tensor = tf.constant(self._stimuli_axis)
        self._outcome_axis = n_axis + outcome_axis
        self._outcome_axis_tensor = tf.constant(self._outcome_axis)

        # Preassemble a reference index for expected `stimulus_set`.
        # Tensor to grab references only (i.e., drop query index).
        self._n_reference = tf.constant(self.n_reference)
        self._reference_indices = tf.range(
            tf.constant(1), tf.constant(self.n_reference + 1)
        )

        # Determine what the shape of `z_q` and `z_r` for the stimulus axis.
        # NOTE: We use `n_axis + 1` in anticipation of the added "embedding
        # dimension axis".
        z_q_shape = [None] * (n_axis + 1)
        z_q_shape[self._stimuli_axis] = 1
        self._z_q_shape = z_q_shape
        z_r_shape = [None] * (n_axis + 1)
        z_r_shape[self._stimuli_axis] = self.n_reference
        self._z_r_shape = z_r_shape

        # Prebuild "outcome indices" that indicate all the possible
        # n-rank-m behavioral outcomes.
        self._outcome_idx, self._n_outcome = self._possible_outcomes()

        # Prebuild a "selection mask" which will be used to mask probabilities
        # associated with non-selection events.
        selection_mask = np.zeros([self.n_reference])
        for i_select in range(self.n_select):
            selection_mask[i_select] = 1.0
        selection_mask = tf.constant(selection_mask, dtype=K.floatx())
        # Add any necessary leading axes before stimulus axis.
        if self._stimuli_axis > 0:
            for i_axis in range(self._stimuli_axis):
                selection_mask = tf.expand_dims(selection_mask, 0)
        # Add outcome axis.
        selection_mask = tf.expand_dims(selection_mask, self._outcome_axis)
        self._selection_mask = selection_mask

    def _possible_outcomes(self):
        """Return the possible outcomes of a rank similarity trial.

        The possible outcomes depends on `n_reference` and `n_select`.

        Returns:
            An 2D Tensor indicating all possible outcomes where the
                values indicate indices of the reference stimuli. Since
                the Tensor will be used in `tf.gather`, each column
                (not row) corresponds to one outcome. Note the indices
                refer to references only and do not include an index
                for the query. Also note that the unpermuted index is
                returned first.

        """
        # TODO maybe encapsulate this expansion method elsewhere so it
        # can be used by other objects and we can be certain that the
        # order is the same.
        n_reference = self.n_reference
        n_select = self.n_select
        outcome_idx = m_prefer_n(n_reference, n_select)
        n_outcome = outcome_idx.shape[0]
        # Transpose `outcome_idx` to make more efficient when used inside
        # `call` method.
        outcome_idx = tf.transpose(tf.constant(outcome_idx))
        n_outcome = tf.constant(n_outcome, dtype=K.floatx())
        return outcome_idx, n_outcome

    def _split_stimulus_set(self, z):
        """Split embedded stimulus set into query and reference.

        Args:
            z: A tensor of embeddings.
                shape=TensorShape(
                    [batch_size, n_ref + 1, n_dim]
                )

        Returns:
            z_q: A tensor of embeddings for the query.
                shape=TensorShape(
                    [batch_size, 1, n_dim]
                )
            z_r: A tensor of embeddings for the references.
                shape=TensorShape(
                    [batch_size, n_ref, n_dim]
                )

        """
        # Split query and reference embeddings:
        z_q, z_r = tf.split(z, [1, self._n_reference], self._stimuli_axis)

        # The `tf.split` op does not infer split dimension shape.
        # TODO Is this still necessary given `_n_reference` defined in
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
        # NOTE: Equivalent to:
        #     is_reference_present = stimulus_set[:, 1:]
        reference_stimulus_set = tf.gather(
            stimulus_set,
            indices=self._reference_indices,
            axis=self._stimuli_axis_tensor,
        )
        # NOTE: Assumes `mask_zero=True`.
        return tf.math.not_equal(reference_stimulus_set, 0)

    def _pairwise_similarity(self, inputs_copied):
        """Compute pairwise similarity."""
        # Embed stimuli indices in n-dimensional space.
        inputs_percept = self.percept_adapter(inputs_copied)
        z = self.percept(inputs_percept)
        # TensorShape=(batch_size, n, [m, ...] n_dim])

        # Prepare retrieved embeddings points for kernel and then compute
        # similarity.
        z_q, z_r = self._split_stimulus_set(z)
        inputs_copied.update(
            {self.data_scope + "_z_q": z_q, self.data_scope + "_z_r": z_r}
        )
        inputs_kernel = self.kernel_adapter(inputs_copied)
        sim_qr = self.kernel(inputs_kernel)
        return sim_qr

    def _compute_outcome_probability(self, is_reference_present, sim_qr):
        """Compute outcome probability.

        Args:
            is_reference_present:
                shape=(batch_size, n_reference)
            sim_qr:
                shape=(batch_size, n_reference)

        NOTE: This computation takes advantage of log-probability
            space, exploiting the fact that log(prob=1)=1 to make
            vectorization cleaner.

        """
        # Zero out similarities involving placeholder/mask IDs using a
        # mask based on reference indices.
        # NOTE: Using presence of references only, since query indices have
        # effectively been "consumed" by the similarity operation.
        # NOTE: `is_reference_present` only relevant for placeholder trials
        # since all trials will have the same number of references.
        is_reference_present = tf.cast(is_reference_present, K.floatx())
        # Zero out non-present similarities.
        sim_qr = tf.math.multiply(
            sim_qr, is_reference_present, name="rank_sim_zero_out_nonpresent"
        )

        # Add trialing outcome axis to `sim_qr` that reflects all possible
        # outcomes.
        sim_qr = tf.gather(sim_qr, self._outcome_idx, axis=self._stimuli_axis_tensor)
        # Add singleton outcome axis to `is_reference_present`.
        is_reference_present = tf.expand_dims(is_reference_present, self._outcome_axis)

        # Determine if outcome is legitimate by checking if at least one
        # reference is present. This is important because some trials are
        # placeholders.
        # NOTE: Equivalent to:
        #     is_outcome = is_reference_present[:, 0]
        is_outcome = tf.gather(
            is_reference_present, indices=tf.constant(0), axis=self._stimuli_axis_tensor
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

        # Mask non-existent selection events (i.e, non-existent reference
        # selections).
        event_logprob = self._selection_mask * event_logprob

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

        # NOTE: Some trials will be placeholders, so we adjust the output
        # probability to be uniform so that downstream loss computation
        # doesn't generate nan's.
        prob_placeholder = tf.cast(tf.math.equal(total_outcome_prob, 0.0), K.floatx())
        outcome_prob = outcome_prob + (prob_placeholder / self._n_outcome)
        total_outcome_prob = total_outcome_prob + prob_placeholder

        # Smooth out any numerical erros in probabilities.
        outcome_prob = tf.math.divide(outcome_prob, total_outcome_prob)
        return outcome_prob

    def get_config(self):
        """Return layer configuration."""
        config = super(RankSimilarityBase, self).get_config()
        config.update(
            {
                "n_reference": self.n_reference,
                "n_select": self.n_select,
                "percept": tf.keras.utils.serialize_keras_object(self.percept),
                "kernel": tf.keras.utils.serialize_keras_object(self.kernel),
                "percept_adapter": tf.keras.utils.serialize_keras_object(
                    self.percept_adapter
                ),
                "kernel_adapter": tf.keras.utils.serialize_keras_object(
                    self.kernel_adapter
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        percept_serial = config["percept"]
        kernel_serial = config["kernel"]
        percept_adapter_serial = config["percept_adapter"]
        kernel_adapter_serial = config["kernel_adapter"]
        config["percept"] = tf.keras.layers.deserialize(percept_serial)
        config["kernel"] = tf.keras.layers.deserialize(kernel_serial)
        config["percept_adapter"] = tf.keras.layers.deserialize(percept_adapter_serial)
        config["kernel_adapter"] = tf.keras.layers.deserialize(kernel_adapter_serial)
        return super().from_config(config)
