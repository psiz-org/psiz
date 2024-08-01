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


import keras
import numpy as np

import psiz.keras.constraints as pk_constraints
from psiz.keras.layers.gates.gate_adapter import GateAdapter
from psiz.utils.m_prefer_n import m_prefer_n


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="RankSimilarityBase"
)
class RankSimilarityBase(keras.layers.Layer):
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
        fit_temperature=False,
        temperature_initializer=None,
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
            fit_temperature (optional): Boolean indicating if variable
                is trainable.
            temperature_initializer (optional): Initializer for
                temperature parameter

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

        self.fit_temperature = fit_temperature
        if temperature_initializer is None:
            temperature_initializer = keras.initializers.Constant(value=1.0)
        self.temperature_initializer = keras.initializers.get(temperature_initializer)
        temperature_trainable = self.trainable and self.fit_temperature
        with keras.name_scope(self.name):
            self.temperature = self.add_weight(
                shape=[],
                initializer=self.temperature_initializer,
                trainable=temperature_trainable,
                name="temperature",
                dtype=keras.backend.floatx(),
                constraint=pk_constraints.GreaterThan(min_value=0.0),
            )

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
        self._stimuli_axis_tensor = keras.ops.convert_to_tensor(self._stimuli_axis)
        self._outcome_axis = n_axis + outcome_axis
        self._outcome_axis_tensor = keras.ops.convert_to_tensor(self._outcome_axis)

        # Preassemble a reference index for expected `stimulus_set`.
        # Tensor to grab references only (i.e., drop query index).
        self._n_reference = keras.ops.convert_to_tensor(self.n_reference)
        self._reference_indices = keras.ops.arange(1, self.n_reference + 1)

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
        selection_mask = keras.ops.convert_to_tensor(
            selection_mask, dtype=keras.backend.floatx()
        )
        # Add any necessary leading axes before stimulus axis.
        if self._stimuli_axis > 0:
            for i_axis in range(self._stimuli_axis):
                selection_mask = keras.ops.expand_dims(selection_mask, 0)
        # Add outcome axis.
        selection_mask = keras.ops.expand_dims(selection_mask, self._outcome_axis)
        self._selection_mask = selection_mask

    def _possible_outcomes(self):
        """Return the possible outcomes of a rank similarity trial.

        The possible outcomes depends on `n_reference` and `n_select`.

        Returns:
            An 2D Tensor indicating all possible outcomes where the
                values indicate indices of the reference stimuli. Since
                the Tensor will be used in `take`, each column
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
        outcome_idx = keras.ops.transpose(keras.ops.convert_to_tensor(outcome_idx))
        n_outcome = keras.ops.convert_to_tensor(n_outcome, dtype=keras.backend.floatx())
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
        z_q, z_r = keras.ops.split(z, [1], axis=self._stimuli_axis)  # TODO verify

        # TODO Is this still necessary in Keras 3? Also `_n_reference` is defined in
        # `build` method
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
        reference_stimulus_set = keras.ops.take(
            stimulus_set,
            indices=self._reference_indices,
            axis=self._stimuli_axis_tensor,
        )
        # NOTE: Assumes `mask_zero=True`.
        return keras.ops.not_equal(reference_stimulus_set, 0)

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
        is_reference_present = keras.ops.cast(
            is_reference_present, keras.backend.floatx()
        )
        # Zero out non-present similarities.
        sim_qr = keras.ops.multiply(sim_qr, is_reference_present)

        # Add trialing outcome axis to `sim_qr` that reflects all possible
        # outcomes.
        sim_qr = keras.ops.take(
            sim_qr, self._outcome_idx, axis=self._stimuli_axis_tensor
        )
        # Add singleton outcome axis to `is_reference_present`.
        is_reference_present = keras.ops.expand_dims(
            is_reference_present, self._outcome_axis
        )

        # Determine if outcome is legitimate by checking if at least one
        # reference is present. This is important because some trials are
        # placeholders.
        # NOTE: Equivalent to:
        #     is_outcome = is_reference_present[:, 0]
        is_outcome = keras.ops.take(
            is_reference_present,
            indices=keras.ops.convert_to_tensor(0),
            axis=self._stimuli_axis_tensor,
        )

        # Compute denominator based on formulation of Luce's choice rule by
        # summing over the different references present in a trial. Note that
        # the similarity for placeholder references will be zero since they
        # were zeroed out by the multiply op with `is_reference_present` above.
        denom = keras.ops.flip(
            keras.ops.cumsum(
                keras.ops.flip(sim_qr, axis=self._stimuli_axis),
                axis=self._stimuli_axis,
            ),
            axis=self._stimuli_axis,
        )

        # Compute log-probability of each selection, assuming all selections
        # occurred. Add fuzz factor to avoid log(0)
        sim_qr = keras.ops.maximum(sim_qr, keras.backend.epsilon())
        denom = keras.ops.maximum(denom, keras.backend.epsilon())
        event_logit = keras.ops.log(sim_qr) - keras.ops.log(denom)

        # Mask non-existent selection events (i.e, non-existent reference
        # selections).
        event_logit = self._selection_mask * event_logit

        # Compute log-probability of outcome (i.e., a sequence of events).
        outcome_logit = keras.ops.sum(event_logit, axis=self._stimuli_axis)

        # Prepare for softmax op.
        # Convert back to probility space.
        outcome_prob = keras.ops.exp(outcome_logit)
        outcome_prob = is_outcome * outcome_prob

        # Clean up numerical errors in probabilities.
        # NOTE: The `reduce_sum` op above means that the outcome axis has been
        # shifted by one, so the next op uses `self._outcome_axis - 1`.
        total_outcome_prob = keras.ops.sum(
            outcome_prob, axis=(self._outcome_axis - 1), keepdims=True
        )

        # NOTE: Some trials will be placeholders, so we adjust the output
        # probability to be uniform so that downstream loss computation
        # doesn't generate nan's.
        prob_placeholder = keras.ops.cast(
            keras.ops.equal(total_outcome_prob, 0.0), keras.backend.floatx()
        )
        outcome_prob = outcome_prob + (prob_placeholder / self._n_outcome)
        # TODO remove if keeping temperature calculation at end of function.
        # NOTE: could `reduce_sum` op to compute `total_outcome_prob`, but
        # the following op is slightly cheaper.
        # total_outcome_prob = total_outcome_prob + prob_placeholder

        # Compute softmax using optional temperature parameter.
        outcome_prob = keras.ops.softmax(
            keras.ops.divide(keras.ops.log(outcome_prob), self.temperature)
        )
        return outcome_prob

    def get_config(self):
        """Return layer configuration."""
        config = super(RankSimilarityBase, self).get_config()
        config.update(
            {
                "n_reference": self.n_reference,
                "n_select": self.n_select,
                "percept": keras.saving.serialize_keras_object(self.percept),
                "kernel": keras.saving.serialize_keras_object(self.kernel),
                "percept_adapter": keras.saving.serialize_keras_object(
                    self.percept_adapter
                ),
                "kernel_adapter": keras.saving.serialize_keras_object(
                    self.kernel_adapter
                ),
                "data_scope": self.data_scope,
                "fit_temperature": self.fit_temperature,
                "temperature_initializer": keras.initializers.serialize(
                    self.temperature_initializer
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
        config["percept"] = keras.saving.deserialize_keras_object(percept_serial)
        config["kernel"] = keras.saving.deserialize_keras_object(kernel_serial)
        config["percept_adapter"] = keras.saving.deserialize_keras_object(
            percept_adapter_serial
        )
        config["kernel_adapter"] = keras.saving.deserialize_keras_object(
            kernel_adapter_serial
        )
        return super().from_config(config)
