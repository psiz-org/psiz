# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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
"""Module for a TensorFlow layers.

Classes:
    ALCOVECell: An RNN-compatible cell implementing the ALCOVE Model.

"""

import copy

import keras
import numpy as np
import tensorflow as tf

import psiz.keras.constraints as pk_constraints
from psiz.keras.layers.gates.gate_adapter import GateAdapter
from psiz.keras.ops.wpnorm import wpnorm


@keras.saving.register_keras_serializable(package="psiz.keras", name="ALCOVECell")
class ALCOVECell(keras.layers.Layer):
    """An RNN-compatible cell implementing ALCOVE."""

    def __init__(
        self,
        units=None,
        percept=None,
        percept_adapter=None,
        similarity=None,
        rho_trainable=True,
        rho_initializer=None,
        rho_regularizer=None,
        rho_constraint=None,
        temperature_trainable=True,
        temperature_initializer=None,
        temperature_regularizer=None,
        temperature_constraint=None,
        lr_attention_trainable=True,
        lr_attention_initializer=None,
        lr_attention_regularizer=None,
        lr_attention_constraint=None,
        lr_association_trainable=True,
        lr_association_initializer=None,
        lr_association_regularizer=None,
        lr_association_constraint=None,
        data_scope=None,
        **kwargs,
    ):
        """Initialize.

        Args:
            units: Positive integer indicating the number of output classes.
            percept: A Keras Layer for computing perceptual embeddings.
            percept_adapter (optional): A layer for adapting inputs
                to match the assumptions of the provided `percept`
                layer.
            similarity: A Keras Layer for mapping distance to
                similarity.
            rho_trainable (optional):
            rho_initializer (optional):
            rho_regularizer (optional):
            rho_constraint (optional):
            temperature_trainable (optional):
            temperature_initializer (optional):
            temperature_regularizer (optional):
            temperature_constraint (optional):
            lr_attention_trainable (optional):
            lr_attention_initializer (optional):
            lr_attention_regularizer (optional):
            lr_attention_constraint (optional):
            lr_association_trainable (optional):
            lr_association_initializer (optional):
            lr_association_regularizer (optional):
            lr_association_constraint (optional):
            data_scope (optional): String indicating the behavioral
                data that should be used for the layer.

        """
        super(ALCOVECell, self).__init__(**kwargs)

        self.percept = percept
        self.similarity = similarity

        if data_scope is None:
            data_scope = "categorize"
        self.data_scope = data_scope

        # Configure percept adapter.
        if percept_adapter is None:
            # Default adapter has no gating keys.
            percept_adapter = GateAdapter(format_inputs_as_tuple=True)
        # NOTE: The second adapter uses the same underlying embedding and
        # gate weights, but takes a different set of indices.
        alcove_adapter = percept_adapter.__class__.from_config(
            percept_adapter.get_config()
        )
        self.percept_adapter = percept_adapter
        self._alcove_adapter = alcove_adapter

        # Set required input keys.
        self.percept_adapter.input_keys = [data_scope + "_alcove_stimset_samples"]
        self._alcove_adapter.input_keys = [data_scope + "_alcove_idx"]

        # Misc. attributes.
        self.units = units
        self.output_logits = False

        # Process incoming layers.
        self.n_dim = percept.output_dim
        n_rbf = percept.input_dim
        if self.percept.mask_zero:
            n_rbf = n_rbf - 1
        self.n_rbf = n_rbf
        self._n_rbf = keras.ops.convert_to_tensor(n_rbf)

        # Satisfy RNNCell contract.
        self.state_size = -1  # Custom definted in get_initial_state.
        self.output_size = units

        # Process self-owned weights.

        # Process `rho`.
        self.rho_trainable = self.trainable and rho_trainable
        if rho_initializer is None:
            rho_initializer = keras.initializers.RandomUniform(minval=1.0, maxval=2.0)
        self.rho_initializer = keras.initializers.get(rho_initializer)
        self.rho_regularizer = keras.regularizers.get(rho_regularizer)
        if rho_constraint is None:
            rho_constraint = pk_constraints.GreaterEqualThan(min_value=1.0)
        self.rho_constraint = keras.constraints.get(rho_constraint)

        # Process `temperature`.
        self.temperature_trainable = self.trainable and temperature_trainable
        if temperature_initializer is None:
            temperature_initializer = keras.initializers.RandomUniform(
                minval=1.0, maxval=5.0
            )
        self.temperature_initializer = keras.initializers.get(temperature_initializer)
        self.temperature_regularizer = keras.regularizers.get(temperature_regularizer)
        if temperature_constraint is None:
            temperature_constraint = pk_constraints.GreaterEqualThan(min_value=0.0)
        self.temperature_constraint = keras.constraints.get(temperature_constraint)

        # Process `lr_attention`.
        self.lr_attention_trainable = self.trainable and lr_attention_trainable
        if lr_attention_initializer is None:
            lr_attention_initializer = keras.initializers.RandomUniform(
                minval=0.0, maxval=0.1
            )
        self.lr_attention_initializer = keras.initializers.get(lr_attention_initializer)
        self.lr_attention_regularizer = keras.regularizers.get(lr_attention_regularizer)
        if lr_attention_constraint is None:
            lr_attention_constraint = pk_constraints.GreaterEqualThan(min_value=0.0)
        self.lr_attention_constraint = keras.constraints.get(lr_attention_constraint)

        # Process `lr_association`.
        self.lr_association_trainable = self.trainable and lr_association_trainable
        if lr_association_initializer is None:
            lr_association_initializer = keras.initializers.RandomUniform(
                minval=0.0, maxval=0.1
            )
        self.lr_association_initializer = keras.initializers.get(
            lr_association_initializer
        )
        self.lr_association_regularizer = keras.regularizers.get(
            lr_association_regularizer
        )
        if lr_association_constraint is None:
            lr_association_constraint = pk_constraints.GreaterEqualThan(min_value=0.0)
        self.lr_association_constraint = keras.constraints.get(
            lr_association_constraint
        )

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial state.

        Returns:
            initial_state: A list of tensors. The first Tensor
            represents "attention" weights. The second Tensor
            represents "association" weights. The attention weights
            include a sington dimension in anticipation of broadcasting
            operations in the `call` method.

        """
        initial_state = [
            keras.ops.ones([batch_size, 1, self.n_dim]) / self.n_dim,
            keras.ops.zeros([batch_size, self.n_rbf, self.units]),
        ]
        return initial_state

    def build(self, input_shape):
        """Build.

        Args:
            input_shape: Expects a dictionary that contains
                "categorize_stimulus_set" with shape =
                (batch_size, [1,]).

        """
        with keras.name_scope(self.name):
            self.rho = self.add_weight(
                shape=[],
                initializer=self.rho_initializer,
                regularizer=self.rho_regularizer,
                trainable=self.rho_trainable,
                name="rho",
                dtype=keras.backend.floatx(),
                constraint=self.rho_constraint,
            )
            self.temperature = self.add_weight(
                shape=[],
                initializer=self.temperature_initializer,
                regularizer=self.temperature_regularizer,
                trainable=self.temperature_trainable,
                name="temperature",
                dtype=keras.backend.floatx(),
                constraint=self.temperature_constraint,
            )
            self.lr_attention = self.add_weight(
                shape=[],
                initializer=self.lr_attention_initializer,
                regularizer=self.lr_attention_regularizer,
                trainable=self.lr_attention_trainable,
                name="lr_attention",
                dtype=keras.backend.floatx(),
                constraint=self.lr_attention_constraint,
            )
            self.lr_association = self.add_weight(
                shape=[],
                initializer=self.lr_association_initializer,
                regularizer=self.lr_association_regularizer,
                trainable=self.lr_association_trainable,
                name="lr_association",
                dtype=keras.backend.floatx(),
                constraint=self.lr_association_constraint,
            )

        # We assume axes semantics based on relative position from last axis.
        stimuli_axis = -1
        # Convert from *relative* axis index to *absolute* axis index.
        n_axis = len(input_shape[self.data_scope + "_stimulus_set"])
        self._stimuli_axis = n_axis + stimuli_axis

        # Precompute indices for all ALCOVE RBFs.
        self._alcove_idx = self._precompute_alcove_indices()

        super().build(input_shape)

    def call(self, inputs, states, training=None):
        """Call.

        Args:
            inputs["stimulus_set"]: The indices of the stimuli.
                shape=(batch_size, 1)
            inputs["objective_query_label"]: One-hot encoding of
                (objectively correct) query label.
                shape=(batch_size, n_output)
            inputs["gate_weights"] (optional): Tensor(s) containing gate
                weights. The actual key value(s) will depend on how
                the user initialized the layer.
            states[0]: A tensor representing batch-specific attention
                weights that modify the Minkowski distance.
            states[1]: A tensor representing batch-specific association
                weights that map RBF activity to class output activitiy.

        """
        # NOTE: The inputs are copied, because modifying the original `inputs`
        # is bad practice in TF. For example, it creates issues when saving
        # a model.
        inputs_copied = copy.copy(inputs)

        batch_size = keras.ops.shape(inputs_copied[self.data_scope + "_stimulus_set"])[
            0
        ]

        stimulus_set = inputs_copied[self.data_scope + "_stimulus_set"]
        objective_query_label_idx = inputs_copied[
            self.data_scope + "_objective_query_label"
        ]

        attention = states[0]  # Previous attention weights state.
        association = states[1]  # Previous association weights state.

        # Embed stimuli indices in n-dimensional space.
        inputs_copied.update(
            {
                self.data_scope + "_alcove_stimset_samples": stimulus_set,
                self.data_scope
                + "_alcove_idx": keras.ops.repeat(self._alcove_idx, batch_size, axis=0),
            }
        )
        inputs_percept = self.percept_adapter(inputs_copied)
        z_in = self.percept(inputs_percept)
        # TensorShape=(batch_size, 1, n_dim])

        # To compute RBF activations (i.e., similarity), start by retrieving
        # the ALCOVE RBF embeddings.
        inputs_alcove = self._alcove_adapter(inputs_copied)
        z_alcove = self.percept(inputs_alcove)
        # shape=(batch_size, n_ref, n_dim)

        # Use TensorFlow gradients to update model state.
        state_variables = [association, attention]
        if keras.backend.backend() == "tensorflow":
            x_out, grad_attention, grad_association = self._tensorflow_update(
                state_variables,
                z_in,
                z_alcove,
                objective_query_label_idx,
                attention,
                association,
            )
        else:
            raise NotImplementedError(
                f"Unrecognized backend {keras.backend.backend()}."
            )

        # Response rule.
        x_out_scaled = keras.ops.multiply(x_out, self.temperature)
        if not self.output_logits:
            x_out_scaled = keras.ops.softmax(x_out_scaled, axis=-1)

        # Update states.
        attention_tplus1 = attention - (self.lr_attention * grad_attention)
        attention_tplus1 = keras.ops.maximum(attention_tplus1, 0)
        association_tplus1 = association - (self.lr_association * grad_association)
        states_tplus1 = [attention_tplus1, association_tplus1]

        return x_out_scaled, states_tplus1

    def distance(self, inputs):
        """Compute distance"""
        # NOTE: Modified from `Minkowski` class `call` method..
        z_0 = inputs[0]
        z_1 = inputs[1]
        w = inputs[2]
        x = z_0 - z_1

        # Broadcast `rho` and `w` to appropriate shape.
        x_shape = keras.ops.shape(x)
        # Broadcast `rho` to shape=(batch_size, [n, m, ...]).
        rho = self.rho * keras.ops.ones(x_shape[0:-1])
        # Broadcast `w` to shape=(batch_size, [n, m, ...] n_dim).
        w = keras.ops.broadcast_to(w, x_shape)

        # Weighted Minkowski distance.
        d_qr = wpnorm(x, w, rho)
        return d_qr

    def humble_teacher_loss(self, y, y_pred):
        """Humble teacher loss as described in ALCOVE model.

        Args:
            y:
                shape=(batch_size, units)
            y_pred:
                shape=(batch_size, units)

        Returns:
            Humble teacher loss.

        """
        # Settings
        min_val = keras.ops.cast(-1.0, keras.backend.floatx())
        max_val = keras.ops.cast(1.0, keras.backend.floatx())

        y_teach_min = keras.ops.minimum(min_val, y_pred)
        y_teach_max = keras.ops.maximum(max_val, y_pred)

        # Zero out correct locations.
        y_teach = y_teach_min - keras.ops.multiply(y, y_teach_min)
        # Add in correct locations.
        y_teach = y_teach + keras.ops.multiply(y, y_teach_max)

        # Sum over outputs (last axis).
        loss = keras.ops.mean(keras.ops.square(y_teach - y_pred), axis=-1)

        return loss

    def get_config(self):
        """Return layer configuration."""
        config = super(ALCOVECell, self).get_config()
        config.update(
            {
                "units": self.units,
                "percept": keras.saving.serialize_keras_object(self.percept),
                "similarity": keras.saving.serialize_keras_object(self.similarity),
                "percept_adapter": keras.saving.serialize_keras_object(
                    self.percept_adapter
                ),
                "rho_initializer": keras.initializers.serialize(self.rho_initializer),
                "rho_regularizer": keras.regularizers.serialize(self.rho_regularizer),
                "rho_constraint": keras.constraints.serialize(self.rho_constraint),
                "rho_trainable": self.rho_trainable,
                "temperature_initializer": keras.initializers.serialize(
                    self.temperature_initializer
                ),
                "temperature_regularizer": keras.regularizers.serialize(
                    self.temperature_regularizer
                ),
                "temperature_constraint": keras.constraints.serialize(
                    self.temperature_constraint
                ),
                "temperature_trainable": self.temperature_trainable,
                "lr_attention_initializer": keras.initializers.serialize(
                    self.lr_attention_initializer
                ),
                "lr_attention_regularizer": keras.regularizers.serialize(
                    self.lr_attention_regularizer
                ),
                "lr_attention_constraint": keras.constraints.serialize(
                    self.lr_attention_constraint
                ),
                "lr_attention_trainable": self.lr_attention_trainable,
                "lr_association_initializer": keras.initializers.serialize(
                    self.lr_association_initializer
                ),
                "lr_association_regularizer": keras.regularizers.serialize(
                    self.lr_association_regularizer
                ),
                "lr_association_constraint": keras.constraints.serialize(
                    self.lr_association_constraint
                ),
                "lr_association_trainable": self.lr_association_trainable,
                "data_scope": self.data_scope,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["similarity"] = keras.layers.deserialize(config["similarity"])
        config["percept"] = keras.layers.deserialize(config["percept"])
        config["percept_adapter"] = keras.layers.deserialize(config["percept_adapter"])
        return cls(**config)

    def _precompute_alcove_indices(self):
        """Precompue indices for all ALCOVE RBFs."""
        alcove_idx = np.arange(self._n_rbf)
        if self.percept.mask_zero:
            alcove_idx = alcove_idx + 1

        # Add "batch" axis.
        # NOTE: Will repeat by actual `batch_size` in `call` method. If you
        # repeat in `build`, raises an error during save because batch axis
        # is `None`.
        alcove_idx = np.expand_dims(alcove_idx, axis=0)
        # shape=(1, n_ref)
        return alcove_idx

    def _tensorflow_update(
        self,
        state_variables,
        z_in,
        z_alcove,
        objective_query_label_idx,
        attention,
        association,
    ):
        state_tape = tf.GradientTape(watch_accessed_variables=False, persistent=True)
        with state_tape:
            state_tape.watch(state_variables)

            # Compute similarity.
            d_qr = self.distance([z_in, z_alcove, attention])
            s_qr = self.similarity(d_qr)
            # Add trailing singleton axis for `units` broadcast compatability.
            s_qr = keras.ops.expand_dims(s_qr, axis=-1)
            # shape=(batch_size, n_rbf, 1)

            # Compute output activations via association weights.
            x_out = keras.ops.multiply(s_qr, association)
            x_out = keras.ops.sum(x_out, axis=self._stimuli_axis)
            # shape=(batch_size, units)

            # Compute loss.
            loss = self.humble_teacher_loss(objective_query_label_idx, x_out)
            # shape=(batch_size, [n_sample])
        # Compute derivatives
        grad_attention = state_tape.gradient(loss, attention)
        grad_association = state_tape.gradient(loss, association)
        del state_tape

        return x_out, grad_attention, grad_association
