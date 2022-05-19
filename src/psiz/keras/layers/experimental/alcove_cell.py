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
"""Module for a TensorFlow layers.

Classes:
    ALCOVECell: An RNN-compatible cell implementing the ALCOVE Model.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints
from psiz.keras.layers.behaviors.behavior import Behavior
from psiz.tf.ops.wpnorm import wpnorm
from psiz.utils import expand_dim_repeat


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras', name='ALCOVECell'
)
class ALCOVECell(Behavior):
    """An RNN-compatible cell implementing ALCOVE."""

    def __init__(
            self, units=None, embedding=None, similarity=None,
            rho_trainable=True, rho_initializer=None,
            rho_regularizer=None, rho_constraint=None,
            temperature_trainable=True, temperature_initializer=None,
            temperature_regularizer=None, temperature_constraint=None,
            lr_attention_trainable=True, lr_attention_initializer=None,
            lr_attention_regularizer=None, lr_attention_constraint=None,
            lr_association_trainable=True, lr_association_initializer=None,
            lr_association_regularizer=None, lr_association_constraint=None,
            **kwargs):
        """Initialize.

        Args:
            units: Positive integer indicating the number of output classes.
            embedding: An embedding layer.
            similarity: A similarity layer.
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

        """
        super(ALCOVECell, self).__init__(**kwargs)

        # Satisfy `GroupsMixin` contract.
        self.supports_groups = False
        self._pass_groups['similarity'] = self.check_supports_groups(
            similarity
        )

        # Misc. attributes.
        self.units = units
        self.output_logits = False
        self.n_sample = 1

        # Process incoming layers.
        self.similarity = similarity
        self.embedding = embedding
        self.n_dim = embedding.output_dim
        n_rbf = embedding.input_dim
        if self.embedding.mask_zero:
            n_rbf = n_rbf - 1
        self.n_rbf = n_rbf
        self._n_rbf = tf.constant(n_rbf)

        # Satisfy RNNCell contract.
        self.state_size = [
            tf.TensorShape([self.n_dim]),
            tf.TensorShape([self.n_rbf, units])
        ]
        self.output_size = tf.TensorShape([units])

        # Process self-owned weights.

        # Process `rho`.
        self.rho_trainable = self.trainable and rho_trainable
        if rho_initializer is None:
            rho_initializer = tf.random_uniform_initializer(1., 2.)
        self.rho_initializer = tf.keras.initializers.get(rho_initializer)
        self.rho_regularizer = tf.keras.regularizers.get(rho_regularizer)
        if rho_constraint is None:
            rho_constraint = pk_constraints.GreaterEqualThan(min_value=1.0)
        self.rho_constraint = tf.keras.constraints.get(rho_constraint)
        with tf.name_scope(self.name):
            self.rho = self.add_weight(
                shape=[], initializer=self.rho_initializer,
                regularizer=self.rho_regularizer, trainable=self.rho_trainable,
                name="rho", dtype=K.floatx(),
                constraint=self.rho_constraint
            )

        # Process `temperature`.
        self.temperature_trainable = self.trainable and temperature_trainable
        if temperature_initializer is None:
            temperature_initializer = tf.random_uniform_initializer(1., 5.)
        self.temperature_initializer = tf.keras.initializers.get(
            temperature_initializer
        )
        self.temperature_regularizer = tf.keras.regularizers.get(
            temperature_regularizer
        )
        if temperature_constraint is None:
            temperature_constraint = pk_constraints.GreaterEqualThan(
                min_value=0.0
            )
        self.temperature_constraint = tf.keras.constraints.get(
            temperature_constraint
        )
        with tf.name_scope(self.name):
            self.temperature = self.add_weight(
                shape=[], initializer=self.temperature_initializer,
                regularizer=self.temperature_regularizer,
                trainable=self.temperature_trainable, name="temperature",
                dtype=K.floatx(), constraint=self.temperature_constraint
            )

        # Process `lr_attention`.
        self.lr_attention_trainable = self.trainable and lr_attention_trainable
        if lr_attention_initializer is None:
            lr_attention_initializer = tf.random_uniform_initializer(0.0, 0.1)
        self.lr_attention_initializer = tf.keras.initializers.get(
            lr_attention_initializer
        )
        self.lr_attention_regularizer = tf.keras.regularizers.get(
            lr_attention_regularizer
        )
        if lr_attention_constraint is None:
            lr_attention_constraint = pk_constraints.GreaterEqualThan(
                min_value=0.0
            )
        self.lr_attention_constraint = tf.keras.constraints.get(
            lr_attention_constraint
        )
        with tf.name_scope(self.name):
            self.lr_attention = self.add_weight(
                shape=[], initializer=self.lr_attention_initializer,
                regularizer=self.lr_attention_regularizer,
                trainable=self.lr_attention_trainable, name="lr_attention",
                dtype=K.floatx(), constraint=self.lr_attention_constraint
            )

        # Process `lr_association`.
        self.lr_association_trainable = (
            self.trainable and lr_association_trainable
        )
        if lr_association_initializer is None:
            lr_association_initializer = tf.random_uniform_initializer(
                0.0, 0.1
            )
        self.lr_association_initializer = tf.keras.initializers.get(
            lr_association_initializer
        )
        self.lr_association_regularizer = tf.keras.regularizers.get(
            lr_association_regularizer
        )
        if lr_association_constraint is None:
            lr_association_constraint = pk_constraints.GreaterEqualThan(
                min_value=0.0
            )
        self.lr_association_constraint = tf.keras.constraints.get(
            lr_association_constraint
        )
        with tf.name_scope(self.name):
            self.lr_association = self.add_weight(
                shape=[], initializer=self.lr_association_initializer,
                regularizer=self.lr_association_regularizer,
                trainable=self.lr_association_trainable, name="lr_association",
                dtype=K.floatx(), constraint=self.lr_association_constraint
            )

    def call(self, inputs, states):
        """Call.

        Args:
            inputs: A tuple of Tensors. Each Tensor is assumed to have
            leading batch and sample axes.
                inputs[1]: Stimulus embedding.
                inputs[2]: Correct label idx of stimulus.
                shape=(batch_size, n_sample, [n, ...])
            states:
                states[0]: A tensor representing batch-specific
                    attention weights that modify the Minkowski
                    distance.
                states[1]: A tensor representing batch-specific
                    association weights that map RBF activity to
                    class output activitiy.

        """
        z_in = inputs[1]
        correct_label_idx = inputs[2]
        batch_size = tf.shape(inputs[1])[0]
        attention = states[0]  # Previous attention weights state.
        association = states[1]  # Previous association weights state.

        # Compute RBF activations.
        # Add singleton for "query" axis.
        z_in = tf.expand_dims(z_in, axis=2)
        # shape=(batch_size, n_sample, 1, n_dim)

        # Retrive ALCOVE RBF embeddings.
        alcove_idx = tf.range(self._n_rbf)
        if self.embedding.mask_zero:
            alcove_idx = alcove_idx + 1
        z_alcove = self.embedding(alcove_idx)
        # Add "sample" axis.
        z_alcove = expand_dim_repeat(
            z_alcove, self.n_sample, axis=0
        )
        # Add "batch" axis.
        z_alcove = expand_dim_repeat(
            z_alcove, batch_size, axis=0
        )
        # shape=(batch_size, n_sample, n_ref, n_dim)

        # Use TensorFlow gradients to update model state.
        state_variables = [association, attention]
        state_tape = tf.GradientTape(
            watch_accessed_variables=False, persistent=True
        )
        with state_tape:
            state_tape.watch(state_variables)

            # Adjust attention dimensions for sample and timestep axis.
            attention = tf.expand_dims(attention, axis=1)
            attention = tf.expand_dims(attention, axis=2)

            # Compute similarity.
            # s_qr = self.kernel([z_in, z_alcove])
            d_qr = self.distance([z_in, z_alcove, attention])
            s_qr = self.similarity(d_qr)
            # Add singleton axis for `units` broadcast compatability.
            s_qr = tf.expand_dims(s_qr, axis=3)
            # shape=(batch_size, n_sample, n_rbf, 1)

            # Compute output activations via association weights.
            x_out = tf.multiply(s_qr, tf.expand_dims(association, axis=1))
            x_out = tf.reduce_sum(x_out, axis=2)
            # shape=(batch_size, n_sample, units)

            # Compute loss.
            correct_label_onehot = tf.one_hot(
                correct_label_idx, self.units, on_value=1.0, off_value=0.0
            )
            loss = self.humble_teacher_loss(correct_label_onehot, x_out)
            # shape=(batch_size, n_sample)
        # Compute derivatives
        grad_attention = state_tape.gradient(loss, attention)
        grad_association = state_tape.gradient(loss, association)
        del state_tape

        # Undo extra attention dimensions.
        attention = tf.squeeze(attention, [1, 2])
        grad_attention = tf.reduce_mean(grad_attention, axis=1)
        grad_attention = tf.reduce_mean(grad_attention, axis=1)

        # Response rule.
        x_out_scaled = tf.multiply(x_out, self.temperature)
        if not self.output_logits:
            x_out_scaled = tf.nn.softmax(x_out_scaled, axis=2)

        # Update states.
        attention_tplus1 = attention - (self.lr_attention * grad_attention)
        attention_tplus1 = tf.math.maximum(attention_tplus1, 0)
        association_tplus1 = association - (
            self.lr_association * grad_association
        )
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
        x_shape = tf.shape(x)
        # Broadcast `rho` to shape=(batch_size, [n, m, ...]).
        rho = self.rho * tf.ones(x_shape[0:-1])
        # Broadcast `w` to shape=(batch_size, [n, m, ...] n_dim).
        w = tf.broadcast_to(w, x_shape)

        # Weighted Minkowski distance.
        d_qr = wpnorm(x, w, rho)
        d_qr = tf.squeeze(d_qr, [-1])
        return d_qr

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial state."""
        initial_state = [
            tf.ones([batch_size, self.n_dim]) / self.n_dim,
            tf.zeros([batch_size, self.n_rbf, self.units])
        ]
        return initial_state

    def humble_teacher_loss(self, y, y_pred):
        """Humble teacher loss as described in ALCOVE model.

        Args:
            y:
                shape=(batch_size, n_sample, units)
            y_pred:
                shape=(batch_size, n_sample, units)

        Returns:
            Humble teacher loss.

        """
        # Settings
        min_val = tf.cast(-1.0, K.floatx())
        max_val = tf.cast(1.0, K.floatx())

        y_teach_min = tf.math.minimum(min_val, y_pred)
        y_teach_max = tf.math.maximum(max_val, y_pred)

        # Zero out correct locations.
        y_teach = y_teach_min - tf.multiply(y, y_teach_min)
        # Add in correct locations.
        y_teach = y_teach + tf.multiply(y, y_teach_max)

        # Sum over outputs.
        loss = tf.reduce_mean(tf.square(y_teach - y_pred), axis=2)

        return loss

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'similarity': tf.keras.utils.serialize_keras_object(
                self.similarity
            ),
            'embedding': tf.keras.utils.serialize_keras_object(self.embedding),
            'rho_initializer':
                tf.keras.initializers.serialize(self.rho_initializer),
            'rho_regularizer':
                tf.keras.regularizers.serialize(self.rho_regularizer),
            'rho_constraint':
                tf.keras.constraints.serialize(self.rho_constraint),
            'rho_trainable': self.rho_trainable,
            'temperature_initializer':
                tf.keras.initializers.serialize(self.temperature_initializer),
            'temperature_regularizer':
                tf.keras.regularizers.serialize(self.temperature_regularizer),
            'temperature_constraint':
                tf.keras.constraints.serialize(self.temperature_constraint),
            'temperature_trainable': self.temperature_trainable,
            'lr_attention_initializer':
                tf.keras.initializers.serialize(self.lr_attention_initializer),
            'lr_attention_regularizer':
                tf.keras.regularizers.serialize(self.lr_attention_regularizer),
            'lr_attention_constraint':
                tf.keras.constraints.serialize(self.lr_attention_constraint),
            'lr_attention_trainable': self.lr_attention_trainable,
            'lr_association_initializer':
                tf.keras.initializers.serialize(
                    self.lr_association_initializer
                ),
            'lr_association_regularizer':
                tf.keras.regularizers.serialize(
                    self.lr_association_regularizer
                ),
            'lr_association_constraint':
                tf.keras.constraints.serialize(self.lr_association_constraint),
            'lr_association_trainable': self.lr_association_trainable,
        })
        return config

    @classmethod
    def from_config(cls, config):
        similarity_serial = config['similarity']
        config['similarity'] = tf.keras.layers.deserialize(similarity_serial)
        embedding_serial = config['embedding']
        config['embedding'] = tf.keras.layers.deserialize(embedding_serial)
        return super().from_config(config)
