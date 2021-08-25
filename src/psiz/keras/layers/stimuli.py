# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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
"""Module for a TensorFlow Stimuli layer.

Classes:
    Stimuli: DEPRECATED An embedding layer with (group-specific) stimuli
        semantics.

"""

import tensorflow as tf
import tensorflow_probability as tfp

from psiz.keras.layers.group_level import GroupLevel


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='Stimuli'
)
class Stimuli(GroupLevel):
    """An embedding that handles group-specific embeddings."""
    def __init__(
            self, embedding=None, **kwargs):
        """Initialize.

        Arguments:
            embedding: An Embedding layer.
            kwargs: Additional key-word arguments.

        """
        super(Stimuli, self).__init__(**kwargs)

        # Check that n_group is compatible with provided embedding layer.
        input_dim = embedding.input_dim
        input_dim_group = input_dim / self.n_group
        if not input_dim_group.is_integer():
            raise ValueError(
                'The provided `n_group`={0} is not compatible with the'
                ' provided embedding. The provided embedding has'
                ' input_dim={1}, which cannot be cleanly reshaped to'
                ' (n_group, input_dim/n_group).'.format(
                    self.n_group, input_dim
                )
            )

        self.input_dim = int(input_dim_group)
        self.embedding = embedding

    def build(self, input_shape):
        """Build."""
        self.embedding.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        """Call."""
        # Route indices by group membership.
        indices = inputs[0]
        groups = inputs[-1][:, self.group_level]
        indices_flat = self._map_embedding_indices(
            indices, groups, self.input_dim
        )
        # Make usual call to embedding layer.
        return self.embedding(indices_flat)

    def get_config(self):
        """Return configuration."""
        config = super(Stimuli, self).get_config()
        config.update({
            'embedding': tf.keras.utils.serialize_keras_object(
                self.embedding
            )
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration.

        This method is the reverse of `get_config`, capable of
        instantiating the same layer from the config dictionary.

        Args:
            config: A Python dictionary, typically the output of
                `get_config`.

        Returns:
            layer: A layer instance.

        """
        config['embedding'] = tf.keras.layers.deserialize(
            config['embedding']
        )
        return cls(**config)

    @property
    def embeddings(self):
        """Getter method for `embeddings`.

        Returns:
            An embedding (tf.Tensor or tfp.Distribution) that reshapes
            source embedding appropriately.

        """
        z_flat = self.embedding.embeddings
        if isinstance(z_flat, tfp.distributions.Distribution):
            # Assumes Independent distribution.
            z = tfp.distributions.BatchReshape(
                z_flat.distribution,
                [self.n_group, self.input_dim, self.output_dim]
            )
            batch_ndims = tf.size(z.batch_shape_tensor())
            z = tfp.distributions.Independent(
                z, reinterpreted_batch_ndims=batch_ndims
            )
        else:
            z = tf.reshape(
                z_flat, [self.n_group, self.input_dim, self.output_dim]
            )
        return z

    @property
    def output_dim(self):
        return self.embedding.output_dim

    @property
    def mask_zero(self):
        return self.embedding.mask_zero

    @property
    def n_stimuli(self):
        n_stimuli = self.input_dim
        if self.embedding.mask_zero:
            n_stimuli -= 1
        return n_stimuli

    def _map_embedding_indices(self, idx, group_idx, n):
        """Map group-specific embedding indices to flat indices.

        Arguments:
            idx: Integer tf.Tensor indicating embedding indices.
                shape=(batch_size, [m, n, ...])
            group_idx: Integer tf.Tensor indicating group identifiers.
                This formulation assumes groups are consecutive,
                zero-based indices.
                shape=(batch_size)
            n: Integer indicating the number of embedding points.

        Returns:
            An integer tf.Tensor with the mapped indices.

        """
        # Make `group_idx` broadcast compatible.
        batch_size = tf.shape(group_idx)
        diff_rank = tf.rank(idx) - tf.rank(group_idx)
        shape_new = tf.ones(diff_rank, dtype=batch_size.dtype)
        shape_new = tf.concat((batch_size, shape_new), 0)
        group_idx = tf.reshape(group_idx, shape_new)

        # Perform mapping.
        return idx + (group_idx * n)
