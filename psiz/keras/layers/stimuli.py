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
    Stimuli: An embedding layer with (group-specific) stimuli
        semantics.

"""

@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='Stimuli'
)
class Stimuli(tf.keras.layers.Layer):
    """An embedding that handles group-specific embeddings."""
    def __init__(
            self, n_group=1, group_level=0, embedding=None, **kwargs):
        """Initialize.
        
        Arguments:
            n_group: Integer indicating the number of groups in the
                embedding.
            group_level (optional): Ingeter indicating the group level
                of the embedding. This will determine which column of
                `group` is used to route the forward pass.
            embedding: An Embedding layer.
            kwargs: Additional key-word arguments.
 
        """
        # Check that n_group is compatible with provided embedding layer.
        input_dim = embedding.input_dim
        input_dim_group = input_dim / n_group
        if not input_dim_group.is_integer():
            raise ValueError(
                'The provided `n_group`={0} is not compatible with the'
                ' provided embedding. The provided embedding has'
                ' input_dim={1}, which cannot be cleanly reshaped to'
                ' (n_group, input_dim/n_group).'.format(n_group, input_dim)
            )

        super(Stimuli, self).__init__(**kwargs)
        self.input_dim = int(input_dim_group)
        self.n_group = n_group
        self.group_level = group_level
        self._embedding = embedding

    def build(self, input_shape):
        """Build."""
        super().build(input_shape)
        self._embedding.build(input_shape)

    def call(self, inputs):
        """Call."""
        # Route indices by group membership.
        indices = inputs[0]
        group_id = inputs[-1][:, self.group_level]
        group_id = tf.expand_dims(group_id, axis=-1)
        group_id = tf.expand_dims(group_id, axis=-1)
        indices_flat = _map_embedding_indices(
            indices, group_id, self.input_dim, False
        )
        # Make usual call to embedding layer.
        return self._embedding(indices_flat)

    def get_config(self):
        """Return configuration."""
        config = super(Stimuli, self).get_config()
        config.update({
            'n_group': int(self.n_group),
            'group_level': int(self.group_level),
            'embedding': tf.keras.utils.serialize_keras_object(
                self._embedding
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
        z_flat = self._embedding.embeddings
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
        return self._embedding.output_dim

    @property
    def mask_zero(self):
        return self._embedding.mask_zero

    @property
    def n_stimuli(self):
        n_stimuli = self.input_dim
        if self._embedding.mask_zero:
            n_stimuli -= 1
        return n_stimuli
