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
"""Module for a TensorFlow GroupEmbedding.

Classes:
    EmbeddingGroup: TODO

"""

import tensorflow as tf

from psiz.keras.layers.embeddings.nd import EmbeddingND


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingGroup'
)
class EmbeddingGroup(EmbeddingND):
    """TODO"""
    def __init__(self, **kwargs):
        """TODO"""
        super(EmbeddingGroup, self).__init__(**kwargs)

    def call(self, inputs):
        idx_stimuli = inputs[0]
        idx_group = inputs[-1]  # TODO group_level
        idx_group = tf.broadcast_to(idx_group, tf.shape(idx_stimuli))
        multi_index = tf.stack([idx_stimuli, idx_group], axis=0)
        return self.embedding(multi_index)
