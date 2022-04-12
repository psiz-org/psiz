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
"""Module for testing models.py."""


# import tensorflow as tf

# import psiz


# def test_init():
#     """Test initialization."""
#     n_stimuli = 30
#     n_dim = 10

#     stimuli = tf.keras.layers.Embedding(
#         n_stimuli + 1, n_dim, mask_zero=True
#     )

#     kernel = psiz.keras.layers.DistanceBased(
#         distance=psiz.keras.layers.Minkowski(
#             rho_initializer=tf.keras.initializers.Constant(2.),
#             w_initializer=tf.keras.initializers.Constant(1.),
#             trainable=False
#         ),
#         similarity=psiz.keras.layers.ExponentialSimilarity(
#             trainable=False,
#             beta_initializer=tf.keras.initializers.Constant(3.),
#             tau_initializer=tf.keras.initializers.Constant(1.),
#             gamma_initializer=tf.keras.initializers.Constant(0.),
#         )
#     )

#     _ = psiz.keras.models.Sort(
#         stimuli=stimuli, kernel=kernel
#     )
