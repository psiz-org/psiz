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
    GateMixin: A multiple inheritance mixin for layers that support
        groups.

"""

import tensorflow as tf


class GateMixin():
    """A mixin for modules that support gating of input."""

    def __init__(self, *args, **kwargs):
        """Initialize.

        Args:

        """
        super().__init__(*args, **kwargs)
        self.supports_gating = tf.constant(True)
        self.inputs_gate_idx = -1
        # Create placeholder for layer switches.
        self._pass_gate_weights = {}

    def check_supports_gating(self, layer):
        """Check if layer supports groups."""
        # Check if implements `GateMixin`.
        if hasattr(layer, 'supports_gating'):
            return layer.supports_gating

        # Check RNN cell.
        elif isinstance(layer, tf.keras.layers.RNN):
            return self.check_supports_gating(layer.cell)
        else:
            return tf.constant(False)
