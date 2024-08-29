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
"""Module of custom TensorFlow constraints.

Classes:
    LessEqualThan:

"""


import keras


@keras.saving.register_keras_serializable(package="psiz.keras.constraints")
class LessEqualThan(keras.constraints.Constraint):
    """Constrains the weights to be greater/equal than a value."""

    def __init__(self, max_value=0.0):
        """Initialize.

        Args:
            max_value: The maximum allowed weight value.

        """
        self.max_value = max_value

    def __call__(self, w):
        """Call."""
        w = w - self.max_value
        w = w * keras.ops.cast(keras.ops.greater_equal(0.0, w), keras.backend.floatx())
        w = w + self.max_value
        return w

    def get_config(self):
        """Return configuration."""
        return {"max_value": self.max_value}
