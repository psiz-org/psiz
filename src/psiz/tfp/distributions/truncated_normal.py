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
"""TruncatedNormal distribution class.

Note: Needed quantile method.

"""

import tensorflow as tf
from tensorflow_probability.python.distributions import truncated_normal
from tensorflow_probability.python.internal import special_math


# pylint: disable=abstract-method
class TruncatedNormal(truncated_normal.TruncatedNormal):
    """Truncated Normal distribution with quantile."""

    def _quantile(self, p):
        """See https://www.ntrand.com/truncated-normal-distribution/"""
        # pylint: disable=arguments-renamed,arguments-differ
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        delta = special_math.ndtr(b) - special_math.ndtr(a)
        x = delta * p + special_math.ndtr(a)
        return tf.math.ndtri(x) * self.scale + self.loc
