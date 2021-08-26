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
"""Gamma distribution class."""

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util


class Gamma(gamma.Gamma):
    """Gamma distribution with corrected mode."""

    def _mode(self):
        concentration = tf.convert_to_tensor(self.concentration)
        rate = tf.convert_to_tensor(self.rate)
        mode = (concentration - 1.) / rate
        if self.allow_nan_stats:
            assertions = []
        else:
            assertions = [assert_util.assert_less(
                tf.ones([], self.dtype), concentration,
                message='Mode not defined when any concentration < 1.')
            ]
        with tf.control_dependencies(assertions):
            return tf.where(
                concentration >= 1.,
                mode,
                dtype_util.as_numpy_dtype(self.dtype)(np.nan)
            )

    def _covariance(self, **kwargs):
        raise NotImplementedError

    def _survival_function(self, value, **kwargs):
        raise NotImplementedError

    def _log_survival_function(self, value, **kwargs):
        raise NotImplementedError

    def _quantile(self, value, **kwargs):
        raise NotImplementedError
