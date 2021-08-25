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
"""Module of custom losses.

Classes:
    NegLogLikelihood: Negative log-likelihood.

"""

import tensorflow as tf
from tensorflow.keras import backend as K


class NegLogLikelihood(tf.keras.losses.Loss):
    """Negative log-likelihood loss."""

    def call(self, y_true, y_pred):
        """Call."""
        # pylint: disable=unused-argument
        return _safe_neg_log_prob(y_pred)


def _safe_neg_log_prob(prob):
    """Safely convert to log probabilites.

    Arguments:
        prob: Probabilities to convert.

    Returns:
        log(prob)

    """
    cap = tf.constant(2.2204e-16, dtype=K.floatx())
    return tf.negative(tf.math.log(tf.maximum(prob, cap)))
