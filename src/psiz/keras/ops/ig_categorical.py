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
"""TensorFlow module.

Functions:
    ig_categorical: A sample-based function for computing expected
        information gain of events with categorical outcomes.

"""


import keras


def ig_categorical(y_pred):
    """Return information gain of events with categorical outcomes.

    A sample-based approximation of information gain is determined by
    computing the mutual information between the candidate event(s)
    and an existing set of observed events (implied by the current
    model state).

    This function is designed to be agnostic to the manner in which
    `y_pred` samples are drawn. For example, these could be dervied
    using MCMC or by sampling output predictions from a model fit using
    variational inference.

    NOTE: This function works with placeholder elements as long as
    `y_pred` is zero for those elements.

    Args:
        y_pred: A tensor of a model's categorical outcome predictions.
            shape=(n_event, n_sample, n_outcome)

    Returns:
        A tensor representing the expected information gain of the candidate event(s).
            shape=(n_event,)

    """
    # First term of mutual information.
    # H(Y | obs, c) = - sum P(y_i | obs, c) log P(y_i | obs, c),
    # where `c` indicates a candidate event that we want to compute the
    # expected information gain for.
    # Take mean over samples to approximate p(y_i | obs, c).
    term0 = keras.ops.mean(y_pred, axis=1)  # shape=(n_event, n_outcome)
    term0 = term0 * keras.ops.log(keras.ops.maximum(term0, keras.backend.epsilon()))
    # NOTE: At this point we would need to zero out place-holder outcomes,
    # but placeholder elements will always have a value of zero since
    # y_pred will be zero for placeholder elements.
    # Sum over possible outcomes.
    term0 = -keras.ops.sum(term0, axis=1)  # shape=(n_event,)

    # Second term of mutual information.
    # E[H(Y | Z, D, x)]
    term1 = y_pred * keras.ops.log(keras.ops.maximum(y_pred, keras.backend.epsilon()))
    # Take the sum over the possible outcomes.
    # NOTE: At this point we would need to zero out place-holder outcomes,
    # but placeholder elements will always have a value of zero since
    # y_pred will be zero for placeholder elements.
    term1 = keras.ops.sum(term1, axis=2)  # shape=(n_event, n_sample)
    # Take the mean over all samples.
    term1 = keras.ops.mean(term1, axis=1)  # shape=(n_event,)

    return term0 + term1
