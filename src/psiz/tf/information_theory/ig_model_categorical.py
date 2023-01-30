# -*- coding: utf-8 -*-
# Copyright 2020 Brett D. Roads. All Rights Reserved.
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
    ig_model_categorical: Compute expected information gain for an
        ensemble of models making categorical predictions.

"""

import copy

import tensorflow as tf

from psiz.tf.information_theory import ig_categorical


def ig_model_categorical(model_list, inputs, n_sample):
    """Ensemble information gain.

    Args:
        model_list: A list of models. All models are treated as part of
            an ensemble.
        inputs: A batch of inputs. The exact length and shape
            depends on the model being used, however it is assumed that
            the first dimension of all inputs has `batch_size`
            semantics.
        n_sample: Integer indicating number of samples.

    Returns:
        Information gain for each input in the batch.
            shape=(batch_size,)

    NOTE: Information gain is computed by concatenating samples from
    all models and then computing expected information gain. Since
    embedding algorithms exhibit sensitivity to intial conditions
    and decent stickiness, the posterior from a single model does not
    completely capture their uncertainty. Combining samples from
    different models is a better estimate of the "total" uncertainty.

    """
    sample_axis = 1
    inputs_copied = copy.copy(inputs)
    inputs_copied = model_list[0].repeat_samples_in_batch_axis(inputs_copied, n_sample)

    output_predictions = []
    for model in model_list:
        outputs = model(inputs_copied, training=False)
        outputs = model.disentangle_repeated_samples(outputs, n_sample)
        output_predictions.append(outputs)
    # Concatenate different ensemble predictions along samples axis.
    output_predictions = tf.concat(output_predictions, sample_axis)
    return ig_categorical(output_predictions)

    # For comparision, here is an alternative computation where IG is computed
    # for each model separately and then averaged.
    # ig = []
    # for model in model_list:
    #     outputs = model(inputs_copied, training=False)
    #     outputs = model.disentangle_repeated_samples(outputs, n_sample)
    #     ig.append(ig_categorical(outputs))
    # # Concatenate different ensemble predictions along samples axis.
    # ig = tf.stack(ig, 0)
    # ig = tf.reduce_mean(ig, 0)
    # return ig
