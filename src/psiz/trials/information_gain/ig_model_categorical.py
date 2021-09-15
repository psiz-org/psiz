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
"""Information gain.

Functions:
    ig_model_categorical: Compute expected information gain for an
        ensemble of models making categorical predictions.

"""

import tensorflow as tf

from psiz.trials.information_gain import ig_categorical


def ig_model_categorical(model_list, inputs):
    """Ensemble information gain.

    Arguments:
        model_list: A list of models. All models are treated as part of
            an ensemble.
        inputs: A batch of inputs. The exact length and shape
            depends on the model being used, however it is assumed that
            the first dimension of all inputs has `batch_size`
            semantics.

    Returns:
        Information gain for each input in the batch.
            shape=(batch_size,)

    NOTE: Information gain is computed by concatenating samples from
    all models and then computing expected information gain.

    """
    output_predictions = []
    for model in model_list:
        output_predictions.append(model(inputs, training=False))
    # Concatenate different ensemble predictions along samples axis.
    output_predictions = tf.concat(output_predictions, 1)
    return ig_categorical(output_predictions)


# TODO
# def ig_ensemble_categorical(model_list, ds_docket, verbose=0):
#     """Ensemble information gain."""
#     if verbose > 0:
#         n_batch = 0
#         for _ in ds_docket:
#             n_batch += 1
#         progbar = ProgressBarRe(
#             np.ceil(n_batch), prefix='expected IG:', length=50
#         )
#         progbar.update(0)

#     expected_ig = []
#     batch_counter = 0
#     for x in ds_docket:
#         # IG computed on ensemble samples collectively.
#         batch_pred = []
#         for model in model_list:
#             batch_pred.append(model(x, training=False))
#         # Concatenate different ensemble predictions along samples axis.
#         batch_pred = tf.concat(batch_pred, 1)
#         # NOTE: Information gain is computed by concatenating samples from all
#         # models and then computing expected information gain.
#         batch_expected_ig = ig_categorical(batch_pred)
#         expected_ig.append(batch_expected_ig)
#         if verbose > 0:
#             progbar.update(batch_counter + 1)
#         batch_counter += 1
#     expected_ig = tf.concat(expected_ig, 0)
#     return expected_ig
