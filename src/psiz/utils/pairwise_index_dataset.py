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
"""Module of utility functions.

Functions:
    pairwise_index_dataset: Generate a TF dataset of pairwise indices.

"""

import numpy as np
import tensorflow as tf

from psiz.utils.generate_group_matrix import generate_group_matrix


def pairwise_index_dataset(
        n_data, batch_size=None, elements='upper', groups=None,
        mask_zero=False, subsample=None, seed=252):
    """Assemble pairwise combinations.

    Arguments:
        n_data: A scalar indicating the number of unique concepts.
        batch_size (optional): Determines the batch size. By default,
            the batch size will be set to the number of combinations.
        elements (optional): Determines which combinations in the pairwise
            matrix will be used. Can be one of 'all', 'upper', 'lower',
            or 'off'.
        groups (optional): Array-like integers indicating group
            membership information. For example, `[4,3]`
            indicates that the first optional column has a value of 4
            and the second optional column has a value of 3.
        mask_zero (optional): Whether the zero index should be masked
            and all indices incremented by one.
        subsample: A float ]0,1] indicating the proportion of all pairs
            that should be retained. By default all pairs are retained.
        seed: Integer controlling which pairs are subsampled.

    Returns:
        ds: A Tensorflow Dataset.
        ds_info: A convenience dictionary.

    """
    if elements == 'all':
        idx = np.meshgrid(np.arange(n_data), np.arange(n_data))
        idx_0 = idx[0].flatten()
        idx_1 = idx[1].flatten()
    elif elements == 'upper':
        idx = np.triu_indices(n_data, 1)
        idx_0 = idx[0]
        idx_1 = idx[1]
    elif elements == 'lower':
        idx = np.tril_indices(n_data, -1)
        idx_0 = idx[0]
        idx_1 = idx[1]
    elif elements == 'off':
        idx_upper = np.triu_indices(n_data, 1)
        idx_lower = np.tril_indices(n_data, -1)
        idx = (
            np.hstack((idx_upper[0], idx_lower[0])),
            np.hstack((idx_upper[1], idx_lower[1])),
        )
        idx_0 = idx[0]
        idx_1 = idx[1]
    del idx

    n_pair = len(idx_0)
    if subsample is not None:
        # Make sure subsample is valid subsample value.
        subsample = np.minimum(1., subsample)
        subsample = np.maximum(0., subsample)

        np.random.seed(seed)
        idx_rand = np.random.permutation(n_pair)
        n_pair = int(np.ceil(n_pair * subsample))
        idx_rand = idx_rand[0:n_pair]
        idx_0 = idx_0[idx_rand]
        idx_1 = idx_1[idx_rand]

    if mask_zero:
        idx_0 = idx_0 + 1
        idx_1 = idx_1 + 1

    idx_0 = tf.constant(idx_0, dtype=tf.int32)
    idx_1 = tf.constant(idx_1, dtype=tf.int32)

    if batch_size is None:
        batch_size = np.minimum(10000, n_pair)

    ds_info = {
        'n_pair': n_pair,
        'batch_size': batch_size,
        'n_batch': np.ceil(n_pair / batch_size),
        'elements': elements,
        'mask_zero': mask_zero
    }

    if groups is not None:
        group_matrix = generate_group_matrix(n_pair, groups=groups)
        group_matrix = tf.constant(group_matrix, dtype=np.int32)

        ds = tf.data.Dataset.from_tensor_slices(
            ((idx_0, idx_1, group_matrix))
        ).batch(
            batch_size, drop_remainder=False
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices(
            ((idx_0, idx_1))
        ).batch(
            batch_size, drop_remainder=False
        )
    return ds, ds_info
