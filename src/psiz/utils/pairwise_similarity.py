
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
    pairwise_similarity: Compute pairwise similarity for a dataset of
        stimulus pairs.

"""

import numpy as np
import tensorflow as tf

from psiz.utils.expand_dim_repeat import expand_dim_repeat
from psiz.utils.progress_bar_re import ProgressBarRe


def pairwise_similarity(
        stimuli, kernel, ds_pairs, use_group_stimuli=False,
        use_group_kernel=False, n_sample=None, compute_average=False,
        verbose=0):
    """Return the similarity between stimulus pairs.

    Arguments:
        stimuli: A tf.keras.layers.Layer with stimuli semantics.
        kernel: A tf.keras.layers.Layer with kernel sematnics.
        ds_pairs: A TF dataset object that yields a 2-tuple or 3-tuple
            composed of stimulus index i, sitmulus index j, and
            (optionally) group membership indices.
        use_group_stimuli: Boolean indicating if `stimuli` layer should
            receive group input.
        use_group_kernel: Boolean indicating if `kernel` layer should
            receive group input.
        n_sample (optional): The size of an additional "sample" axis.
        compute_average (optional): Boolean indicating if an average
            across samples should be computed.
        verbose (optional): Verbosity of output.

    Returns:
        s: A tf.Tensor of similarities between stimulus i and stimulus
            j (using the requested group-level parameters from the
            stimuli layer and the kernel layer).
            shape=(n_pair, [n_sample])

    """
    if verbose > 0:
        n_batch = 0
        for _ in ds_pairs:
            n_batch += 1
        progbar = ProgressBarRe(n_batch, prefix='Similarity:', length=50)
        progbar.update(0)
        progbar_counter = 0
        # Determine how often progbar should update (we use 50 since that is
        # the visual length of the progbar).
        progbar_update = np.maximum(1, int(np.ceil(n_batch / 50)))

    s = []
    for x_batch in ds_pairs:
        idx_0 = x_batch[0]
        idx_1 = x_batch[1]
        if use_group_stimuli or use_group_kernel:
            group = x_batch[2]

        if n_sample is not None:
            idx_0 = expand_dim_repeat(
                idx_0, n_sample, axis=1
            )
            idx_1 = expand_dim_repeat(
                idx_1, n_sample, axis=1
            )

        if use_group_stimuli:
            z_0 = stimuli([idx_0, group])
            z_1 = stimuli([idx_1, group])
        else:
            z_0 = stimuli(idx_0)
            z_1 = stimuli(idx_1)

        if use_group_kernel:
            s_batch = kernel([z_0, z_1, group])
        else:
            s_batch = kernel([z_0, z_1])

        if compute_average:
            s_batch = tf.reduce_mean(s_batch, axis=1)

        s.append(s_batch)

        if verbose > 0:
            if (np.mod(progbar_counter, progbar_update) == 0):
                progbar.update(progbar_counter + 1)
            progbar_counter += 1

    if verbose > 0:
        progbar.update(n_batch)

    # Concatenate along pairs dimension (i.e., the first axis).
    s = tf.concat(s, 0)
    return s
