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
# ==============================================================================

"""Module for performing cross-validation.

Functions:
    crossvalidate: Use a validation procedure to select the dimensionality for
        an embedding procedure.

"""

import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy.stats import sem

import psiz.utils as ut


def crossvalidate(
        obs, emb, n_split=10, n_fold=10, random_state=None, verbose=0,
        fit_kargs=None):
    """Compute cross validation loss.

    This procedure can be used as the inner logic in a search for an
    optimal hyperparameter setting. By using the same random_state,
    each hyperparameter setting is evaluated using the same
    cross-validation partitions, making the comparisons as equitable as
    possible.

    Notes:
        The procedure attempts to create balanced splits by looking
        at the configuration index of each observation. In other words,
        it attempts to put each numbers of each trial type in each
        split.

    Arguments:
        obs: An RankObservations object representing the observed data.
            embedding_constructor: A PsychologicalEmbedding
            constructor.
        emb: A psiz.PsychologicalEmbedding object that is ready to be fit.
        n_split (optional): Integer specifying how many splits to
            create from the data. The proportion of held out data for
            each fold will be 1/n_split.
        n_fold (optional): Integer specifying the number of folds to
            evaluate during cross-validation. Must be at least two and
            cannot be more than n_split.
        verbose (optional): An integer specifying the verbosity of
            printed output.

    Returns:
        summary: A dictionary.
            loss_train: The training loss.
                shape=(n_fold,)
            loss_test: The test loss.
                shape=(n_fold,)

    Raises:
        ValueError: If `n_split` or `n_fold` arguments are invalid.

    """
    # Check n_split.
    if n_split < 2:
        raise ValueError((
            "The argument `n_split` must be an integer equal to or greater "
            "than 2."
        ))
    # Check n_fold.
    if n_fold < 1 or n_fold > n_split:
        raise ValueError((
            "The argument `n_fold` must be an interger between 1 and "
            "`n_split` (inclusive)."
        ))

    if (verbose > 0):
        print('[psiz] Evaluating model using cross-validation ...')
    if (verbose > 1):
        print(
            '    Settings: n_split: {0} | n_fold: {1}'.format(n_split, n_fold)
        )

    # Instantiate the balanced k-fold cross-validation object.
    skf = StratifiedKFold(
        n_splits=n_split, shuffle=True, random_state=random_state
    )
    split_list = list(skf.split(obs.stimulus_set, obs.config_idx))

    loss_train = np.zeros([n_fold])
    loss_val = np.zeros([n_fold])
    loss_test = np.zeros([n_fold])

    for i_fold in range(n_fold):
        (train_index, test_index) = split_list[i_fold]
        if verbose > 1:
            print('    Fold: ', i_fold)

        # Reset embedding. Is this necessary? Does it matter if folds build on each other?
        # emb.reinitialize()  # TODO PROBLEM this will change untrainable settings.

        # Split data into train and test.
        obs_train = obs.subset(train_index)
        obs_test = obs.subset(test_index)

        # Train.
        loss_train[i_fold], loss_val[i_fold] = emb.fit(
            obs_train, **fit_kargs
        )
        # Test.
        loss_test[i_fold] = emb.evaluate(obs_test)

    # Combine train and validation loss used during fit.
    loss_train_val = (.9 * loss_train) + (.1 * loss_val)

    if verbose > 1:
        print(
            "        Avg. train/val loss: {0:.2f}".format(
                np.mean(loss_train_val)
            )
        )
        print("        Avg. test loss: {0:.2f}".format(np.mean(loss_test)))

    result = {
        "loss_train": loss_train,
        "loss_val": loss_val,
        "loss_train_val": loss_train_val,
        "loss_test": loss_test,
    }
    return result
