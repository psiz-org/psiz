# -*- coding: utf-8 -*-
# Copyright 2019 The PsiZ Authors. All Rights Reserved.
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

"""Module for selecting the dimensionality of an embedding.

Functions:
    dimension_search: Use a validation procedure to select
        the dimensionality for an embedding procedure.
    visualize_dimension_search: Visualize results of dimension search.
"""

import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy.stats import sem

import psiz.utils as ut


def dimension_search(
        obs, embedding_constructor, n_stimuli, dim_list=None,
        modifier_func=None, n_restart=20, n_split=5, n_fold=1,
        max_patience=1, verbose=0):
    """Suggest an embedding dimensionality given provided observations.

    Search over the list of candidate dimensions, starting with the
    smallest, in order to find the best dimensionality for the data.
    Dimensions are examined in ascending order. The search stops when
    adding dimensions does not reduce loss or there are no more
    dimensions in the search list. Each dimension is evaluated using
    the same cross-validation partitions.

    Arguments:
        obs: An Observations object representing the observed data.
            embedding_constructor: A PsychologicalEmbedding
            constructor.
        n_stimuli:  An integer indicating the number of unqiue stimuli.
        dim_list (optional): A list of integers indicating the
            dimensions to search over.
        modifier_func (optional): A function that takes an embedding
            as the only argument and returns a modified embedding. This
            argument can be used to modify an embedding after it is
            initialized. For example, to set and freeze parameters.
        n_restart (optional): An integer specifying the number of
            restarts to use for the inference procedure. Since the
            embedding procedure finds local optima, multiple restarts
            helps find the global optimum.
        n_split (optional): Integer specifying how many splits to
            create from the data. This defines the proportion of
            train and test data.
        n_fold (optional): Integer specifying the number of folds to
            use for cross-validation when selection the dimensionality.
            Must be at least one and cannot be more than n_split.
        max_patience (optional): Integer specifying how many dimensions
            to wait for an improvement in test loss.
        verbose (optional): An integer specifying the verbosity of
            printed output.

    Returns:
        summary: A dictionary.
            dim_best: An integer indicating the dimensionality
                (from the candiate list) that minimized the loss
                function.

    """
    n_group = len(np.unique(obs.group_id))

    if dim_list is None:
        dim_list = range(2, 21)
    else:
        # Make sure dimensions are in ascending order.
        dim_list = np.sort(dim_list)

    if (verbose > 0):
        print('Selecting dimensionality ...')
        print('  Settings:')
        print('    Dimensionaltiy search list: ', dim_list)
        print('    Splits: ', n_split)
        print('    Folds: ', n_fold)
        print('    Restarts per fold: ', n_restart)
        print('    Patience: ', max_patience)
        print('')

    # Instantiate the balanced k-fold cross-validation object.
    skf = StratifiedKFold(n_splits=n_split)
    split_list = list(
        skf.split(obs.stimulus_set, obs.group_id)
    )

    # Sweep over the list of candidate dimensions.
    loss_test_avg_best = np.inf
    loss_train = np.nan * np.ones([len(dim_list), n_fold])
    loss_test = np.nan * np.ones([len(dim_list), n_fold])
    patience = 0
    for idx_dim, i_dimension in enumerate(dim_list):
        # Instantiate embedding.
        emb = embedding_constructor(
            n_stimuli, n_dim=i_dimension, n_group=n_group
        )
        if modifier_func is not None:
            emb = modifier_func(emb)
        if verbose > 1:
            print('  Dimensionality: ', i_dimension)

        for i_fold in range(n_fold):
            (train_index, test_index) = split_list[i_fold]
            if verbose > 2:
                print('    Fold: ', i_fold)
            # Train
            obs_train = obs.subset(train_index)
            loss_train[idx_dim, i_fold], _ = emb.fit(
                obs_train, n_restart=n_restart, verbose=verbose-1
            )
            # Test
            obs_test = obs.subset(test_index)
            loss_test[idx_dim, i_fold] = emb.evaluate(obs_test)

            i_fold = i_fold + 1
        # Compute average cross-validation train and test loss.
        loss_train_avg = np.mean(loss_train[idx_dim, :])
        loss_test_avg = np.mean(loss_test[idx_dim, :])

        if verbose > 1:
            print("    Avg. Train Loss: {0:.2f}".format(loss_train_avg))
            print("    Avg. Test Loss: {0:.2f}".format(loss_test_avg))

        if loss_test_avg < loss_test_avg_best:
            # Larger dimensionality yielded a better test loss.
            loss_test_avg_best = loss_test_avg
            best_dimensionality = i_dimension
            patience = 0
            if verbose > 1:
                print("    Test loss improved.")
        else:
            # Larger dimensionality yielded a worse test loss.
            patience = patience + 1
            if verbose > 1:
                print(
                    "    Test loss did not improve."
                    "(patience={0})".format(patience)
                )
        if verbose > 1:
            print("")

        if patience > max_patience:
            # Stop search.
            print('break block')
            break

    if verbose > 0:
        print('Best dimensionality: ', best_dimensionality)

    summary = {
        "dim_list": dim_list,
        "loss_train": loss_train,
        "loss_test": loss_test,
        "dim_best": best_dimensionality
    }
    return summary


def visualize_dimension_search(summary, fp_fig=None):
    """Visualize dimensionality search."""
    dim_list = summary["dim_list"]
    dim_best = summary["dim_best"]

    train_mean = np.mean(summary["loss_train"], axis=1)
    test_mean = np.mean(summary["loss_test"], axis=1)

    plt.plot(dim_list, train_mean, 'b', label="Train")
    plt.plot(dim_list, test_mean, 'r', label="Test")

    if summary["loss_train"].shape[1] > 1:
        train_sem = sem(summary["loss_train"], axis=1)
        test_sem = sem(summary["loss_test"], axis=1)
        plt.fill_between(
            dim_list, train_mean - train_sem, train_mean + train_sem,
            alpha=.5
        )
        plt.fill_between(
            dim_list, test_mean - test_sem, test_mean + test_sem,
            alpha=.5
        )
    plt.scatter(
        dim_best, test_mean[np.equal(dim_list, dim_best)], c="r"
    )

    plt.xlabel("Dimensionality")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    if fp_fig is None:
        plt.show()
    else:
        plt.savefig(
            os.fspath(fp_fig), format='pdf', bbox_inches="tight", dpi=300
        )
