# -*- coding: utf-8 -*-
# Copyright 2018 The PsiZ Authors. All Rights Reserved.
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
    suggest_dimensionality: Use a cross-validation procedure to select
        a dimensionality for an embedding procedure. DEPRECATED
    dimension_search: Use a validation procedure to select
        the dimensionality for an embedding procedure.
    visualize_dimension_search: Visualize results of dimension search.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

import psiz.utils as ut


def suggest_dimensionality(
        obs, embedding_constructor, n_stimuli, dim_list=None,
        freeze_options=None, n_restart=20, n_fold=3, verbose=0):
    """Suggest an embedding dimensionality given provided observations.

    DEPRECATED use `dimension_search`.

    Sweep over the list of candidate dimensions, starting with the
    smallest, in order to find the best dimensionality for the data.
    Dimensions are examined in ascending order. The search stops when
    adding dimensions does not reduce loss or there are no more
    dimensions in the dimension list. Each dimension is evaluated using
    the same cross-validation partitions.

    Arguments:
        obs: An Observations object representing the observed data.
            embedding_constructor: A PsychologicalEmbedding
            constructor.
        n_stimuli:  An integer indicating the number of unqiue stimuli.
        dim_list (optional): A list of integers indicating the dimensions to
            search over.
        freeze_options (optional): Dictionary of freeze options.
        n_restart (optional): An integer specifying the number of
            restarts to use for the inference procedure. Since the
            embedding procedure finds local optima, multiple restarts
            helps find the global optimum.
        n_fold (optional): Integer specifying the number of folds to
            use for cross-validation when selection the dimensionality.
        verbose (optional): An integer specifying the verbosity of
            printed output.

    Returns:
        best_dimensionality: An integer indicating the dimensionality
        (from the candiate list) that minimized the loss function.

    """
    n_group = len(np.unique(obs.group_id))

    if dim_list is None:
        dim_list = range(2, 20)
    else:
        # Make sure dimensions are in ascending order
        dim_list = np.sort(dim_list)

    if (verbose > 0):
        print('Selecting dimensionality ...')
        print('  Settings:')
        print('    Candidate dimensionaltiy list: ', dim_list)
        print('    Folds: ', n_fold)
        print('    Restarts per fold: ', n_restart)

    # Instantiate the balanced k-fold cross-validation object.
    skf = StratifiedKFold(n_splits=n_fold)

    # Sweep over the list of candidate dimensions.
    best_dimensionality = dim_list[0]
    loss_test_avg_best = np.inf
    for i_dimension in dim_list:
        # Instantiate embedding
        embedding = embedding_constructor(
            n_stimuli, n_dim=i_dimension, n_group=n_group)
        if freeze_options is not None:
            embedding.freeze(freeze_options)
        if verbose > 1:
            print('  Dimensionality: ', i_dimension)
        loss_train = np.empty((n_fold))
        loss_test = np.empty((n_fold))
        i_fold = 0
        for train_index, test_index in skf.split(
                obs.stimulus_set, obs.config_idx):
            if verbose > 1:
                print('    Fold: ', i_fold)
            # Train
            obs_train = obs.subset(train_index)
            loss_train[i_fold], _ = embedding.fit(
                obs_train, n_restart=n_restart, verbose=verbose-1)
            # Test
            obs_test = obs.subset(test_index)
            loss_test[i_fold] = embedding.evaluate(obs_test)
            i_fold = i_fold + 1
        # Compute average cross-validation test loss.
        loss_test_avg = np.mean(loss_test)

        if loss_test_avg < loss_test_avg_best:
            # Larger dimensionality yielded a better loss.
            loss_test_avg_best = loss_test_avg
            best_dimensionality = i_dimension
        else:
            # Larger dimensionality yielded a worse loss. Stop sweep.
            break

    if verbose > 0:
        print('Best dimensionality: ', best_dimensionality)

    return best_dimensionality


def dimension_search(
        obs, embedding_constructor, n_stimuli, dim_list=None,
        freeze_options=None, n_restart=20, n_split=5, n_fold=1,
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
        dim_list (optional): A list of integers indicating the dimensions to
            search over.
        freeze_options (optional): Dictionary of freeze options.
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
        # Make sure dimensions are in ascending order
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
    loss_train_list = np.nan * np.ones(len(dim_list))
    loss_test_list = np.nan * np.ones(len(dim_list))
    patience = 0
    for idx_dim, i_dimension in enumerate(dim_list):
        # Instantiate embedding
        embedding = embedding_constructor(
            n_stimuli, n_dim=i_dimension, n_group=n_group)
        if freeze_options is not None:
            embedding.freeze(freeze_options)
        if verbose > 1:
            print('  Dimensionality: ', i_dimension)
        loss_train = np.empty((n_fold))
        loss_test = np.empty((n_fold))
        for i_fold in range(n_fold):
            (train_index, test_index) = split_list[i_fold]
            if verbose > 2:
                print('    Fold: ', i_fold)
            # Train
            obs_train = obs.subset(train_index)
            loss_train[i_fold], _ = embedding.fit(
                obs_train, n_restart=n_restart, verbose=verbose-1)
            # Test
            obs_test = obs.subset(test_index)
            loss_test[i_fold] = embedding.evaluate(obs_test)
            i_fold = i_fold + 1
        # Compute average cross-validation train and test loss.
        loss_train_avg = np.mean(loss_train)
        loss_test_avg = np.mean(loss_test)
        loss_train_list[idx_dim] = loss_train_avg
        loss_test_list[idx_dim] = loss_test_avg

        if verbose > 1:
            print("    Avg. Train Loss: {0:.2f}".format(loss_train_avg))
            print("    Avg. Test Loss: {0:.2f}".format(loss_test_avg))

        if loss_test_avg < loss_test_avg_best:
            # Larger dimensionality yielded a better loss.
            loss_test_avg_best = loss_test_avg
            best_dimensionality = i_dimension
            patience = 0
            if verbose > 1:
                print("    Test loss improved.")
        else:
            # Larger dimensionality yielded a worse loss.
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
        "loss_train": loss_train_list,
        "loss_test": loss_test_list,
        "dim_best": best_dimensionality
    }
    return summary


def visualize_dimension_search(summary, fp_fig=None):
    """Visualize dimensionality search."""
    dim_list = summary["dim_list"]
    dim_best = summary["dim_best"]

    plt.plot(dim_list, summary["loss_train"], 'b', label="Train")
    plt.plot(dim_list, summary["loss_test"], 'r', label="Test")
    plt.scatter(
        dim_best, summary["loss_test"][np.equal(dim_list, dim_best)], c="r"
    )
    plt.xlabel("Dimensionality")
    plt.ylabel("Loss")
    plt.legend()

    if fp_fig is None:
        # plt.tight_layout()
        plt.show()
    else:
        # Note: The dpi must be supplied otherwise the aspect ratio will be
        # changed when savefig is called.
        plt.savefig(fp_fig, format='pdf', bbox_inches="tight", dpi=300)
