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
"""Module for selecting the dimensionality of an embedding.

Functions:
    search: Use a validation procedure to select the dimensionality for
        an embedding procedure.
    dimension_search: Use a validation procedure to select the
        dimensionality for an embedding procedure. DEPRECATED, use
        search instead.
    visualize_dimension_search: Visualize results of dimension search.

"""

import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy.stats import sem

import psiz.utils as ut


def search(obs, model_spec, search_spec=None, verbose=0):
    """Search for an appropriate embedding dimensionality.

    Search over the list of candidate dimensions, starting with the
    smallest, in order to find the best dimensionality for the data.
    Dimensions are examined in ascending order. The search stops when
    adding dimensions does not reduce loss or there are no more
    dimensions in the search list. Each dimension is evaluated using
    the same cross-validation partitions in order to make comparisons
    as equitable as possible.

    Arguments:
        obs: An RankObservations object representing the observed data.
            embedding_constructor: A PsychologicalEmbedding
            constructor.
        model_spec: A dictionary specifying the embedding model to use.
        search_spec (optional): A dictionary specifying the parameters
            of the search procedure.
        verbose (optional): An integer specifying the verbosity of
            printed output.

        model_spec: TODO
        n_stimuli:  An integer indicating the number of unique stimuli.
        n_restart (optional): An integer specifying the number of
            restarts to use for the inference procedure. Since the
            embedding procedure finds local optima, multiple restarts
            helps find the global optimum.
        modifier_func (optional): A function that takes an embedding
            as the only argument and returns a modified embedding. This
            argument can be used to modify an embedding after it is
            initialized. For example, to set and freeze parameters.

        search_spec: TODO
        dim_list (optional): A list of integers indicating the
            dimensions to search over.
        n_split (optional): Integer specifying how many splits to
            create from the data. This defines the proportion of
            train and test data.
        n_fold (optional): Integer specifying the number of folds to
            use for cross-validation when selection the dimensionality.
            Must be at least one and cannot be more than n_split.
        max_patience (optional): Integer specifying how many dimensions
            to wait for an improvement in test loss.

    Returns:
        summary: A dictionary.
            dim_list: The dimensionalities searched.
            loss_train: The training loss.
                shape=(len(dim_list), n_fold)
            loss_test: The test loss.
                shape=(len(dim_list), n_fold)
            dim_best: An integer indicating the dimensionality
                (from the candidate list) that minimized loss on the
                held-out data.

    """
    if search_spec is None:
        search_spec = {
            'dim_list': range(2, 51),
            'n_restart': 100,
            'n_split': 10,
            'n_fold': 10
        }

    # Unpack and check.
    dim_list = np.sort(search_spec['dim_list'])
    n_restart = search_spec['n_restart']
    n_split = search_spec['n_split']
    n_fold = search_spec['n_fold']
    max_patience = search_spec['max_patience']
    # if search_spec['n_fold'] > search_spec['n_split']: TODO issue error

    if (verbose > 0):
        print('[psiz] Searching dimensionality ...')
        print('  Settings:')
        print('    Dimensionality search list: ', dim_list)
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
    loss_val = np.nan * np.ones([len(dim_list), n_fold])
    loss_test = np.nan * np.ones([len(dim_list), n_fold])
    patience = 0
    for idx_dim, i_dimension in enumerate(dim_list):
        # Instantiate embedding.
        emb = model_spec['model'](
            model_spec['n_stimuli'], n_dim=i_dimension,
            n_group=model_spec['n_group']
        )
        if model_spec['modifier'] is not None:
            emb = model_spec['modifier'](emb)
        if verbose > 1:
            print('  Dimensionality: ', i_dimension)

        for i_fold in range(n_fold):
            (train_index, test_index) = split_list[i_fold]
            if verbose > 2:
                print('    Fold: ', i_fold)
            # Train
            obs_train = obs.subset(train_index)
            loss_train[idx_dim, i_fold], loss_val[idx_dim, i_fold] = emb.fit(
                obs_train, n_restart=n_restart, verbose=verbose-1
            )
            # Test
            obs_test = obs.subset(test_index)
            loss_test[idx_dim, i_fold] = emb.evaluate(obs_test)

            i_fold = i_fold + 1
        # Compute average cross-validation train and test loss.
        loss_train_avg = np.mean(
            (.9 * loss_train[idx_dim, :]) + (.1 * loss_val[idx_dim, :])
        )
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
        obs: An RankObservations object representing the observed data.
            embedding_constructor: A PsychologicalEmbedding
            constructor.
        n_stimuli:  An integer indicating the number of unique stimuli.
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
                (from the candidate list) that minimized the loss
                function.

    """
    n_group = len(np.unique(obs.group_id))

    if dim_list is None:
        dim_list = range(2, 51)
    else:
        # Make sure dimensions are in ascending order.
        dim_list = np.sort(dim_list)

    if (verbose > 0):
        print('Searching dimensionality ...')
        print('  Settings:')
        print('    Dimensionality search list: ', dim_list)
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
            loss_train[idx_dim, i_fold], loss_val[idx_dim, i_fold] = emb.fit(
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


def visualize_dimension_search(ax, summary):
    """Visualize dimensionality search.

    Arguments:
        ax: A Matplotlib axis

    Example usage:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax = plt.subplot(1, 1, 1)
        visualize_dimension_search(ax, dim_summary)

    """
    dim_list = summary["dim_list"]
    dim_best = summary["dim_best"]

    train_mean = np.mean(summary["loss_train"], axis=1)
    test_mean = np.mean(summary["loss_test"], axis=1)

    ax.plot(dim_list, train_mean, 'o-b', markersize=3, label="Train")
    ax.plot(dim_list, test_mean, 'o-r', markersize=3, label="Test")

    if summary["loss_train"].shape[1] > 1:
        train_sem = sem(summary["loss_train"], axis=1)
        test_sem = sem(summary["loss_test"], axis=1)
        ax.fill_between(
            dim_list, train_mean - train_sem, train_mean + train_sem,
            alpha=.5
        )
        ax.fill_between(
            dim_list, test_mean - test_sem, test_mean + test_sem,
            alpha=.5
        )
    ax.scatter(
        dim_best, test_mean[np.equal(dim_list, dim_best)], c="r", s=50,
        marker='x', label='Best Dimensionality'
    )

    ax.set_xlabel("Dimensionality")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title('Dimensionality Search\n(Mean and SEM)')
