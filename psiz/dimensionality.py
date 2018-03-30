"""Module for selecting the dimensionality of an embedding.

Author: B D Roads
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from psiz.models import Observations
import psiz.utils as ut


def suggest_dimensionality(obs, embedding_constructor, n_stimuli, dim_list=None, n_restart=20, n_fold=3, 
    verbose=0):
    """Suggest an embedding dimensionality given the provided observations.

    Sweep over the list of candidate dimensions, starting with the 
    smallest, in order to find the best dimensionality for the data.
    Dimensions are examined in ascending order. The search stops when
    adding dimensions does not reduce loss or there are no more dimensions
    in the dimension list. Each dimension is evaluated using the same
    cross-validation partion.

    Parameters:
      obs: An Observations object representing the observed data.
      embedding_constructor: A PsychologicalEmbedding constructor.
      n_stimuli:  An integer indicating the number of unqiue stimuli.
      dim_list: A list of integers indicating the dimensions to search 
        over.
      n_restart: An integer specifying the number of restarts to use for 
        the inference procedure. Since the embedding procedure sometimes
        gets stuck in local optima, multiple restarts helps find the global
        optimum.
      n_fold: Integer specifying the number of folds to use for cross-
        validation when selection the dimensionality.
      verbose: An integer specifying the verbosity of printed output.
        selection_threshold: 
    Returns:
      best_dimensionality: An integer indicating the dimensionality (from
        the candiate list) that minimized the loss function.
    """

    n_group = len(np.unique(obs.group_id))

    if dim_list is None:
        dim_list = range(2,10)
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
    J_test_avg_best = np.inf
    for i_dimension in dim_list:
        # Instantiate embedding
        embedding = embedding_constructor(n_stimuli, dimensionality=i_dimension, n_group=n_group)
        if verbose > 1:
            print('  Dimensionality: ', i_dimension)
        J_train = np.empty((n_fold))
        J_test = np.empty((n_fold))
        i_fold = 0
        for train_index, test_index in skf.split(obs.stimulus_set, obs.configuration_id):
            if verbose > 1:
                print('    Fold: ', i_fold)
            # Train
            obs_train = obs.subset(train_index)
            J_train[i_fold] = embedding.fit(obs_train, n_restart=n_restart, verbose=0)
            # Test
            obs_test = obs.subset(test_index)
            J_test[i_fold] = embedding.evaluate(obs_test)
            i_fold = i_fold + 1
        # Compute average cross-validation test loss.
        J_test_avg = np.mean(J_test)
        if J_test_avg < J_test_avg_best:
            # Larger dimensionality yielded a better loss.
            J_test_avg_best = J_test_avg
            best_dimensionality = i_dimension
        else:
            # Larger dimensionality yielded a worse loss. Stop sweep.
            break
    
    if verbose > 0:
        print('Best dimensionality: ', best_dimensionality)

    return best_dimensionality
