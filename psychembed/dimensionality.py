'''Module for selecting the dimensionality of an embedding.

Author: B D Roads
'''

import numpy as np
from sklearn.model_selection import StratifiedKFold
import psychembed.utils as ut

def suggest_dimensionality(embedding_constructor, n_stimuli, displays, n_selected=None, 
    is_ranked=None, group_id=None, dim_list=None, n_restart=20, n_fold=3, 
    verbose=0):
    '''Suggest an embedding dimensionality given the provided observations.

    Sweep over the list of candidate dimensions, starting with the 
    smallest, in order to find the best dimensionality for the data.
    Dimensions are examined in ascending order. The search stops when
    adding dimensions does not reduce loss or there are no more dimensions
    in the dimension list. Each dimension is evaluated using the same
    cross-validation partion.

    Parameters:
      embedding_constructor: A PsychologicalEmbedding constructor.
      n_stimuli:  An integer indicating the number of unqiue stimuli.
      displays: An integer matrix representing the displays (rows) that 
        have been judged based on similarity. The shape implies the 
        number of references in shown in each display. The first column 
        is the query, then the selected references in order of selection,
        and then any remaining unselected references.
        shape = [n_display, max(n_reference) + 1]
      n_selected: An integer array indicating the number of references 
        selected in each display.
        shape = [n_display, 1]
      is_ranked:  Boolean array indicating which displays had selected
        references that were ordered.
        shape = [n_display, 1]
      group_id: An integer array indicating the group membership of each 
        display. It is assumed that group is composed of integers from 
        [0,N] where N is the total number of groups. Separate attention 
        weights are inferred for each group.
        shape = [n_display, 1]
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
    '''

    n_display = displays.shape[0]
    # Handle default settings
    if n_selected is None:
        n_selected = np.ones((n_display))
    if is_ranked is None:
        is_ranked = np.full((n_display), True)
    if group_id is None:
        group_id = np.zeros((n_display))
        n_group = 1
    else:
        n_group = len(np.unique(group_id))

    # Infer n_reference for each display
    n_reference = ut.infer_n_reference(displays)

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

    # Infer the display type IDs.
    display_type_id = ut.generate_display_type_id(n_reference, 
    n_selected, is_ranked, group_id)
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
        for train_index, test_index in skf.split(displays, display_type_id):
            if verbose > 1:
                print('    Fold: ', i_fold)
            # Train
            displays_train = displays[train_index,:]
            n_selected_train = n_selected[train_index]
            is_ranked_train = is_ranked[train_index]
            group_id_train = group_id[train_index]
            J_train[i_fold] = embedding.fit(displays_train, 
            n_selected=n_selected_train, is_ranked=is_ranked_train, 
            group_id=group_id_train, n_restart=n_restart, verbose=0)
            # Test
            displays_test = displays[test_index,:]
            n_selected_test = n_selected[test_index]
            is_ranked_test = is_ranked[test_index]
            group_id_test = group_id[test_index]
            J_test[i_fold] = embedding.evaluate(displays_test, 
            n_selected=n_selected_test, is_ranked=is_ranked_test, 
            group_id=group_id_test)
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
