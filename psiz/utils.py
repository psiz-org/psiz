'''Module of helpful utility functions.

Author: B D Roads
'''

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def matrix_correlation(mat_A, mat_B):
    '''Returns the R^2 score between two square matrices.

    Parameters:
      mat_A: A square matrix.
      mat_B: A square matrix the same size as mat_A
      is_sym: Boolean indicating if the matrices are symmetric.
    '''
    n_row = mat_A.shape[0]
    iu1 = np.triu_indices(n_row,1)

    # Explained variance score.
    return r2_score(mat_A[iu1], mat_B[iu1])

def infer_n_reference(displays):
    ''' Infer the number of references in each display.

    Helper function that infers the number of available references for a 
    given display. The function assumes that values less than zero, are
    placeholder values and should be treated as non-existent.

    Parameters:
      displays: 
        shape = [n_display, 1]
    
    Returns:
      n_reference: An integer array indicating the number of references in each
        display.
        shape = [n_display, 1]
    '''
    max_ref = displays.shape[1] - 1
    n_reference = max_ref - np.sum(displays<0, axis=1)            
    return np.array(n_reference)

def generate_display_type_id(n_reference, n_selected, is_ranked, 
    group_id, assignment_id=None):
    '''Generate a unique ID for each display configuration.

    Helper function that generates a unique ID for each of the unique
    display configurations in the provided data set.

    Parameters:
      n_reference: An integer array indicating the number of references in each
        display.
        shape = [n_display, 1]
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
      assignment_id: An integer array indicating the assignment ID of the
        display. It is assumed that displays with a given assignment ID were
        judged by a single person although a single person may have completed
        multiple assignments (e.g., Amazon Mechanical Turk).
        shape = [n_display, 1]
    Returns:
      display_type_id: a unique id for each type of display configuration 
        present in the data.
    '''
    n_display = len(n_reference)

    if assignment_id is None:
        assignment_id = np.ones((n_display))
    
    d = {'col0': n_reference, 'col1': n_selected, 'col2': is_ranked, 
    'col3': group_id, 'col4': assignment_id}
    df = pd.DataFrame(d)
    df_unique = df.drop_duplicates()
    n_display_type = len(df_unique)

    display_type_id = np.empty(n_display)
    for i_type in range(n_display_type):
        a = (n_reference == df_unique['col0'].iloc[i_type])
        b = (n_selected == df_unique['col1'].iloc[i_type])
        c = (is_ranked == df_unique['col2'].iloc[i_type])
        d = (group_id == df_unique['col3'].iloc[i_type])
        e = (assignment_id == df_unique['col4'].iloc[i_type])
        f = np.array((a,b,c,d,e))
        display_type_locs = np.all(f, axis=0)
        display_type_id[display_type_locs] = i_type
    return display_type_id