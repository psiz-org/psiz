'''Module of helpful utility functions.

Author: B D Roads
'''

import numpy as np
import pandas as pd

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
    group_id):
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
    Returns:
      display_type_id: a unique id for each type of display configuration 
        present in the data.
    '''
    d = {'col0': n_reference, 'col1': n_selected, 'col2': is_ranked, 
    'col3': group_id}
    df = pd.DataFrame(d)
    df_unique = df.drop_duplicates()
    n_display_type = len(df_unique)

    display_type_id = np.empty(len(n_reference))
    for i_type in range(n_display_type):
        a = (n_reference == df_unique['col0'].iloc[i_type])
        b = (n_selected == df_unique['col1'].iloc[i_type])
        c = (is_ranked == df_unique['col2'].iloc[i_type])
        d = (group_id == df_unique['col3'].iloc[i_type])
        e = np.array((a,b,c,d))
        display_type_locs = np.any(e, axis=0)
        display_type_id[display_type_locs] = i_type
    return display_type_id