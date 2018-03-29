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

