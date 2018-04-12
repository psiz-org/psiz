"""Module for testing utils.py

Todo:
    group into classes
"""

import numpy as np

from psiz.utils import possible_outcomes
from psiz.trials import UnjudgedTrials

def test_possible_outcomes_2c1():
    """
    """
    displays = np.array(((0, 1, 2), (9, 12, 7)))
    n_selected = 1 * np.ones((2))
    obs = UnjudgedTrials(displays, n_selected=n_selected)
    
    po = possible_outcomes(obs.configurations.iloc[0])

    correct = np.array(((0, 1), (1, 0)))
    np.testing.assert_array_equal(po, correct)

def test_possible_outcomes_3c2():
    """
    """
    displays = np.array(((0, 1, 2, 3), (33, 9, 12, 7)))
    n_selected = 2 * np.ones((2))
    obs = UnjudgedTrials(displays, n_selected=n_selected)

    po = possible_outcomes(obs.configurations.iloc[0])

    correct = np.array(((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), 
    (2, 0, 1), (2, 1, 0)))
    np.testing.assert_array_equal(po, correct)

def test_possible_outcomes_4c2():
    """
    """
    # displays = np.array(((0, 1, 2, 3, 4, 5, 6, 7, 8), (9, 11, 13, 2, 6, 12, 7, 23, 9)))
    displays = np.array(((0, 1, 2, 3, 4), (45, 33, 9, 12, 7)))
    n_selected = 2 * np.ones((2))
    obs = UnjudgedTrials(displays, n_selected=n_selected)

    po = possible_outcomes(obs.configurations.iloc[0])

    correct = np.array(((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), 
    (1, 0, 2, 3), (1, 2, 0, 3), (1, 3, 0, 2),
    (2, 0, 1, 3), (2, 1, 0, 3), (2, 3, 0, 1),
    (3, 0, 1, 2), (3, 1, 0, 2), (3, 2, 0, 1)))
    np.testing.assert_array_equal(po, correct)

def test_possible_outcomes_8c1():
    """
    """
    # displays = np.array(((0, 1, 2, 3, 4, 5, 6, 7, 8), (9, 11, 13, 2, 6, 12, 7, 23, 9)))
    displays = np.array(((0, 1, 2, 3, 4, 5, 6, 7, 8), 
    (45, 33, 9, 12, 7, 2, 5, 4, 3)))
    n_selected = 1 * np.ones((2))
    obs = UnjudgedTrials(displays, n_selected=n_selected)

    po = possible_outcomes(obs.configurations.iloc[0])

    correct = np.array(((0, 1, 2, 3, 4, 5, 6, 7), 
    (1, 0, 2, 3, 4, 5, 6, 7), 
    (2, 0, 1, 3, 4, 5, 6, 7), 
    (3, 0, 1, 2, 4, 5, 6, 7), 
    (4, 0, 1, 2, 3, 5, 6, 7), 
    (5, 0, 1, 2, 3, 4, 6, 7),
    (6, 0, 1, 2, 3, 4, 5, 7), 
    (7, 0, 1, 2, 3, 4, 5, 6)))
    np.testing.assert_array_equal(po, correct)
