import numpy as np

from psiz.utils import possible_outcomes
from psiz.trials import UnjudgedTrials

displays = np.array(((0, 1, 2, 3, 4), (45, 33, 9, 12, 7)))
n_selected = 2 * np.ones((2))
obs = UnjudgedTrials(displays, n_selected=n_selected)

po = possible_outcomes(obs.configurations.iloc[0])

correct = np.array(((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), 
(1, 0, 2, 3), (1, 2, 0, 3), (1, 3, 0, 2),
(2, 0, 1, 3), (2, 1, 0, 3), (2, 3, 0, 1),
(3, 0, 1, 2), (3, 1, 0, 2), (3, 2, 0, 1)))
np.testing.assert_array_equal(po, correct)