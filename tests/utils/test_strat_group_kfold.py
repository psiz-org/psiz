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
"""Module for testing utils.py."""

import numpy as np
from psiz import utils


def test_strat_group_kfold():

    x = np.ones((17, 2))
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    group = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    skf = utils.StratifiedGroupKFold(n_splits=3)

    y_count = []
    for train_idx, test_idx in skf.split(x, y, group):
        x_train = x[train_idx]
        y_train = y[train_idx]
        group_train = group[train_idx]

        x_test = x[test_idx]
        y_test = y[test_idx]
        group_test = group[test_idx]

        # Assert that groups are mutually exclusive in train and test.
        intersect_arr = np.intersect1d(
            np.unique(group_train), np.unique(group_test)
        )
        assert len(intersect_arr) == 0
        y_count.append(np.sum(np.equal(y_train, 0)))

    # Assert balanced classes across splits.
    np.testing.assert_array_equal(y_count, np.array([7, 8, 7]))
