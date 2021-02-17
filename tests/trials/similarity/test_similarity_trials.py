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
"""Test `trials` module."""

import pytest
import numpy as np

from psiz.trials.similarity.similarity_trials import SimilarityTrials


def test_split_groups_columns():
    n_trial = 5
    groups = np.zeros([n_trial, 1])
    d_groups = SimilarityTrials._split_groups_columns(groups)

    groups_0_expected = np.zeros([n_trial])
    np.testing.assert_array_equal(
        d_groups['groups_0'], groups_0_expected
    )

    groups = np.hstack(
        (np.zeros([n_trial, 1]), np.ones([n_trial, 1]))
    )
    d_groups = SimilarityTrials._split_groups_columns(groups)

    groups_0_expected = np.zeros([n_trial])
    groups_1_expected = np.ones([n_trial])
    np.testing.assert_array_equal(
        d_groups['groups_0'], groups_0_expected
    )
    np.testing.assert_array_equal(
        d_groups['groups_1'], groups_1_expected
    )
