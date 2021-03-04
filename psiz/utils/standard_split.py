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
"""Module of utility functions.

Functions:
    standard_split: Standard 80-10-10 split of observations.

"""

from sklearn.model_selection import StratifiedKFold


def standard_split(obs, shuffle=False, seed=None):
    """Creata a standard 80-10-10 split of the observations.

    Arguments:
        obs: A set of observations.
        shuffle (optional): Boolean indicating if the data should be
            shuffled before splitting.
        seed: Integer to seed randomness.

    Returns:
        obs_train: A train set (80%).
        obs_val: A validation set (10%).
        obs_test: A test set (10%).

    """
    skf = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=seed)
    (train_idx, holdout_idx) = list(
        skf.split(obs.stimulus_set, obs.config_idx)
    )[0]
    obs_train = obs.subset(train_idx)
    obs_holdout = obs.subset(holdout_idx)
    skf = StratifiedKFold(n_splits=2, shuffle=shuffle, random_state=seed)
    (val_idx, test_idx) = list(
        skf.split(obs_holdout.stimulus_set, obs_holdout.config_idx)
    )[0]
    obs_val = obs_holdout.subset(val_idx)
    obs_test = obs_holdout.subset(test_idx)
    return obs_train, obs_val, obs_test
