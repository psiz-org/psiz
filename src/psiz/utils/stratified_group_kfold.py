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

Classes:
    StratifiedGroupKFold: A variant of stratified K-Folds iterator that
        ensures non-overlapping groups.

"""

from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import check_random_state


class StratifiedGroupKFold(_BaseKFold):
    """A stratified K-Folds iterator with non-overlapping groups.

    This cross-validation object is a variation of StratifiedKFold that
    returns stratified folds with non-overlapping groups. The folds are
    made by preserving the percentage of samples for each class.

    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).

    The difference between GroupKFold and StratifiedGroupKFold is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of samples for each class.

    Example:
    ```
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> cv = StratifiedGroupKFold(n_splits=3)
    >>> for train_idxs, test_idxs in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [2 2 4 5 5 5 5 6 6 7]
           [1 1 1 0 0 0 0 0 0 0]
     TEST: [1 1 3 3 3 8 8]
           [0 0 1 1 1 0 0]
    TRAIN: [1 1 3 3 3 4 5 5 5 5 8 8]
           [0 0 1 1 1 1 0 0 0 0 0 0]
     TEST: [2 2 6 6 7]
           [1 1 0 0 0]
    TRAIN: [1 1 2 2 3 3 3 6 6 7 8 8]
           [0 0 1 1 1 1 1 0 0 0 0 0]
     TEST: [4 5 5 5 5]
           [1 0 0 0 0]
    ```

    See also:
        StratifiedKFold: Takes class information into account to build
            folds which retain class distributions (for binary or
            multiclass classification tasks).

        GroupKFold: K-fold iterator variant with non-overlapping
            groups.

    Note:
        The implementation based on a kaggle kernel that was released
        under the Apache 2.0 open source license.
        https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation

    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        """Initialize.

        Args:
        n_splits (optional): int, default=5
            Number of folds. Must be at least 2.
        shuffle (optional): bool, default=False
            Whether to shuffle each class's samples before splitting
            into batches. Note that the samples within each split will
            not be shuffled.
        random_state (optional): int or RandomState object, default=None
            When `shuffle` is True, `random_state` affects the ordering
            of the indices, which controls the randomness of each fold
            for each class. Otherwise, leave `random_state` as `None`.
            Pass an int for reproducible output across multiple
            function calls.

        """
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        # pylint: disable=signature-differs
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, group in zip(y, groups):
            y_counts_per_group[group][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        groups_and_y_counts = list(y_counts_per_group.items())
        rng = check_random_state(self.random_state)
        if self.shuffle:
            rng.shuffle(groups_and_y_counts)

        for group, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                y_counts_per_fold[i] += y_counts
                std_per_label = []
                for label in range(labels_num):
                    std_per_label.append(
                        np.std(
                            [
                                y_counts_per_fold[j][label] / y_distr[label]
                                for j in range(self.n_splits)
                            ]
                        )
                    )
                y_counts_per_fold[i] -= y_counts
                fold_eval = np.mean(std_per_label)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(group)

        for i in range(self.n_splits):
            test_indices = [
                idx for idx, group in enumerate(groups) if group in groups_per_fold[i]
            ]
            yield test_indices
