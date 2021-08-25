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
"""Module for similarity judgment trials.

Classes:
    SimilarityTrials: Abstract class for similarity judgment trials.

Functions:

Notes:
    A `stimulus_id` of `-1` is a reserved value to be used as a
        placeholder.
    Groups are used to identify distinct populations of agents. For
        example, similarity judgments could be collected from two
        groups: novices and experts. During inference, group
        information can be used to infer a separate set of attention
        weights for each group while sharing all other parameters.

TODO:
    * Add SortDocket class
    * Add SortObservations class
    * Add Observations "interface" which requires `as_dataset()` method
        which returns a tf.data.Dataset object.

"""

from abc import ABCMeta, abstractmethod

import numpy as np


class SimilarityTrials(metaclass=ABCMeta):
    """Abstract base class for similarity judgment trials.

    This abstract base class is used to organize data associated with
    similarity judgment trials. As the class name suggests, this object
    handles data associated with multiple trials. Depending on the
    concrete subclass, the similarity trials represent to-be-shown
    trials (i.e., a docket) or judged trials (i.e., observations).

    Attributes:
        n_trial: An integer indicating the number of trials.
        stimulus_set: An integer matrix containing indices that
            indicate the set of stimuli used in each trial. Each row
            indicates the stimuli used in one trial.
        config_idx: An integer array indicating the
            configuration of each trial. The integer is an index
            referencing the row of config_list and the element of
            outcome_idx_list.
            shape = (n_trial,)
        config_list: A DataFrame object describing the unique trial
            configurations.
        outcome_idx_list: A list of 2D arrays indicating all possible
            outcomes for a trial configuration. Each element in the
            list corresponds to a trial configuration in config_list.
            Each row of the 2D array indicates one potential outcome.

    Methods:
        subset: Return a subset of similarity trials given an index.
        save: Save the object to disk.
        is_present: Indicate if a stimulus is present.

    """

    def __init__(self, stimulus_set):
        """Initialize.

        Arguments:
            stimulus_set: An integer matrix containing indices that
                indicate the set of stimuli used in each trial. Each
                row indicates the stimuli used in one trial. The value
                '-1' can be used as a masking placeholder to indicate
                non-existent stimuli if each trial has a different
                number of stimuli.
                shape = (n_trial, max_n_stimuli_per_trial)

        """
        stimulus_set = self._check_stimulus_set(stimulus_set)
        self.stimulus_set = stimulus_set
        self.n_trial = stimulus_set.shape[0]

        # Attributes determined by concrete class.
        self.config_idx = None
        self.config_list = None
        self.groups = None
        self.outcome_idx_list = None

    def _check_stimulus_set(self, stimulus_set):
        """Check the argument `stimulus_set`.

        Raises:
            ValueError

        """
        # Check that provided values are integers.
        if not issubclass(stimulus_set.dtype.type, np.integer):
            raise ValueError((
                "The argument `stimulus_set` must be a 2D array of "
                "integers."
            ))

        # Check that all values are greater than or equal to -1.
        # NOTE: The value '-1' is used as a masking placeholder.
        if np.sum(np.less(stimulus_set, -1)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must only contain integers "
                "greater than or equal to -1."
            ))

        # NOTE: ii32.max -1 since we will perform a +1 operation.
        ii32 = np.iinfo(np.int32)
        if np.sum(np.greater(stimulus_set, ii32.max - 1)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must only contain integers "
                "in the int32 range."
            ))
        return stimulus_set.astype(np.int32)

    def _check_groups(self, groups):
        """Check the argument groups."""
        groups = groups.astype(np.int32)
        if not (groups.ndim == 2):
            raise ValueError((
                "The argument 'groups' must be a rank 2 ND array."))
        # Check n_trial shape agreement.
        if not (groups.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'groups' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        # Check lowerbound support limit.
        bad_locs = groups < 0
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The parameter 'groups' contains integers less than 0. "
                "Found {0} bad trial(s).").format(n_bad))
        return groups

    @abstractmethod
    def _set_configuration_data(self, *args, **kwargs):
        """Generate a unique ID for each trial configuration.

        Helper function that generates a unique ID for each of the
        unique trial configurations in the provided data set.

        Notes:
            Sets three attributes of object.
            config_idx: A unique index for each type of trial
                configuration.
            config_list: A DataFrame containing all the unique
                trial configurations.
            outcome_idx_list: A list of the possible outcomes for each
                trial configuration.

        """

    @abstractmethod
    def subset(self, index):
        """Return subset of trials as new SimilarityTrials object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new SimilarityTrials object.

        """

    @abstractmethod
    def save(self, filepath):
        """Save the SimilarityTrials object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """

    def is_present(self):
        """Return a 2D Boolean array indicating a present stimulus."""
        is_present = np.not_equal(self.stimulus_set, -1)
        return is_present

    def _infer_n_present(self, stimulus_set):
        """Return the number of stimuli present in each trial.

        Assumes that -1 is a placeholder value.

        Arguments:
            stimulus_set: A 2D array of stimulus IDs.
                shape = [n_trial, None]

        Returns:
            n_present: An integer array indicating the number of
                stimuli present in each trial.
                shape = [n_trial, 1]

        """
        n_present = np.sum(self.is_present(), axis=1)
        return n_present.astype(dtype=np.int32)

    @classmethod
    def _split_groups_columns(cls, groups):
        """Split 2D `groups` into separate columns."""
        d = {}
        n_col = groups.shape[1]
        for i_col in range(n_col):
            dkey = 'groups_{0}'.format(i_col)
            d[dkey] = groups[:, i_col]
        return d

    def _find_trials_matching_config(self, row):
        """Find trials matching configuration.

        Arguments:
            row: A pandas.Series object representing a trial
                configuration.

        """
        bidx = np.ones([self.n_trial], dtype=bool)
        for index, value in row.items():
            if 'groups' in index:
                # Must handle groups separately.
                parts = index.split('_')
                group_col = int(parts[-1])
                bidx_key = np.equal(self.groups[:, group_col], value)
            else:
                bidx_key = np.equal(getattr(self, index), value)
            # Determine intersection.
            bidx = np.logical_and(bidx, bidx_key)
        return bidx
