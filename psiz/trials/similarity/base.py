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
    Groups are used to identify distinct populations of agents. For  TODO expand and make general
        example, similarity judgments could be collected from two
        groups: novices and experts. During inference, group
        information can be used to infer a separate set of attention
        weights for each group while sharing all other parameters.

TODO:
    * Add SortDocket class
    * Add SortObservations class
    * Add Observations "interface" which requires `as_dataset()` method
        which returns a tf.data.Dataset object, agent_id, group_id, and
        session_id.
    * MAYBE restructure group_id and agent_id. If we wanted to allow
    for arbitrary hierarchical models, maybe better off making
    group_id a 2D array of shape=(n_trial, n_group_level)

"""

from abc import ABCMeta, abstractmethod
from itertools import permutations
import copy
import warnings

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K


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

    @abstractmethod
    def _set_configuration_data(self, *args):
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
        pass

    @abstractmethod
    def subset(self, index):
        """Return subset of trials as new SimilarityTrials object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new SimilarityTrials object.

        """
        pass

    @abstractmethod
    def save(self, filepath):
        """Save the SimilarityTrials object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        pass

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
