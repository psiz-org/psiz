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
"""Rate trials module.

On each similarity judgment trial, an agent rates the similarity
between a two stimuli.

Classes:
    RateDocket: Unjudged 'Rate' trials.

"""

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from psiz.trials.similarity.rate.rate_trials import RateTrials


class RateDocket(RateTrials):
    """Object that encapsulates unseen trials.

    The attributes and behavior of RateDocket is largely inherited
    from RateTrials.

    Attributes:
        n_trial: An integer indicating the number of trials.
        stimulus_set: An integer matrix containing indices that
            indicate the set of stimuli used in each trial. Each row
            indicates the stimuli used in one trial.
            shape = (n_trial, max(n_present))
        config_idx: An integer array indicating the
            configuration of each trial. The integer is an index
            referencing the row of config_list.
            shape = (n_trial,)
        config_list: A DataFrame object describing the unique trial
            configurations. The columns are 'n_present' and
            'n_outcome'.

    Notes:
        stimulus_set: The order of the stimuli is not important.
        Unique configurations and configuration IDs are determined by
            'n_present'.

    Methods:
        save: Save the Docket object to disk.
        subset: Return a subset of unjudged trials given an index.

    """

    def __init__(self, stimulus_set):
        """Initialize.

        Arguments:
            stimulus_set: The order of the indices is not important.

        """
        RateTrials.__init__(self, stimulus_set)

        # Determine unique display configurations.
        self._set_configuration_data(self.n_present)

    def subset(self, index):
        """Return subset of trials as a new RateDocket object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new RateDocket object.

        """
        return RateDocket(self.stimulus_set[index, :])

    def _set_configuration_data(self, n_present):
        """Generate a unique ID for each trial configuration.

        Helper function that generates a unique ID for each of the
        unique trial configurations in the provided data set.

        Arguments:
            n_present: An integer array indicating the number of
                stimuli present in each trial.
                shape = (n_trial,)

        Notes:
            Sets two attributes of object.
            config_idx: A unique index for each type of trial
                configuration.
            config_list: A DataFrame containing all the unique
                trial configurations.

        """
        n_trial = len(n_present)

        # Determine unique display configurations.
        d = {'n_present': n_present}
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)

        # Assign display configuration ID for every observation.
        config_idx = np.empty(n_trial, dtype=np.int32)
        for i_config in range(n_config):
            # Find trials matching configuration.
            a = (n_present == df_config['n_present'].iloc[i_config])
            # f = np.array((a, b, c))
            # display_type_locs = np.all(f, axis=0)
            config_idx[a] = i_config

        self.config_idx = config_idx
        self.config_list = df_config

    def save(self, filepath):
        """Save the RateDocket object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        f = h5py.File(filepath, "w")
        f.create_dataset("class_name", data="RateDocket")
        f.create_dataset("stimulus_set", data=self.stimulus_set)
        f.close()

    def as_dataset(self, groups=None):
        """Return TensorFlow dataset.

        Arguments:
            groups (optional): ND array indicating group membership
                information for each trial.

        Returns:
            x: A TensorFlow dataset.

        """
        if groups is None:
            groups = np.zeros([self.n_trial, 1], dtype=np.int32)
        else:
            groups = self._check_groups(groups)

        # Return tensorflow dataset.
        stimulus_set = self.stimulus_set + 1
        x = {
            'stimulus_set': tf.constant(stimulus_set, dtype=tf.int32),
            'groups': tf.constant(groups, dtype=tf.int32)
        }
        return tf.data.Dataset.from_tensor_slices((x))

    @classmethod
    def stack(cls, trials_list):
        """Return a RateTrials object containing all trials.

        The stimulus_set of each SimilarityTrials object is padded
        first to match the maximum number of stimuli across all the
        objects.

        Arguments:
            trials_list: A tuple of RateTrials objects to be stacked.

        Returns:
            A new RateTrials object.

        """
        # Determine the maximum number of stimuli present.
        max_n_present = 0
        for i_trials in trials_list:
            if i_trials.max_n_present > max_n_present:
                max_n_present = i_trials.max_n_present

        # Grab relevant information from first entry in list.
        n_pad = max_n_present - trials_list[0].max_n_present
        pad_width = ((0, 0), (0, n_pad))
        stimulus_set = np.pad(
            trials_list[0].stimulus_set,
            pad_width, mode='constant', constant_values=-1
        )

        for i_trials in trials_list[1:]:
            n_pad = max_n_present - i_trials.max_n_present
            pad_width = ((0, 0), (0, n_pad))
            curr_stimulus_set = np.pad(
                i_trials.stimulus_set,
                pad_width, mode='constant', constant_values=-1
            )
            stimulus_set = np.vstack((stimulus_set, curr_stimulus_set))

        trials_stacked = RateDocket(stimulus_set)
        return trials_stacked

    @classmethod
    def load(cls, filepath):
        """Load trials.

        Arguments:
            filepath: The location of the hdf5 file to load.

        """
        f = h5py.File(filepath, "r")
        stimulus_set = f["stimulus_set"][()]
        f.close()
        trials = RateDocket(stimulus_set)
        return trials
