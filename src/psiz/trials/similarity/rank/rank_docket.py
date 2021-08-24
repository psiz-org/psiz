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
"""Rank trials module.

On each similarity judgment trial, an agent judges the similarity
between a single query stimulus and multiple reference stimuli.

Classes:
    RankDocket: Unjudged 'Rank' trials.

"""

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from psiz.trials.similarity.rank.rank_trials import RankTrials


class RankDocket(RankTrials):
    """Object that encapsulates unseen trials.

    The attributes and behavior of RankDocket is largely inherited
    from RankTrials.

    Attributes:
        n_trial: An integer indicating the number of trials.
        stimulus_set: An integer matrix containing indices that
            indicate the set of stimuli used in each trial. Each row
            indicates the stimuli used in one trial. The first column
            is the query stimulus. The remaining, columns indicate
            reference stimuli. Negative integers are used as
            placeholders to indicate non-existent references.
            shape = (n_trial, max(n_reference) + 1)
        n_reference: An integer array indicating the number of
            references in each trial.
            shape = (n_trial,)
        n_select: An integer array indicating the number of references
            selected in each trial.
            shape = (n_trial,)
        is_ranked: A Boolean array indicating which trials require
            reference selections to be ranked.
            shape = (n_trial,)
        config_idx: An integer array indicating the
            configuration of each trial. The integer is an index
            referencing the row of config_list and the element of
            outcome_idx_list.
            shape = (n_trial,)
        config_list: A DataFrame object describing the unique trial
            configurations. The columns are 'n_reference',
            'n_select', 'is_ranked'. and 'n_outcome'.
        outcome_idx_list: A list of 2D arrays indicating all possible
            outcomes for a trial configuration. Each element in the
            list corresponds to a trial configuration in config_list.
            Each row of the 2D array indicates one potential outcome.
            The values in the rows are the indices of the the reference
            stimuli (as specified in the attribute `stimulus_set`.

    Notes:
        stimulus_set: The order of the reference stimuli is
            unimportant. As usual, the the first column contains
            indices indicating query stimulus. The remaining columns
            contain indices indicating the reference stimuli in any
            order.
        Unique configurations and configuration IDs are determined by
            'n_reference', 'n_select', and 'is_ranked'.

    Methods:
        save: Save the Docket object to disk.
        subset: Return a subset of unjudged trials given an index.

    """

    def __init__(self, stimulus_set, n_select=None, is_ranked=None):
        """Initialize.

        Arguments:
            stimulus_set: The order of the reference indices is not
                important. See SimilarityTrials.
            n_select (optional): See SimilarityTrials.
            is_ranked (optional): See SimilarityTrials.

        """
        RankTrials.__init__(self, stimulus_set, n_select, is_ranked)

        # Determine unique display configurations.
        self._set_configuration_data(
            self.n_reference, self.n_select, self.is_ranked
        )

    def subset(self, index):
        """Return subset of trials as a new RankDocket object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new RankDocket object.

        """
        return RankDocket(
            self.stimulus_set[index, :], self.n_select[index],
            self.is_ranked[index]
        )

    def _set_configuration_data(self, n_reference, n_select, is_ranked):
        """Generate a unique ID for each trial configuration.

        Helper function that generates a unique ID for each of the
        unique trial configurations in the provided data set.

        Arguments:
            n_reference: An integer array indicating the number of
                references in each trial.
                shape = (n_trial,)
            n_select: An integer array indicating the number of
                references selected in each trial.
                shape = (n_trial,)
            is_ranked:  Boolean array indicating which trials had
                selected references that were ordered.
                shape = (n_trial,)

        Notes:
            Sets three attributes of object.
            config_idx: A unique index for each type of trial
                configuration.
            config_list: A DataFrame containing all the unique
                trial configurations.
            outcome_idx_list: A list of the possible outcomes for each
                trial configuration.

        """
        n_trial = len(n_reference)

        # Determine unique display configurations.
        n_outcome_placeholder = np.zeros(n_select.shape[0], dtype=np.int32)
        d = {
            'n_reference': n_reference, 'n_select': n_select,
            'is_ranked': is_ranked, 'n_outcome': n_outcome_placeholder
        }
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)
        n_out_idx = df_config.columns.get_loc("n_outcome")

        # Assign display configuration ID for every observation.
        config_idx = np.empty(n_trial, dtype=np.int32)
        outcome_idx_list = []
        for i_config in range(n_config):
            # Determine number of possible outcomes for configuration.
            outcome_idx = self._possible_rank_outcomes(
                df_config.iloc[i_config]
            )
            outcome_idx_list.append(outcome_idx)
            n_outcome = outcome_idx.shape[0]
            df_config.iloc[i_config, n_out_idx] = n_outcome
            # Find trials matching configuration.
            a = (n_reference == df_config['n_reference'].iloc[i_config])
            b = (n_select == df_config['n_select'].iloc[i_config])
            c = (is_ranked == df_config['is_ranked'].iloc[i_config])
            f = np.array((a, b, c))
            display_type_locs = np.all(f, axis=0)
            config_idx[display_type_locs] = i_config

        self.config_idx = config_idx
        self.config_list = df_config
        self.outcome_idx_list = outcome_idx_list

    def save(self, filepath):
        """Save the RankDocket object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        f = h5py.File(filepath, "w")
        f.create_dataset("class_name", data="RankDocket")
        f.create_dataset("stimulus_set", data=self.stimulus_set)
        f.create_dataset("n_select", data=self.n_select)
        f.create_dataset("is_ranked", data=self.is_ranked)
        f.close()

    def as_dataset(self, groups):
        """Return TensorFlow dataset.

        Arguments:
            groups: ND array indicating group membership information for
                each trial.

        Returns:
            x: A TensorFlow dataset.

        """
        if groups is None:
            groups = np.zeros([self.n_trial, 1], dtype=np.int32)
        else:
            groups = self._check_groups(groups)

        # Return tensorflow dataset.
        stimulus_set = self.all_outcomes()
        x = {
            'stimulus_set': tf.constant(
                stimulus_set + 1, dtype=tf.int32
            ),
            'is_select': tf.constant(
                np.expand_dims(self.is_select(compress=False), axis=2),
                dtype=tf.bool
            ),
            'groups': tf.constant(groups, dtype=tf.int32)
        }
        return tf.data.Dataset.from_tensor_slices((x))

    @classmethod
    def load(cls, filepath):
        """Load trials.

        Arguments:
            filepath: The location of the hdf5 file to load.

        """
        f = h5py.File(filepath, "r")
        stimulus_set = f["stimulus_set"][()]
        n_select = f["n_select"][()]
        is_ranked = f["is_ranked"][()]
        f.close()
        trials = RankDocket(
            stimulus_set, n_select=n_select, is_ranked=is_ranked
        )
        return trials

    @classmethod
    def stack(cls, trials_list):
        """Return a RankTrials object containing all trials.

        The stimulus_set of each SimilarityTrials object is padded first to
        match the maximum number of references of all the objects.

        Arguments:
            trials_list: A tuple of RankTrials objects to be stacked.

        Returns:
            A new RankTrials object.

        """
        # Determine the maximum number of references.
        max_n_reference = 0
        for i_trials in trials_list:
            if i_trials.max_n_reference > max_n_reference:
                max_n_reference = i_trials.max_n_reference

        # Grab relevant information from first entry in list.
        n_pad = max_n_reference - trials_list[0].max_n_reference
        pad_width = ((0, 0), (0, n_pad))
        stimulus_set = np.pad(
            trials_list[0].stimulus_set,
            pad_width, mode='constant', constant_values=-1
        )

        n_select = trials_list[0].n_select
        is_ranked = trials_list[0].is_ranked

        for i_trials in trials_list[1:]:
            n_pad = max_n_reference - i_trials.max_n_reference
            pad_width = ((0, 0), (0, n_pad))
            curr_stimulus_set = np.pad(
                i_trials.stimulus_set,
                pad_width, mode='constant', constant_values=-1
            )
            stimulus_set = np.vstack((stimulus_set, curr_stimulus_set))
            n_select = np.hstack((n_select, i_trials.n_select))
            is_ranked = np.hstack((is_ranked, i_trials.is_ranked))

        trials_stacked = RankDocket(
            stimulus_set, n_select, is_ranked
        )
        return trials_stacked
