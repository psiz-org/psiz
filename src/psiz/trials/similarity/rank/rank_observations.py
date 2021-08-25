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
    RankObservations: Judged 'Rank' trials.

"""

import copy

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from psiz.trials.similarity.rank.rank_trials import RankTrials


class RankObservations(RankTrials):
    """Object that encapsulates seen trials.

    The attributes and behavior of RankObservations are largely inherited
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
            configurations.
        outcome_idx_list: A list of 2D arrays indicating all possible
            outcomes for a trial configuration. Each element in the
            list corresponds to a trial configuration in config_list.
            Each row of the 2D array indicates one potential outcome.
            The values in the rows are the indices of the the reference
            stimuli (as specified in the attribute `stimulus_set`.
        groups: An integer 2D array indicating the group membership
            of each trial. It is assumed that `groups` is composed of
            integers from [0, M-1] where M is the total number of
            groups for a particular column.
            shape = (n_trial, n_col)
        agent_id: An integer array indicating the agent ID of a trial.
            It is assumed that all IDs are non-negative and that
            observations with the same agent ID were judged by a single
            agent.
            shape = (n_trial,)
        session_id: An integer array indicating the session ID of a
            trial. It is assumed that all IDs are non-negative. Trials
            with different session IDs were obtained during different
            sessions.
            shape = (n_trial,)
        weight: An float array indicating the inference weight of each
            trial.
            shape = (n_trial,)
        rt_ms: An array indicating the response time (in milliseconds)
            of the agent for each trial.

    Notes:
        stimulus_set: The order of the reference stimuli is important.
            As usual, the the first column contains indices indicating
            query stimulus. The remaining columns contain indices
            indicating the reference stimuli. An agent's selected
            references are listed first (in order of selection if the
            trial is ranked) and remaining unselected references are
            listed in any order.
        Unique configurations and configuration IDs are determined by
            'groups' in addition to the usual 'n_reference',
            'n_select', and 'is_ranked' variables.

    Methods:
        subset: Return a subset of judged trials given an index.
        set_groups: Override the group ID of all trials.
        set_weight: Override the weight of all trials.
        save: Save the observations data structure to disk.

    """

    def __init__(self, stimulus_set, n_select=None, is_ranked=None,
                 groups=None, agent_id=None, session_id=None, weight=None,
                 rt_ms=None):
        """Initialize.

        Extends initialization of SimilarityTrials.

        Arguments:
            stimulus_set: The order of reference indices is important.
                An agent's selected references are listed first (in
                order of selection if the trial is ranked) and
                remaining unselected references are listed in any
                order. See SimilarityTrials.
            n_select (optional): See SimilarityTrials.
            is_ranked (optional): See SimilarityTrials.
            groups (optional): An integer 2D array indicating the
                group membership of each trial. It is assumed that
                `groups` is composed of integers from [0, M-1] where
                M is the total number of groups.
                shape = (n_trial, n_col)
            agent_id: An integer array indicating the agent ID of a
                trial. It is assumed that all IDs are non-negative and
                that observations with the same agent ID were judged by
                a single agent.
                shape = (n_trial,)
            session_id: An integer array indicating the session ID of a
                trial. It is assumed that all IDs are non-negative.
                Trials with different session IDs were obtained during
                different sessions.
                shape = (n_trial,)
            weight (optional): A float array indicating the inference
                weight of each trial.
                shape = (n_trial,1)
            rt_ms(optional): An array indicating the response time (in
                milliseconds) of the agent for each trial.
                shape = (n_trial,1)

        """
        RankTrials.__init__(self, stimulus_set, n_select, is_ranked)

        # Handle default settings.
        if groups is None:
            groups = np.zeros([self.n_trial, 1], dtype=np.int32)
        else:
            groups = self._check_groups(groups)
        self.groups = groups

        if agent_id is None:
            agent_id = np.zeros((self.n_trial), dtype=np.int32)
        else:
            agent_id = self._check_agent_id(agent_id)
        self.agent_id = agent_id

        if session_id is None:
            session_id = np.zeros((self.n_trial), dtype=np.int32)
        else:
            session_id = self._check_session_id(session_id)
        self.session_id = session_id

        if weight is None:
            weight = np.ones((self.n_trial))
        else:
            weight = self._check_weight(weight)
        self.weight = weight

        if rt_ms is None:
            rt_ms = -np.ones((self.n_trial))
        else:
            rt_ms = self._check_rt(rt_ms)
        self.rt_ms = rt_ms

        # Determine unique display configurations.
        self._set_configuration_data(
            self.n_reference, self.n_select, self.is_ranked, self.groups
        )

    def _check_agent_id(self, agent_id):
        """Check the argument agent_id."""
        agent_id = agent_id.astype(np.int32)
        # Check shape agreement.
        if not (agent_id.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'agent_id' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        # Check lowerbound support limit.
        bad_locs = agent_id < 0
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The parameter 'agent_id' contains integers less than 0. "
                "Found {0} bad trial(s).").format(n_bad))
        return agent_id

    def _check_session_id(self, session_id):
        """Check the argument session_id."""
        session_id = session_id.astype(np.int32)
        # Check shape agreement.
        if not (session_id.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'session_id' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        # Check lowerbound support limit.
        bad_locs = session_id < 0
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The parameter 'session_id' contains integers less than 0. "
                "Found {0} bad trial(s).").format(n_bad))
        return session_id

    def _check_weight(self, weight):
        """Check the argument weight."""
        weight = weight.astype(np.float)
        # Check shape agreement.
        if not (weight.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'weight' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        return weight

    def _check_rt(self, rt_ms):
        """Check the argument rt_ms."""
        rt_ms = rt_ms.astype(np.float)
        # Check shape agreement.
        if not (rt_ms.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'rt_ms' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        return rt_ms

    def subset(self, index):
        """Return subset of trials as a new RankObservations object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new RankObservations object.

        """
        return RankObservations(
            self.stimulus_set[index, :], n_select=self.n_select[index],
            is_ranked=self.is_ranked[index], groups=self.groups[index],
            agent_id=self.agent_id[index], session_id=self.session_id[index],
            weight=self.weight[index], rt_ms=self.rt_ms[index]
        )

    def _set_configuration_data(
            self, n_reference, n_select, is_ranked, groups,
            session_id=None):
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
            groups:
                shape = (n_trial,)
            session_id: An integer array indicating the session ID of
                a trial. It is assumed that observations with the same
                session ID were judged by a single agent. A single
                agent may have completed multiple sessions.
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

        if session_id is None:
            session_id = np.zeros((n_trial), dtype=np.int32)

        # Determine unique display configurations by create a dictionary on
        # which to determine distinct configurations.
        d = {
            'n_reference': n_reference, 'n_select': n_select,
            'is_ranked': is_ranked, 'session_id': session_id
        }
        d_groups = self._split_groups_columns(groups)
        d.update(d_groups)
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)

        # Loop over distinct configurations in order to determine
        # configuration index for all trials.
        n_outcome = np.zeros(n_config, dtype=np.int32)

        # Assign display configuration index for every observation.
        config_idx = np.zeros(n_trial, dtype=np.int32)
        outcome_idx_list = []
        for i_config in range(n_config):
            # Determine number of possible outcomes for configuration.
            outcome_idx = self._possible_rank_outcomes(
                df_config.iloc[i_config]
            )
            outcome_idx_list.append(outcome_idx)
            n_outcome[i_config] = outcome_idx.shape[0]

            bidx = self._find_trials_matching_config(df_config.iloc[i_config])
            config_idx[bidx] = i_config

        # Add n_outcome to `df_config`.
        df_config['n_outcome'] = n_outcome

        self.config_idx = config_idx
        self.config_list = df_config
        self.outcome_idx_list = outcome_idx_list

    def set_groups(self, groups):
        """Override the existing groups.

        Arguments:
            groups: The new group IDs.
                shape=(n_trial, n_col)

        """
        groups = self._check_groups(groups)
        self.groups = copy.copy(groups)

        # Re-derive unique display configurations.
        self._set_configuration_data(
            self.n_reference, self.n_select, self.is_ranked, groups
        )

    def set_weight(self, weight):
        """Override the existing weights.

        Arguments:
            weight: The new weight. Can be an float or an array
                of floats with shape=(self.n_trial,).

        """
        if np.isscalar(weight):
            weight = weight * np.ones((self.n_trial), dtype=np.int32)
        else:
            weight = self._check_weight(weight)
        self.weight = copy.copy(weight)

    def save(self, filepath):
        """Save the RankObservations object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        f = h5py.File(filepath, "w")
        f.create_dataset("class_name", data="RankObservations")
        f.create_dataset("stimulus_set", data=self.stimulus_set)
        f.create_dataset("n_select", data=self.n_select)
        f.create_dataset("is_ranked", data=self.is_ranked)
        f.create_dataset("groups", data=self.groups)
        f.create_dataset("agent_id", data=self.agent_id)
        f.create_dataset("session_id", data=self.session_id)
        f.create_dataset("weight", data=self.weight)
        f.create_dataset("rt_ms", data=self.rt_ms)
        f.close()

    def as_dataset(self):
        """Format necessary data as Tensorflow.data.Dataset object.

        Returns:
            ds_obs: The data necessary for inference, formatted as a
            tf.data.Dataset object.

        """
        # NOTE: Should not use single dimension inputs. Add a singleton
        # dimensions if necessary, because restoring a SavedModel adds
        # singleton dimensions on the call signautre for inputs that only
        # have one dimension. Add the singleton dimensions here solves the
        # problem.
        # NOTE: We use stimulus_set + 1, since TensorFlow requires "0", not
        # "-1" to indicate a masked value.
        # NOTE: The dimensions of inputs are expanded to have an additional
        # singleton third dimension to indicate that there is only one outcome
        # that we are interested for each trial.
        stimulus_set = self.all_outcomes()
        x = {
            'stimulus_set': stimulus_set + 1,
            'is_select': np.expand_dims(
                self.is_select(compress=False), axis=2
            ),
            'groups': self.groups
        }
        # NOTE: The outputs `y` indicate a one-hot encoding of the outcome
        # that occurred.
        y = np.zeros([self.n_trial, stimulus_set.shape[2]])
        y[:, 0] = 1

        y = tf.constant(y, dtype=K.floatx())

        # Observation weight.
        w = tf.constant(self.weight, dtype=K.floatx())

        # Create dataset.
        ds_obs = tf.data.Dataset.from_tensor_slices((x, y, w))
        return ds_obs

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
        try:
            groups = f["groups"][()]
        except KeyError:
            groups = f["group_id"][()]
            # Patch for old saving assumptions.
            # pylint: disable=no-member
            if groups.ndim == 1:
                groups = np.expand_dims(groups, axis=1)

        # For backwards compatability.
        if "weight" in f:
            weight = f["weight"][()]
        else:
            weight = np.ones((len(n_select)))
        if "rt_ms" in f:
            rt_ms = f["rt_ms"][()]
        else:
            rt_ms = -np.ones((len(n_select)))
        if "agent_id" in f:
            agent_id = f["agent_id"][()]
        else:
            agent_id = np.zeros((len(n_select)))
        if "session_id" in f:
            session_id = f["session_id"][()]
        else:
            session_id = np.zeros((len(n_select)))
        f.close()

        trials = RankObservations(
            stimulus_set, n_select=n_select, is_ranked=is_ranked,
            groups=groups, agent_id=agent_id, session_id=session_id,
            weight=weight, rt_ms=rt_ms
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
        groups = trials_list[0].groups
        agent_id = trials_list[0].agent_id
        session_id = trials_list[0].session_id
        weight = trials_list[0].weight
        rt_ms = trials_list[0].rt_ms

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
            groups = np.vstack((groups, i_trials.groups))
            agent_id = np.hstack((agent_id, i_trials.agent_id))
            session_id = np.hstack((session_id, i_trials.session_id))
            weight = np.hstack((weight, i_trials.weight))
            rt_ms = np.hstack((rt_ms, i_trials.rt_ms))

        trials_stacked = RankObservations(
            stimulus_set, n_select=n_select, is_ranked=is_ranked,
            groups=groups, agent_id=agent_id, session_id=session_id,
            weight=weight, rt_ms=rt_ms
        )
        return trials_stacked
