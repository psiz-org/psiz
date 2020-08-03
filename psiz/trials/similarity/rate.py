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
    RateTrials: Abstract base class for 'Rate' trials.
    RateDocket: Unjudged 'Rate' trials.
    RateObservations: Judged 'Rate' trials.

"""

from abc import ABCMeta, abstractmethod
import copy
import warnings

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from psiz.trials.similarity.base import SimilarityTrials


class RateTrials(SimilarityTrials, metaclass=ABCMeta):
    """Abstract base class for rank-type trials."""

    def __init__(self, stimulus_set):
        """Initialize.

        Arguments:
            stimulus_set: An integer matrix containing indices that
                indicate the set of stimuli used in each trial. Each
                row indicates the stimuli used in one trial. The first
                column is the query stimulus. The remaining columns
                indicate reference stimuli. It is assumed that stimuli
                indices are composed of integers from [0, N-1], where N
                is the number of unique stimuli. The value -1 can be
                used as a placeholder for non-existent references.
                shape = (n_trial, max(n_reference) + 1)
            n_select (optional): An integer array indicating the number
                of references selected in each trial. Values must be
                greater than zero but less than the number of
                references for the corresponding trial.
                shape = n_trial,)
            is_ranked (optional): A Boolean array indicating which
                trials require reference selections to be ranked.
                shape = (n_trial,)

        """
        SimilarityTrials.__init__(self, stimulus_set)

        n_present = self._infer_n_present(stimulus_set)
        self.n_present = self._check_n_present(n_reference)

        # Format stimulus set.
        self.max_n_reference = np.amax(self.n_reference)
        self.stimulus_set = self.stimulus_set[:, 0:self.max_n_reference+1]

    def _check_n_present(self, n_present):
        """Check the argument `n_reference`.

        Raises:
            ValueError

        """
        if np.sum(np.less(n_present, 2)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must contain at least two "
                "non-negative integers per a row, i.e., at least "
                "two stimuli per trial."))
        return n_present


class RateDocket(RateTrials):
    """Object that encapsulates unseen trials.

    The attributes and behavior of RateDocket is largely inherited
    from RateTrials.

    Attributes:
        n_trial: An integer indicating the number of trials.
        stimulus_set: An integer matrix containing indices that
            indicate the set of stimuli used in each trial. Each row
            indicates the stimuli used in one trial. The first column
            is the query stimulus. The remaining, columns indicate
            reference stimuli. Negative integers are used as
            placeholders to indicate non-existent references.
            shape = (n_trial, max(n_reference) + 1)
        config_idx: An integer array indicating the
            configuration of each trial. The integer is an index
            referencing the row of config_list and the element of
            outcome_idx_list.
            shape = (n_trial,)
        config_list: A DataFrame object describing the unique trial
            configurations. The columns are 'n_reference',
            'n_select', 'is_ranked'. and 'n_outcome'.

    Notes:
        stimulus_set: The order of the reference stimuli is
            unimportant.
        Unique configurations and configuration IDs are determined by
            'n_present'.

    Methods:
        save: Save the Docket object to disk.
        subset: Return a subset of unjudged trials given an index.

    """

    def __init__(self, stimulus_set):
        """Initialize.

        Arguments:
            stimulus_set: The order of the reference indices is not
                important. See SimilarityTrials.
            n_select (optional): See SimilarityTrials.
            is_ranked (optional): See SimilarityTrials.

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
        n_trial = len(n_reference)

        # Determine unique display configurations.
        n_outcome_placeholder = np.zeros(n_select.shape[0], dtype=np.int32)
        d = {'n_present': n_present}
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)

        # Assign display configuration ID for every observation.
        config_idx = np.empty(n_trial, dtype=np.int32)
        outcome_idx_list = []
        for i_config in range(n_config):
            # Find trials matching configuration.
            a = (n_present == df_config['n_present'].iloc[i_config])
            f = np.array((a))
            display_type_locs = np.all(f, axis=0)
            config_idx[display_type_locs] = i_config

        self.config_idx = config_idx
        self.config_list = df_config

    def save(self, filepath):
        """Save the RateDocket object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        f = h5py.File(filepath, "w")
        f.create_dataset("trial_type", data="RateDocket")
        f.create_dataset("stimulus_set", data=self.stimulus_set)
        f.close()

    # TODO
    def as_dataset(self, group, all_outcomes=True):
        """Return TensorFlow dataset.

        Arguments:
            group: ND array indicating group membership information for
                each trial.
            all_outcomes (optional): Boolean indicating whether all
                possible outcomes (along third dimension) should be
                included in returned dataset.

        Returns:
            x: A TensorFlow dataset.

        """
        if group.ndim == 1:
            group = np.expand_dims(group, axis=1)
        group_level_0 = np.zeros([group.shape[0], 1], dtype=np.int32)
        group = np.hstack([group_level_0, group])
        # Return tensorflow dataset.
        if all_outcomes:
            stimulus_set = self.all_outcomes()
            x = {
                'stimulus_set': tf.constant(
                    stimulus_set + 1, dtype=tf.int32
                ),
                'is_select': tf.constant(
                    np.expand_dims(self.is_select(compress=False), axis=2),
                    dtype=tf.bool
                ),
                'group': tf.constant(group, dtype=tf.int32)
            }
        else:
            stimulus_set = np.expand_dims(self.stimulus_set + 1, axis=2)
            x = {
                'stimulus_set': tf.constant(stimulus_set, dtype=tf.int32),
                'is_select': tf.constant(
                    np.expand_dims(self.is_select(compress=False), axis=2),
                    dtype=tf.bool
                ),
                'group': tf.constant(group, dtype=tf.int32)
            }
        return x


class RateObservations(RateTrials):
    """Object that encapsulates seen trials.

    The attributes and behavior of RateObservatiosn are largely inherited
    from RateTrials.

    Attributes:
        n_trial: An integer indicating the number of trials.
        stimulus_set: An integer matrix containing indices that
            indicate the set of stimuli used in each trial. Each row
            indicates the stimuli used in one trial. The first column
            is the query stimulus. The remaining, columns indicate
            reference stimuli. Negative integers are used as
            placeholders to indicate non-existent references.
            shape = (n_trial, max(n_reference) + 1)
        n_present: An integer array indicating the number of
            stimuli present in each trial.
            shape = (n_trial,)
        config_idx: An integer array indicating the
            configuration of each trial. The integer is an index
            referencing the row of config_list and the element of
            outcome_idx_list.
            shape = (n_trial,)
        config_list: A DataFrame object describing the unique trial
            configurations.
        group_id: An integer array indicating the group membership of
            each trial. It is assumed that group_id is composed of
            integers from [0, M-1] where M is the total number of
            groups.
            shape = (n_trial,)
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
        Unique configurations and configuration IDs are determined by
            'group_id' in addition to the usual 'n_present'.

    Methods:
        subset: Return a subset of judged trials given an index.
        set_group_id: Override the group ID of all trials.
        set_weight: Override the weight of all trials.
        save: Save the observations data structure to disk.

    """

    def __init__(
            self, stimulus_set, group_id=None, agent_id=None, session_id=None,
            weight=None, rt_ms=None):
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
            group_id (optional): An integer array indicating the group
                membership of each trial. It is assumed that group_id
                is composed of integers from [0, M-1] where M is the
                total number of groups.
                shape = (n_trial,)
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
        RateTrials.__init__(self, stimulus_set)

        # Handle default settings.
        if group_id is None:
            group_id = np.zeros((self.n_trial), dtype=np.int32)
        else:
            group_id = self._check_group_id(group_id)
        self.group_id = group_id

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
        self._set_configuration_data(self.n_present, group_id)

    def _check_group_id(self, group_id):
        """Check the argument group_id."""
        group_id = group_id.astype(np.int32)
        # Check shape agreement.
        if not (group_id.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'group_id' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        # Check lowerbound support limit.
        bad_locs = group_id < 0
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The parameter 'group_id' contains integers less than 0. "
                "Found {0} bad trial(s).").format(n_bad))
        return group_id

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
        """Return subset of trials as a new RateObservatiosn object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new RateObservatiosn object.

        """
        return RateObservatiosn(
            self.stimulus_set[index, :], n_select=self.n_select[index],
            is_ranked=self.is_ranked[index], group_id=self.group_id[index],
            agent_id=self.agent_id[index], session_id=self.session_id[index],
            weight=self.weight[index], rt_ms=self.rt_ms[index]
        )

    def _set_configuration_data(self, n_present, group_id, session_id=None):
        """Generate a unique ID for each trial configuration.

        Helper function that generates a unique ID for each of the
        unique trial configurations in the provided data set.

        Arguments:
            n_present: An integer array indicating the number of
                stimuli present in each trial.
                shape = (n_trial,)
            group_id: An integer array indicating the group membership
                of each trial. It is assumed that group is composed of
                integers from [0, M-1] where M is the total number of
                groups. Separate attention weights are inferred for
                each group.
                shape = (n_trial,)
            session_id: An integer array indicating the session ID of
                a trial. It is assumed that observations with the same
                session ID were judged by a single agent. A single
                agent may have completed multiple sessions.
                shape = (n_trial,)

        Notes:
            Sets two attributes of object.
            config_idx: A unique index for each type of trial
                configuration.
            config_list: A DataFrame containing all the unique
                trial configurations.

        """
        n_trial = len(n_reference)

        if session_id is None:
            session_id = np.zeros((n_trial), dtype=np.int32)

        # Determine unique display configurations.
        n_outcome_placeholder = np.zeros(n_select.shape[0], dtype=np.int32)
        d = {
            'n_present': n_present, 'group_id': group_id,
            'session_id': session_id
            }
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)
        n_out_idx = df_config.columns.get_loc("n_outcome")

        # Assign display configuration index for every observation.
        config_idx = np.empty(n_trial, dtype=np.int32)
        outcome_idx_list = []
        for i_config in range(n_config):
            # Find trials matching configuration.
            a = (n_present == df_config['n_prsent'].iloc[i_config])
            d = (group_id == df_config['group_id'].iloc[i_config])
            e = (session_id == df_config['session_id'].iloc[i_config])
            f = np.array((a, d, e))
            display_type_locs = np.all(f, axis=0)
            config_idx[display_type_locs] = i_config

        self.config_idx = config_idx
        self.config_list = df_config

    def set_group_id(self, group_id):
        """Override the existing group_ids.

        Arguments:
            group_id: The new group IDs. Can be an integer or an array
                of integers with shape=(self.n_trial,).

        """
        if np.isscalar(group_id):
            group_id = group_id * np.ones((self.n_trial), dtype=np.int32)
        else:
            group_id = self._check_group_id(group_id)
        self.group_id = copy.copy(group_id)

        # Re-derive unique display configurations.
        self._set_configuration_data(self.m_present, group_id)

    def set_weight(self, weight):
        """Override the existing group_ids.

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
        """Save the RateObservatiosn object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        f = h5py.File(filepath, "w")
        f.create_dataset("trial_type", data="RateObservatiosn")
        f.create_dataset("stimulus_set", data=self.stimulus_set)
        f.create_dataset("group_id", data=self.group_id)
        f.create_dataset("agent_id", data=self.agent_id)
        f.create_dataset("session_id", data=self.session_id)
        f.create_dataset("weight", data=self.weight)
        f.create_dataset("rt_ms", data=self.rt_ms)
        f.close()

    # TODO
    def as_dataset(self, all_outcomes=True):
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
        group_level_0 = np.zeros([self.group_id.shape[0]], dtype=np.int32)
        if all_outcomes:
            stimulus_set = self.all_outcomes()
            x = {
                'stimulus_set': stimulus_set + 1,
                'is_select': np.expand_dims(
                    self.is_select(compress=False), axis=2
                ),
                'group': np.stack(
                    (group_level_0, self.group_id, self.agent_id), axis=-1
                )
            }
            # NOTE: The outputs `y` indicate a one-hot encoding of the outcome
            # that occurred.
            y = np.zeros([self.n_trial, stimulus_set.shape[2]])
            y[:, 0] = 1
        else:
            x = {
                'stimulus_set': np.expand_dims(self.stimulus_set + 1, axis=2),
                'is_select': np.expand_dims(
                    self.is_select(compress=False), axis=2
                ),
                'group': np.stack(
                    (group_level_0, self.group_id, self.agent_id), axis=-1
                )
            }
            # NOTE: The outputs `y` indicate a sparse encoding of the outcome
            # that occurred.
            y = np.zeros([self.n_trial])

        y = tf.constant(y, dtype=K.floatx())

        # Observation weight.
        w = tf.constant(self.weight, dtype=K.floatx())

        # Create dataset.
        ds_obs = tf.data.Dataset.from_tensor_slices((x, y, w))
        return ds_obs

