# -*- coding: utf-8 -*-
# Copyright 2019 The PsiZ Authors. All Rights Reserved.
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
# ==============================================================================

"""Module for similarity judgment trials.

Classes:
    SimilarityTrials: Abstract class for similarity judgment trials.
    Docket: Unjudged similarity judgment trials.
    Observations: Similarity judgment trials that have been judged and
        will serve as observed data during inference.

Functions:
    stack: Combine a list of multiple SimilarityTrial objects into one.
    squeeze: Squeeze indices to be small and consecutive.
    load_trials: Load a hdf5 file, saved using the `save` class method,
        as a SimilarityTrial object.

Notes:
    On each similarity judgment trial, an agent judges the similarity
        between a single query stimulus and multiple reference stimuli.
    Groups are used to identify distinct populations of agents. For
        example, similarity judgments could be collected from two
        groups: novices and experts. During inference, group
        information can be used to infer a separate set of attention
        weights for each group while sharing all other parameters.

Todo:
    - MAYBE restructure group_id and agent_id. If we wanted to allow
    for arbitrary hierarchical models, maybe better off making
    group_id a 2D array of shape=(n_trial, n_group_level)
    - MAYBE make config_list a custom object

"""

from abc import ABCMeta, abstractmethod
from itertools import permutations
import copy
import warnings

import h5py
import numpy as np
import pandas as pd


class SimilarityTrials(object):
    """Abstract base class for similarity judgment trials.

    This abstract base class is used to organize data associated with
    similarity judgment trials. As the class name suggests, this object
    handles data associated with multiple trials. Depending on the
    concrete subclass, the similarity trials represent unjudged trials
    or judged trials.

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

    Methods:
        subset: Return a subset of similarity trials given an index.

    """

    __metaclass__ = ABCMeta

    def __init__(self, stimulus_set, n_select=None, is_ranked=None):
        """Initialize.

        Arguments:
            stimulus_set: An integer matrix containing indices that
                indicate the set of stimuli used in each trial. Each
                row indicates the stimuli used in one trial. The first
                column is the query stimulus. The remaining, columns
                indicate reference stimuli. It is assumed that stimuli
                indices are composed of integers from [0, N-1], where N
                is the number of unique stimuli. Negative integers are
                used as placeholders to indicate non-existent
                references.
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
        stimulus_set = self._check_stimulus_set(stimulus_set)

        self.n_trial = stimulus_set.shape[0]
        n_reference = self._infer_n_reference(stimulus_set)
        self.n_reference = self._check_n_reference(n_reference)

        # Format stimulus set.
        self.max_n_reference = np.amax(self.n_reference)
        self.stimulus_set = stimulus_set[:, 0:self.max_n_reference+1]

        if n_select is None:
            n_select = np.ones((self.n_trial), dtype=np.int32)
        else:
            n_select = self._check_n_select(n_select)
        self.n_select = n_select

        if is_ranked is None:
            is_ranked = np.full((self.n_trial), True)
        else:
            is_ranked = self._check_is_ranked(is_ranked)
        self.is_ranked = is_ranked

        # Attributes determined by concrete class.
        self.config_idx = None
        self.config_list = None
        self.outcome_idx_list = None

    def _check_stimulus_set(self, stimulus_set):
        """Check the argument `stimulus_set`.

        Raises:
            ValueError

        """
        if not issubclass(stimulus_set.dtype.type, np.integer):
            raise ValueError((
                "The argument `stimulus_set` must be a 2D array of "
                "integers."))
        if np.sum(np.less(stimulus_set, -1)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must only contain integers "
                "greater than or equal to -1."))
        return stimulus_set

    def _infer_n_reference(self, stimulus_set):
        """Return the number of references in each trial.

        Infers the number of available references for each trial. The
        function assumes that values less than zero, are placeholder
        values and should be treated as non-existent.

        Arguments:
            stimulus_set: shape = [n_trial, 1]

        Returns:
            n_reference: An integer array indicating the number of
                references in each trial.
                shape = [n_trial, 1]

        """
        max_ref = stimulus_set.shape[1] - 1
        n_reference = max_ref - np.sum(stimulus_set < 0, axis=1)
        return n_reference.astype(dtype=np.int32)

    def _check_n_reference(self, n_reference):
        """Check the argument `n_reference`.

        Raises:
            ValueError

        """
        if np.sum(np.less(n_reference, 2)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must contain at least three "
                "non-negative integers per a row, i.e. one query and at least "
                "two reference stimuli per trial."))
        return n_reference

    def _check_n_select(self, n_select):
        """Check the argument `n_select`.

        Raises:
            ValueError

        """
        n_select = n_select.astype(np.int32)
        # Check shape argreement.
        if not (n_select.shape[0] == self.n_trial):
            raise ValueError((
                "The argument `n_select` must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        # Check lowerbound support limit.
        bad_locs = n_select < 1
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The argument `n_select` contains integers less than 1. "
                "Found {0} bad trial(s).").format(n_bad))
        # Check upperbound support limit.
        bad_locs = np.greater_equal(n_select, self.n_reference)
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The argument `n_select` contains integers greater than "
                "or equal to the corresponding 'n_reference'. Found {0} bad "
                "trial(s).").format(n_bad))
        return n_select

    def _check_is_ranked(self, is_ranked):
        """Check the argument `is_ranked`.

        Raises:
            ValueError

        """
        if not (is_ranked.shape[0] == self.n_trial):
            raise ValueError((
                "The argument `n_select` must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        bad_locs = np.not_equal(is_ranked, True)
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The unranked version is not implemented, Found {0} bad "
                "trial(s).").format(n_bad))
        return is_ranked

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

    def outcome_tensor(self):
        """Return outcome tensor."""
        # TODO better doc, test
        n_config = self.config_list.shape[0]

        n_outcome_list = self.config_list['n_outcome'].values
        max_n_outcome = np.max(n_outcome_list)

        n_reference_list = self.config_list['n_reference'].values
        max_n_reference = np.max(n_reference_list)

        outcome_tensor = -1 * np.ones(
            (n_config, max_n_outcome, max_n_reference), dtype=np.int32)
        for i_config in range(n_config):
            outcome_tensor[
                i_config,
                0:n_outcome_list[i_config],
                0:n_reference_list[i_config]] = self.outcome_idx_list[i_config]
        return outcome_tensor

    @abstractmethod
    def save(self, filepath):
        """Save the SimilarityTrials object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        pass


class Docket(SimilarityTrials):
    """Object that encapsulates unjudged similarity trials.

    The attributes and behavior of Docket is largely inherited
    from SimilarityTrials.

    Attributes:
        config_list: A DataFrame object describing the unique trial
            configurations. The columns are 'n_reference',
            'n_select', 'is_ranked'. and 'n_outcome'.

    Notes:
        stimulus_set: The order of the reference stimuli is
            unimportant. As usual, the the first column contains
            indices indicating query stimulus. The remaining columns
            contain indices indicating the reference stimuli in any
            order.
        Unique configurations and configuration IDs are determined by
            'n_reference', 'n_select', and 'is_ranked'.

    Methods:
        subset: Return a subset of unjudged trials given an index.

    """

    def __init__(self, stimulus_set, n_select=None, is_ranked=None):
        """Initialize.

        Extends initialization of SimilarityTrials.

        Arguments:
            stimulus_set: The order of the reference indices is not
                important. See SimilarityTrials.
            n_select (optional): See SimilarityTrials.
            is_ranked (optional): See SimilarityTrials.
        """
        SimilarityTrials.__init__(self, stimulus_set, n_select, is_ranked)

        # Determine unique display configurations.
        self._set_configuration_data(
            self.n_reference, self.n_select, self.is_ranked)

    def subset(self, index):
        """Return subset of trials as a new Docket object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new Docket object.

        """
        return Docket(
            self.stimulus_set[index, :], self.n_select[index],
            self.is_ranked[index])

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
            outcome_idx = _possible_outcomes(df_config.iloc[i_config])
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
        """Save the Docket object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        f = h5py.File(filepath, "w")
        f.create_dataset("trial_type", data="Docket")
        f.create_dataset("stimulus_set", data=self.stimulus_set)
        f.create_dataset("n_select", data=self.n_select)
        f.create_dataset("is_ranked", data=self.is_ranked)
        f.close()


class Observations(SimilarityTrials):
    """Object that encapsulates judged similarity trials.

    The attributes and behavior of Observations are largely inherited
    from SimilarityTrials.

    Attributes:
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
        weight: An float array indicating the inference weight of each
            trial.
            shape = (n_trial,)
        rt_ms: An array indicating the response time (in milliseconds)
            of the agent for each trial.
        session_id: An integer array indicating the session ID of
            a trial. It is assumed that observations with the same
            session ID were judged by a single agent. A single agent
            may have completed multiple sessions.
            shape = (n_trial,) TODO MAYBE

    Notes:
        response_set: The order of the reference stimuli is important.
            As usual, the the first column contains indices indicating
            query stimulus. The remaining columns contain indices
            indicating the reference stimuli. An agent's selected
            references are listed first (in order of selection if the
            trial is ranked) and remaining unselected references are
            listed in any order.
        Unique configurations and configuration IDs are determined by
            'group_id' in addition to the usual 'n_reference',
            'n_select', and 'is_ranked' variables.

    Methods:
        subset: Return a subset of judged trials given an index.
        set_group_id: Override the group ID of all trials.
        set_weight: Override the weight of all trials.
        save: Save the observations data structure to disk.

    """

    def __init__(self, response_set, n_select=None, is_ranked=None,
                 group_id=None, agent_id=None, weight=None, rt_ms=None):
        """Initialize.

        Extends initialization of SimilarityTrials.

        Arguments:
            response_set: The order of reference indices is important.
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
            weight (optional): A float array indicating the inference
                weight of each trial.
                shape = (n_trial,1)
            rt_ms(optional): An array indicating the response time
                (in milliseconds) of the agent for each trial.
                shape = (n_trial,1)
        """
        SimilarityTrials.__init__(self, response_set, n_select, is_ranked)

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
            self.n_reference, self.n_select, self.is_ranked, group_id)

    def _check_group_id(self, group_id):
        """Check the argument group_id."""
        group_id = group_id.astype(np.int32)
        # Check shape argreement.
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
        # Check shape argreement.
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

    def _check_weight(self, weight):
        """Check the argument weight."""
        weight = weight.astype(np.float)
        # Check shape argreement.
        if not (weight.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'weight' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        return weight

    def _check_rt(self, rt_ms):
        """Check the argument rt_ms."""
        rt_ms = rt_ms.astype(np.float)
        # Check shape argreement.
        if not (rt_ms.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'rt_ms' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        return rt_ms

    def subset(self, index):
        """Return subset of trials as a new Observations object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new Observations object.

        """
        return Observations(
            self.stimulus_set[index, :], n_select=self.n_select[index],
            is_ranked=self.is_ranked[index], group_id=self.group_id[index],
            agent_id=self.agent_id[index], weight=self.weight[index],
            rt_ms=self.rt_ms[index]
        )

    def _set_configuration_data(
                self, n_reference, n_select, is_ranked, group_id,
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

        # Determine unique display configurations.
        n_outcome_placeholder = np.zeros(n_select.shape[0], dtype=np.int32)
        d = {
            'n_reference': n_reference, 'n_select': n_select,
            'is_ranked': is_ranked, 'group_id': group_id,
            'session_id': session_id, 'n_outcome': n_outcome_placeholder
            }
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)
        n_out_idx = df_config.columns.get_loc("n_outcome")

        # Assign display configuration index for every observation.
        config_idx = np.empty(n_trial, dtype=np.int32)
        outcome_idx_list = []
        for i_config in range(n_config):
            # Determine number of possible outcomes for configuration.
            outcome_idx = _possible_outcomes(df_config.iloc[i_config])
            outcome_idx_list.append(outcome_idx)
            n_outcome = outcome_idx.shape[0]
            df_config.iloc[i_config, n_out_idx] = n_outcome
            # Find trials matching configuration.
            a = (n_reference == df_config['n_reference'].iloc[i_config])
            b = (n_select == df_config['n_select'].iloc[i_config])
            c = (is_ranked == df_config['is_ranked'].iloc[i_config])
            d = (group_id == df_config['group_id'].iloc[i_config])
            e = (session_id == df_config['session_id'].iloc[i_config])
            f = np.array((a, b, c, d, e))
            display_type_locs = np.all(f, axis=0)
            config_idx[display_type_locs] = i_config

        self.config_idx = config_idx
        self.config_list = df_config
        self.outcome_idx_list = outcome_idx_list

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
        self._set_configuration_data(
            self.n_reference, self.n_select, self.is_ranked, group_id)

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
        """Save the Docket object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        f = h5py.File(filepath, "w")
        f.create_dataset("trial_type", data="Observations")
        f.create_dataset("stimulus_set", data=self.stimulus_set)
        f.create_dataset("n_select", data=self.n_select)
        f.create_dataset("is_ranked", data=self.is_ranked)
        f.create_dataset("group_id", data=self.group_id)
        f.create_dataset("agent_id", data=self.agent_id)
        f.create_dataset("weight", data=self.weight)
        f.create_dataset("rt_ms", data=self.rt_ms)
        f.close()


def stack(trials_list):
        """Return a SimilarityTrials object containing all trials.

        The stimulus_set of each SimilarityTrials object is padded
        first to match the maximum number of references of all the
        objects.

        Arguments:
            trials_list: A tuple of SimilarityTrials objects to be
                stacked.

        Returns:
            A new SimilarityTrials object.

        """
        # Determine the maximum number of references.
        max_n_reference = 0
        for i_trials in trials_list:
            if i_trials.max_n_reference > max_n_reference:
                max_n_reference = i_trials.max_n_reference

        # Grab relevant information from first entry in list.
        stimulus_set = _pad_stimulus_set(
            trials_list[0].stimulus_set,
            max_n_reference
        )
        n_select = trials_list[0].n_select
        is_ranked = trials_list[0].is_ranked
        is_judged = True
        try:
            group_id = trials_list[0].group_id
            agent_id = trials_list[0].agent_id
            weight = trials_list[0].weight
            rt_ms = trials_list[0].rt_ms
        except AttributeError:
            is_judged = False

        for i_trials in trials_list[1:]:
            stimulus_set = np.vstack((
                stimulus_set,
                _pad_stimulus_set(i_trials.stimulus_set, max_n_reference)
            ))
            n_select = np.hstack((n_select, i_trials.n_select))
            is_ranked = np.hstack((is_ranked, i_trials.is_ranked))
            if is_judged:
                group_id = np.hstack((group_id, i_trials.group_id))
                agent_id = np.hstack((agent_id, i_trials.agent_id))
                weight = np.hstack((weight, i_trials.weight))
                rt_ms = np.hstack((rt_ms, i_trials.rt_ms))

        if is_judged:
            trials_stacked = Observations(
                stimulus_set, n_select=n_select, is_ranked=is_ranked,
                group_id=group_id, agent_id=agent_id, weight=weight,
                rt_ms=rt_ms
            )
        else:
            trials_stacked = Docket(
                stimulus_set, n_select, is_ranked)
        return trials_stacked


def squeeze(sim_trials, mode="sg"):
    """Squeeze indices in trials to be small and consecutive.

    Indices are reset to be between 0 and N-1 where N is the number of
    unique stimuli used in sim_trials.

    Arguments:
        sim_trials: A SimilarityTrials object.
        mode (optional): The mode in which to squueze the indices.

    Returns:
        sim_trials_sq: A SimilarityTrials object.

    """
    unique_stimuli = np.unique(sim_trials.stimulus_set)

    # Remove placeholder value from list of unique stimuli indices.
    loc = np.equal(unique_stimuli, -1)
    unique_stimuli = unique_stimuli[np.logical_not(loc)]

    # Squeeze trials.
    sim_trials_sq = copy.deepcopy(sim_trials)
    for new_idx, old_idx in enumerate(unique_stimuli):
        locs = np.equal(sim_trials.stimulus_set, old_idx)
        sim_trials_sq.stimulus_set[locs] = new_idx

    # MAYBE Squeeze groups.
    return sim_trials_sq, unique_stimuli


def load_trials(filepath, verbose=0):
    """Load data saved via the save method.

    The loaded data is instantiated as a concrete class of
    SimilarityTrials.

    Arguments:
        filepath: The location of the hdf5 file to load.
        verbose (optional): Controls the verbosity of printed summary.

    Returns:
        Loaded trials.

    Raises:
        ValueError

    """
    f = h5py.File(filepath, "r")
    # Common attributes.
    trial_type = f["trial_type"][()]
    stimulus_set = f["stimulus_set"][()]
    n_select = f["n_select"][()]
    is_ranked = f["is_ranked"][()]

    if trial_type == "Docket":
        loaded_trials = Docket(
            stimulus_set, n_select=n_select, is_ranked=is_ranked
        )
    elif trial_type == "Observations":
        # Observations specific attributes.
        group_id = f["group_id"][()]

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
        loaded_trials = Observations(
            stimulus_set, n_select=n_select, is_ranked=is_ranked,
            group_id=group_id, agent_id=agent_id, weight=weight,
            rt_ms=rt_ms
        )
    else:
        raise ValueError('No class found matching the provided `trial_type`.')
    f.close()

    if verbose > 0:
        print("Trial Summary")
        print('  trial_type: {0}'.format(trial_type))
        print('  n_trial: {0}'.format(loaded_trials.n_trial))
        if trial_type == "Observations":
            print('  n_agent: {0}'.format(len(np.unique(loaded_trials.agent_id))))
            print('  n_group: {0}'.format(len(np.unique(loaded_trials.group_id))))
        print('')
    return loaded_trials


def _pad_stimulus_set(stimulus_set, max_n_reference):
    """Pad 2D array with columns composed of -1."""
    n_trial = stimulus_set.shape[0]
    n_pad = max_n_reference - (stimulus_set.shape[1] - 1)
    if n_pad > 0:
        pad_mat = -1 * np.ones((n_trial, n_pad), dtype=np.int32)
        stimulus_set = np.hstack((stimulus_set, pad_mat))
    return stimulus_set


def _possible_outcomes(trial_configuration):
    """Return the possible outcomes of a trial configuration.

    Arguments:
        trial_configuration: A trial configuration Pandas Series.

    Returns:
        An 2D array indicating all possible outcomes where the values
            indicate indices of the reference stimuli. Each row
            corresponds to one outcome. Note the indices refer to
            references only and does not include an index for the
            query. Also note that the unpermuted index is returned
            first.

    """
    n_reference = int(trial_configuration['n_reference'])
    n_select = int(trial_configuration['n_select'])

    reference_list = range(n_reference)

    # Get all permutations of length n_select.
    perm = permutations(reference_list, n_select)

    selection = list(perm)
    n_outcome = len(selection)

    outcomes = np.empty((n_outcome, n_reference), dtype=np.int32)
    for i_outcome in range(n_outcome):
        # Fill in selections.
        outcomes[i_outcome, 0:n_select] = selection[i_outcome]
        # Fill in unselected.
        dummy_idx = np.arange(n_reference)
        for i_selected in range(n_select):
            loc = dummy_idx != outcomes[i_outcome, i_selected]
            dummy_idx = dummy_idx[loc]

        outcomes[i_outcome, n_select:] = dummy_idx

    return outcomes
