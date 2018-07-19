# -*- coding: utf-8 -*-
# Copyright 2018 The PsiZ Authors. All Rights Reserved.
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
    UnjudgedTrials: Unjudged similarity judgment trials.
    JudgedTrials: Similarity judgment trials that have been judged and
        will serve as observed data during inference.

Notes:
    On each similarity judgment trial, an agent judges the similarity
        between a single query stimulus and multiple reference stimuli.
    Groups are used to identify distinct populations of agents. For
        example, similarity judgments could be collected from two
        groups: novices and experts. During inference, group
        information can be used to infer a separate set of attention
        weights for each group while sharing all other parameters.

Todo:

"""

from abc import ABCMeta, abstractmethod
from itertools import permutations

import numpy as np
import pandas as pd
import warnings


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
        n_selected: An integer array indicating the number of
            references selected in each trial.
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
            stimuli (as specified in the attribute 'stimulus_set'.

    Methods:
        subset: Return a subset of similarity trials given an index.

    """

    __metaclass__ = ABCMeta

    def __init__(self, stimulus_set, n_selected=None, is_ranked=None):
        """Initialize.

        Args:
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
            n_selected (optional): An integer array indicating the
                number of references selected in each trial. Values
                must be greater than zero but less than the number of
                references for the corresponding trial.
                shape = n_trial,)
            is_ranked (optional): A Boolean array indicating which
                trials require reference selections to be ranked.
                shape = (n_trial,)
        """
        self.n_trial = stimulus_set.shape[0]

        self.n_reference = self._infer_n_reference(stimulus_set)

        # Format stimulus set.
        # max_n_reference = 9
        self.max_n_reference = np.amax(self.n_reference)
        self.stimulus_set = stimulus_set[:, 0:self.max_n_reference+1]
        # self.stimulus_set = pad_stimulus_set(
        #     stimulus_set, max_n_reference)

        if n_selected is None:
            n_selected = np.ones((self.n_trial), dtype=np.int64)
        else:
            n_selected = self._check_n_selected(n_selected)
        self.n_selected = n_selected

        if is_ranked is None:
            is_ranked = np.full((self.n_trial), True)
        else:
            is_ranked = self._check_is_ranked(is_ranked)
        self.is_ranked = is_ranked

        # Attributes determined by concrete class.
        self.config_idx = None
        self.config_list = None
        self.outcome_idx_list = None

    def _infer_n_reference(self, stimulus_set):
        """Return the number of references in each trial.

        Infers the number of available references for each trial. The
        function assumes that values less than zero, are placeholder
        values and should be treated as non-existent.

        Args:
            stimulus_set: shape = [n_trial, 1]

        Returns:
            n_reference: An integer array indicating the number of
                references in each trial.
                shape = [n_trial, 1]

        """
        max_ref = stimulus_set.shape[1] - 1
        n_reference = max_ref - np.sum(stimulus_set < 0, axis=1)
        return n_reference.astype(dtype=np.int64)

    def _check_n_selected(self, n_selected):
        """Check the argument n_selected."""
        n_selected = n_selected.astype(np.int64)
        # Check shape argreement.
        if not (n_selected.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'n_selected' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        # Check lowerbound support limit.
        bad_locs = n_selected < 1
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The parameter 'n_selected' contains integers less than 1. "
                "Found {0} bad trial(s).").format(n_bad))
        # Check upperbound support limit.
        bad_locs = np.greater_equal(n_selected, self.n_reference)
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The parameter 'n_selected' contains integers greater than "
                "or equal to the corresponding 'n_reference'. Found {0} bad "
                "trial(s).").format(n_bad))
        return n_selected

    def _check_is_ranked(self, is_ranked):
        """Check the argument is_ranked."""
        if not (is_ranked.shape[0] == self.n_trial):
            raise ValueError((
                "The argument 'n_selected' must have the same length as the "
                "number of rows in the argument 'stimulus_set'."))
        bad_locs = np.not_equal(is_ranked, True)
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError((
                "The unranked version is not implemented, Found {0} bad "
                "trial(s).").format(n_bad))
        return is_ranked

    @abstractmethod
    def _generate_configuration_id(self, *args):
        """Generate a unique ID for each trial configuration.

        Helper function that generates a unique ID for each of the
        unique trial configurations in the provided data set.

        Returns:
            config_idx: A unique index for each type of trial
                configuration.
            df_config: A DataFrame containing all the unique
                trial configurations.
            outcome_idx_list: A list of the possible outcomes for each
                trial configuration.

        """
        pass

    @abstractmethod
    def subset(self, index):
        """Return subset of trials as new SimilarityTrials object.

        Args:
            index: The indices corresponding to the subset.

        Returns:
            A new SimilarityTrials object.

        """
        pass


class UnjudgedTrials(SimilarityTrials):
    """Object that encapsulates unjudged similarity trials.

    The attributes and behavior of UnjudgedTrials is largely inherited
    from SimilarityTrials.

    Attributes:
        config_list: A DataFrame object describing the unique trial
            configurations. The columns are 'n_reference',
            'n_selected', 'is_ranked'. and 'n_outcome'.

    Notes:
        stimulus_set: The order of the reference stimuli is
            unimportant. As usual, the the first column contains
            indices indicating query stimulus. The remaining columns
            contain indices indicating the reference stimuli in any
            order.
        Unique configurations and configuration IDs are determined by
            'n_reference', 'n_selected', and 'is_ranked'.

    Methods:
        subset: Return a subset of unjudged trials given an index.

    """

    def __init__(self, stimulus_set, n_selected=None, is_ranked=None):
        """Initialize.

        Extends initialization of SimilarityTrials.

        Args:
            stimulus_set: The order of the reference indices is not
                important. See SimilarityTrials.
            n_selected (optional): See SimilarityTrials.
            is_ranked (optional): See SimilarityTrials.
        """
        SimilarityTrials.__init__(self, stimulus_set, n_selected, is_ranked)

        # Determine unique display configurations.
        (config_idx, config_list, outcome_idx_list) = self._generate_configuration_id(
            self.n_reference, self.n_selected, self.is_ranked)
        self.config_idx = config_idx
        self.config_list = config_list
        self.outcome_idx_list = outcome_idx_list

    def subset(self, index):
        """Return subset of trials as new UnjudgedTrials object.

        Args:
            index: The indices corresponding to the subset.

        Returns:
            A new UnjudgedTrials object.

        """
        return UnjudgedTrials(self.stimulus_set[index, :],
                              self.n_selected[index], self.is_ranked[index])

    def _generate_configuration_id(self, n_reference, n_selected, is_ranked):
        """Generate a unique ID for each trial configuration.

        Helper function that generates a unique ID for each of the
        unique trial configurations in the provided data set.

        Args:
            n_reference: An integer array indicating the number of
                references in each trial.
                shape = (n_trial,)
            n_selected: An integer array indicating the number of
                references selected in each trial.
                shape = (n_trial,)
            is_ranked:  Boolean array indicating which trials had
                selected references that were ordered.
                shape = (n_trial,)

        Returns:
            df_config: A DataFrame containing all the unique
                trial configurations.
            config_idx: A unique index for each type of trial
                configuration.

        """
        n_trial = len(n_reference)

        # Determine unique display configurations.
        n_outcome_placeholder = np.zeros(n_selected.shape[0], dtype=np.int64)
        d = {
            'n_reference': n_reference, 'n_selected': n_selected,
            'is_ranked': is_ranked, 'n_outcome': n_outcome_placeholder
            }
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)
        n_out_idx = df_config.columns.get_loc("n_outcome")

        # Assign display configuration ID for every observation.
        config_idx = np.empty(n_trial, dtype=np.int64)
        outcome_idx_list = []
        for i_config in range(n_config):
            # Determine number of possible outcomes for configuration.
            outcome_idx = possible_outcomes(df_config.iloc[i_config])
            outcome_idx_list.append(outcome_idx)
            n_outcome = outcome_idx.shape[0]
            df_config.iloc[i_config, n_out_idx] = n_outcome
            # Find trials matching configuration.
            a = (n_reference == df_config['n_reference'].iloc[i_config])
            b = (n_selected == df_config['n_selected'].iloc[i_config])
            c = (is_ranked == df_config['is_ranked'].iloc[i_config])
            f = np.array((a, b, c))
            display_type_locs = np.all(f, axis=0)
            config_idx[display_type_locs] = i_config

        return (config_idx, df_config, outcome_idx_list)


class JudgedTrials(SimilarityTrials):
    """Object that encapsulates judged similarity trials.

    The attributes and behavior of JudgedTrials are largely inherited
    from SimilarityTrials.

    Attributes:
        group_id: An integer array indicating the group membership of
            each trial. It is assumed that group_id is composed of
            integers from [0, M-1] where M is the total number of
            groups.
            shape = (n_trial,)
        session_id: An integer array indicating the session ID of
            a trial. It is assumed that observations with the same
            session ID were judged by a single agent. A single agent
            may have completed multiple sessions.
            shape = (n_trial,)

    Notes:
        stimulus_set: The order of the reference stimuli is important.
            As usual, the the first column contains indices indicating
            query stimulus. The remaining columns contain indices
            indicating the reference stimuli. An agent's selected
            references are listed first (in order of selection if the
            trial is ranked) and remaining unselected references are
            listed in any order.
        Unique configurations and configuration IDs are determined by
            'group_id' in addition to the usual 'n_reference',
            'n_selected', and 'is_ranked' variables.

    Methods:
        subset: Return a subset of judged trials given an index.

    """

    def __init__(self, stimulus_set, n_selected=None, is_ranked=None,
                 group_id=None):
        """Initialize.

        Extends initialization of SimilarityTrials.

        Args:
            stimulus_set: The order of reference indices is important.
                An agent's selected references are listed first (in
                order of selection if the trial is ranked) and
                remaining unselected references are listed in any
                order. See SimilarityTrials.
            n_selected (optional): See SimilarityTrials.
            is_ranked (optional): See SimilarityTrials.
            group_id (optional): An integer array indicating the group
                membership of each trial. It is assumed that group_id
                is composed of integers from [0, M-1] where M is the
                total number of groups.
                shape = (n_trial,)
        """
        SimilarityTrials.__init__(self, stimulus_set, n_selected, is_ranked)

        # Handle default settings.
        if group_id is None:
            group_id = np.zeros((self.n_trial), dtype=np.int64)
        else:
            group_id = self._check_group_id(group_id)
        self.group_id = group_id

        # Determine unique display configurations.
        (config_idx, config_list, outcome_idx_list) = self._generate_configuration_id(
            self.n_reference, self.n_selected, self.is_ranked, group_id)
        self.config_idx = config_idx
        self.config_list = config_list
        self.outcome_idx_list = outcome_idx_list

    def _check_group_id(self, group_id):
        """Check the argument n_selected."""
        group_id = group_id.astype(np.int64)
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

    def subset(self, index):
        """Return subset of trials as new JudgedTrials object.

        Args:
            index: The indices corresponding to the subset.

        Returns:
            A new JudgedTrials object.

        """
        return JudgedTrials(self.stimulus_set[index, :],
                            self.n_selected[index], self.is_ranked[index],
                            self.group_id[index])

    def _generate_configuration_id(self, n_reference, n_selected, is_ranked,
                                   group_id, session_id=None):
        """Generate a unique ID for each trial configuration.

        Helper function that generates a unique ID for each of the
        unique trial configurations in the provided data set.

        Args:
            n_reference: An integer array indicating the number of
                references in each trial.
                shape = (n_trial,)
            n_selected: An integer array indicating the number of
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

        Returns:
            df_config: A DataFrame containing all the unique
                trial configurations.
            config_idx: A unique index for each type of trial
                configuration.

        """
        n_trial = len(n_reference)

        if session_id is None:
            session_id = np.zeros((n_trial), dtype=np.int64)

        # Determine unique display configurations.
        n_outcome_placeholder = np.zeros(n_selected.shape[0], dtype=np.int64)
        d = {
            'n_reference': n_reference, 'n_selected': n_selected,
            'is_ranked': is_ranked, 'group_id': group_id,
            'session_id': session_id, 'n_outcome': n_outcome_placeholder
            }
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)
        n_out_idx = df_config.columns.get_loc("n_outcome")

        # Assign display configuration index for every observation.
        config_idx = np.empty(n_trial, dtype=np.int64)
        outcome_idx_list = []
        for i_config in range(n_config):
            # Determine number of possible outcomes for configuration.
            outcome_idx = possible_outcomes(df_config.iloc[i_config])
            outcome_idx_list.append(outcome_idx)
            n_outcome = outcome_idx.shape[0]
            df_config.iloc[i_config, n_out_idx] = n_outcome
            # Find trials matching configuration.
            a = (n_reference == df_config['n_reference'].iloc[i_config])
            b = (n_selected == df_config['n_selected'].iloc[i_config])
            c = (is_ranked == df_config['is_ranked'].iloc[i_config])
            d = (group_id == df_config['group_id'].iloc[i_config])
            e = (session_id == df_config['session_id'].iloc[i_config])
            f = np.array((a, b, c, d, e))
            display_type_locs = np.all(f, axis=0)
            config_idx[display_type_locs] = i_config

        return (config_idx, df_config, outcome_idx_list)


def pad_stimulus_set(stimulus_set, max_n_reference):
    """Pad 2D array with columns composed of -1."""
    n_trial = stimulus_set.shape[0]
    n_pad = max_n_reference - (stimulus_set.shape[1] - 1)
    if n_pad > 0:
        pad_mat = -1 * np.ones((n_trial, n_pad), dtype=np.int64)
        stimulus_set = np.hstack((stimulus_set, pad_mat))
    return stimulus_set


def possible_outcomes(trial_configuration):
    """Return the possible outcomes of a trial configuration.

    Args:
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
    n_selected = int(trial_configuration['n_selected'])

    reference_list = range(n_reference)

    # Get all permutations of length n_selected.
    perm = permutations(reference_list, n_selected)

    selection = list(perm)
    n_outcome = len(selection)

    outcomes = np.empty((n_outcome, n_reference), dtype=np.int64)
    for i_outcome in range(n_outcome):
        # Fill in selections.
        outcomes[i_outcome, 0:n_selected] = selection[i_outcome]
        # Fill in unselected.
        dummy_idx = np.arange(n_reference)
        for i_selected in range(n_selected):
            loc = dummy_idx != outcomes[i_outcome, i_selected]
            dummy_idx = dummy_idx[loc]

        outcomes[i_outcome, n_selected:] = dummy_idx

    return outcomes


def stack(trials_list):
        """Return a SimilarityTrials object containing all trials.

        The stimulus_set of each SimilarityTrials object is padded
        first to match the maximum number of references of all the
        objects.

        Args:
            trials_list: A list of SimilarityTrials objects to be
                stacked.

        Returns:
            A new SimilarityTrials object.

        """
        # Determine the maximum number of references.
        max_n_reference = 0
        for trials in trials_list:
            if trials.max_n_reference > max_n_reference:
                max_n_reference = trials.max_n_reference

        # Grab relevant information from first entry in list.
        stimulus_set = pad_stimulus_set(
            trials_list[0].stimulus_set,
            max_n_reference
        )
        n_selected = trials_list[0].n_selected
        is_ranked = trials_list[0].is_ranked
        is_judged = True
        try:
            group_id = trials_list[0].group_id
        except AttributeError:
            is_judged = False

        for trials in trials_list[1:]:
            stimulus_set = np.vstack((
                stimulus_set,
                pad_stimulus_set(trials.stimulus_set, max_n_reference)
            ))
            n_selected = np.hstack((n_selected, trials.n_selected))
            is_ranked = np.hstack((is_ranked, trials.is_ranked))
            if is_judged:
                group_id = np.hstack((group_id, trials.group_id))
        
        if is_judged:
            trials_stacked = JudgedTrials(
                stimulus_set, n_selected, is_ranked, group_id)
        else:
            trials_stacked = UnjudgedTrials(
                stimulus_set, n_selected, is_ranked)
        return trials_stacked