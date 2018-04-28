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
    - join method

"""

from abc import ABCMeta, abstractmethod

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
            shape = [n_trial, max(n_reference) + 1]
        n_reference: An integer array indicating the number of
            references in each trial.
            shape = [n_trial, 1]
        n_selected: An integer array indicating the number of
            references selected in each trial.
            shape = [n_trial, 1]
        is_ranked: A Boolean array indicating which trials require
            reference selections to be ranked.
            shape = [n_trial, 1]
        config_id: An integer array indicating the
            configuration of each trial.
            shape = [n_trial, 1]
        config_list: A DataFrame object describing the unique trial
            configurations.

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
                shape = [n_trial, max(n_reference) + 1]
            n_selected (optional): An integer array indicating the
                number of references selected in each trial. Values are
                assumed to be greater than zero but less than or equal
                to the number of references for the corresponding
                trial.
                shape = [n_trial, 1]
            is_ranked (optional): A Boolean array indicating which
                trials require reference selections to be ranked.
                shape = [n_trial, 1]
        """
        n_trial = stimulus_set.shape[0]

        # Handle default settings.
        if n_selected is None:
            n_selected = np.ones((n_trial), dtype=np.int64)
        else:
            bad_locs = n_selected < 1
            n_bad = np.sum(bad_locs)
            if n_bad != 0:
                warnings.warn("The parameter 'n_selected' containes integers \
                    less than 1. Setting these values to 1.")
                n_selected[bad_locs] = np.ones((n_bad), dtype=np.int64)
        if is_ranked is None:
            is_ranked = np.full((n_trial), True)

        # Infer n_reference for each display.
        n_reference = self._infer_n_reference(stimulus_set)

        self.stimulus_set = stimulus_set
        self.n_trial = n_trial
        self.n_reference = n_reference
        self.n_selected = n_selected
        self.is_ranked = is_ranked

        # Attributes determined by concrete class.
        self.config_id = None
        self.config_list = None

    def _infer_n_reference(self, stimulus_set):
        """Return the number of references in each trial.

        Helper function that infers the number of available references
        for a given trial. The function assumes that values less than
        zero, are placeholder values and should be treated as
        non-existent.

        Args:
            stimulus_set: shape = [n_trial, 1]

        Returns:
            n_reference: An integer array indicating the number of
                references in each trial.
                shape = [n_trial, 1]

        """
        max_ref = stimulus_set.shape[1] - 1
        n_reference = max_ref - np.sum(stimulus_set < 0, axis=1)
        return np.array(n_reference, dtype=np.int64)

    @abstractmethod
    def _generate_configuration_id(self, *args):
        """Generate a unique ID for each trial configuration.

        Helper function that generates a unique ID for each of the
        unique trial configurations in the provided data set.

        Returns:
            df_config: A DataFrame containing all the unique
                trial configurations.
            config_id: A unique ID for each type of trial
                configuration.

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
                shape = [n_trial, max(n_reference) + 1]
            n_selected (optional): See SimilarityTrials.
            is_ranked (optional): See SimilarityTrials.
        """
        SimilarityTrials.__init__(self, stimulus_set, n_selected, is_ranked)

        # Determine unique display configurations.
        (config_list, config_id) = self._generate_configuration_id(
            self.n_reference, self.n_selected, self.is_ranked)
        self.config_id = config_id
        self.config_list = config_list

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
                shape = [n_trial, 1]
            n_selected: An integer array indicating the number of
                references selected in each trial.
                shape = [n_trial, 1]
            is_ranked:  Boolean array indicating which trials had
                selected references that were ordered.
                shape = [n_trial, 1]

        Returns:
            df_config: A DataFrame containing all the unique
                trial configurations.
            config_id: A unique ID for each type of trial
                configuration.

        """
        n_trial = len(n_reference)

        # Determine unique display configurations.
        d = {
            'n_reference': n_reference, 'n_selected': n_selected,
            'is_ranked': is_ranked
            }
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)

        # Assign display configuration ID for every observation.
        config_id = np.empty(n_trial)
        for i_type in range(n_config):
            a = (n_reference == df_config['n_reference'].iloc[i_type])
            b = (n_selected == df_config['n_selected'].iloc[i_type])
            c = (is_ranked == df_config['is_ranked'].iloc[i_type])
            f = np.array((a, b, c))
            display_type_locs = np.all(f, axis=0)
            config_id[display_type_locs] = i_type

        return (df_config, config_id)


class JudgedTrials(SimilarityTrials):
    """Object that encapsulates judged similarity trials.

    The attributes and behavior of JudgedTrials is largely inherited
    from SimilarityTrials.

    Attributes:
        group_id: An integer array indicating the group membership of
            each trial. It is assumed that group_id is composed of
            integers from [0, M-1] where M is the total number of
            groups.
            shape = [n_trial, 1]
        session_id: An integer array indicating the session ID of
            a trial. It is assumed that observations with the same
            session ID were judged by a single agent. A single agent
            may have completed multiple sessions.
            shape = [n_trial, 1]

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
                shape = [n_trial, 1]
        """
        SimilarityTrials.__init__(self, stimulus_set, n_selected, is_ranked)

        # Handle default settings.
        if group_id is None:
            group_id = np.zeros((self.n_trial), dtype=np.int64)
        else:
            bad_locs = group_id < 0
            n_bad = np.sum(bad_locs)
            if n_bad != 0:
                warnings.warn("The parameter 'group_id' containes integers \
                    less than 0. Setting these values to 0.")
                group_id[bad_locs] = np.zeros((n_bad), dtype=np.int64)

        self.group_id = group_id

        # Determine unique display configurations.
        (config_list, config_id) = self._generate_configuration_id(
            self.n_reference, self.n_selected, self.is_ranked, group_id)
        self.config_id = config_id
        self.config_list = config_list

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
                shape = [n_trial, 1]
            n_selected: An integer array indicating the number of
                references selected in each trial.
                shape = [n_trial, 1]
            is_ranked:  Boolean array indicating which trials had
                selected references that were ordered.
                shape = [n_trial, 1]
            group_id: An integer array indicating the group membership
                of each trial. It is assumed that group is composed of
                integers from [0, M-1] where M is the total number of
                groups. Separate attention weights are inferred for
                each group.
                shape = [n_trial, 1]
            session_id: An integer array indicating the session ID of
                a trial. It is assumed that observations with the same
                session ID were judged by a single agent. A single
                agent may have completed multiple sessions.
                shape = [n_trial, 1]

        Returns:
            df_config: A DataFrame containing all the unique
                trial configurations.
            config_id: A unique ID for each type of trial
                configuration.

        """
        n_trial = len(n_reference)

        if session_id is None:
            session_id = np.zeros((n_trial), dtype=np.int64)

        # Determine unique display configurations.
        d = {
            'n_reference': n_reference, 'n_selected': n_selected,
            'is_ranked': is_ranked, 'group_id': group_id,
            'session_id': session_id
            }
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)

        # Assign display configuration ID for every observation.
        config_id = np.empty(n_trial)
        for i_type in range(n_config):
            a = (n_reference == df_config['n_reference'].iloc[i_type])
            b = (n_selected == df_config['n_selected'].iloc[i_type])
            c = (is_ranked == df_config['is_ranked'].iloc[i_type])
            d = (group_id == df_config['group_id'].iloc[i_type])
            e = (session_id == df_config['session_id'].iloc[i_type])
            f = np.array((a, b, c, d, e))
            display_type_locs = np.all(f, axis=0)
            config_id[display_type_locs] = i_type

        return (df_config, config_id)
