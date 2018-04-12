"""Module for similarity judgment trials.

Classes:
    SimilarityTrials: Abstract class for similarity judgment trials.
    UnjudgedTrials: Unjudged similarity judgment trials.
    JudgedTrials: Similarity judgment trials that have been judged and
        will serve as observed data during inference.

Notes: TODO
    plural
    judged versus unjudged
        sorted stimulus set, group_id, (assignment_id)
    query stimulus:
    reference stimulus:
    group: A distinct population of agents. For example, observations
        could be collected from two groups: novices and experts. A 
        separate set of attention weights is inferred for each group.

Todo:
    - add assignment_id to JudgedTrials?
    - test module

License Boilerplate TODO

Author: B. D. Roads
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd


class SimilarityTrials(object):
    """Abstract base class for similarity judgment trials.
    
    Attributes:
        n_trial: An integer indicating the number of trials.
        stimulus_set: An integer matrix representing the set of stimuli 
            in each trial. Each row indicates the stimuli used in one 
            trial. The shape of the matrix implies the number of 
            references used for each trial. The first column is the 
            query stimulus, then the selected references (in order of 
            selection), and then any remaining unselected references.
            shape = [n_trial, max(n_reference) + 1]
        n_reference: An integer array indicating the number of 
            references in each trial.
            shape = [n_trial, 1]
        n_selected: An integer array indicating the number of 
            references selected in each trial.
            shape = [n_trial, 1]
        is_ranked:  Boolean array indicating which trials had selected
            references that were ordered.
            shape = [n_trial, 1]
        configuration_id: An integer array indicating the 
            configuration of each trial.
            shape = [n_trial, 1]
        configurations: A DataFrame object describing the unique trial
            configurations.
    
    Methods:
        subset: Return a subset of similarity trials given an index.
    """
    __metaclass__ = ABCMeta

    def __init__(self, stimulus_set, n_selected=None, is_ranked=None):
        """Initialize.

        Args:
            stimulus_set:
            n_selected (optional):
            is_ranked (optional):
        """

        n_trial = stimulus_set.shape[0]

        # Handle default settings.
        if n_selected is None:
            n_selected = np.ones((n_trial))
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
        self.configuration_id = None
        self.configurations = None
    
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
        n_reference = max_ref - np.sum(stimulus_set<0, axis=1)            
        return np.array(n_reference)
    
    @abstractmethod
    def _generate_configuration_id(self, n_reference, n_selected, is_ranked, 
        args):
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
            TODO
            group_id: An integer array indicating the group membership 
                of each trial. It is assumed that group is composed of 
                integers from [0,N] where N is the total number of 
                groups. Separate attention weights are inferred for 
                each group.
                shape = [n_trial, 1]
            assignment_id: An integer array indicating the assignment 
                ID of the trial. It is assumed that observations with a 
                given assignment ID were judged by a single person 
                although a single person may have completed multiple 
                assignments (e.g., Amazon Mechanical Turk).
                shape = [n_trial, 1]
        
        Returns:
            df_config: A DataFrame containing all the unique 
                trial configurations.
            configuration_id: A unique ID for each type of trial 
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
    
    Attributes:
        n_trial: An integer indicating the number of trials.
        stimulus_set: An integer matrix representing the set of stimuli 
            in each trial. Each row indicates the stimuli used in one 
            trial. The shape of the matrix implies the number of 
            references used for each trial. The first column is the 
            query stimulus, then the selected references (in order of 
            selection), and then any remaining unselected references.
            shape = [n_trial, max(n_reference) + 1]
        n_reference: An integer array indicating the number of 
            references in each trial.
            shape = [n_trial, 1]
        n_selected: An integer array indicating the number of 
            references selected in each trial.
            shape = [n_trial, 1]
        is_ranked:  Boolean array indicating which trials had selected
            references that were ordered.
            shape = [n_trial, 1]
        configuration_id: An integer array indicating the 
            configuration of each trial.
            shape = [n_trial, 1]
        configurations: A DataFrame object describing the unique trial
            configurations.
    
    Methods:
        subset: Return a subset of trials given an index.
    """

    def __init__(self, stimulus_set, n_selected=None, is_ranked=None, group_id=None):
        """Initialize.

        Args:
            stimulus_set:
            n_selected (optional):
            is_ranked (optional):
            group_id (optional):
        """
        SimilarityTrials.__init__(self, stimulus_set, n_selected, is_ranked)

        # Determine unique display configurations.
        (configurations, configuration_id) = self._generate_configuration_id(
            self.n_reference, self.n_selected, self.is_ranked)
        self.configuration_id = configuration_id
        self.configurations = configurations
    
    def subset(self, index):
        """Return subset of trials as new UnjudgedTrials object.

        Args:
            index: The indices corresponding to the subset.

        Returns:
            A new UnjudgedTrials object.
        """
        return UnjudgedTrials(self.stimulus_set[index,:], self.n_selected[index], self.is_ranked[index])
    
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
            configuration_id: A unique ID for each type of trial 
                configuration.
        """
        n_trial = len(n_reference)
        
        # Determine unique display configurations.
        d = {'n_reference': n_reference, 'n_selected': n_selected, 
        'is_ranked': is_ranked}
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)

        # Assign display configuration ID for every observation.
        configuration_id = np.empty(n_trial)
        for i_type in range(n_config):
            a = (n_reference == df_config['n_reference'].iloc[i_type])
            b = (n_selected == df_config['n_selected'].iloc[i_type])
            c = (is_ranked == df_config['is_ranked'].iloc[i_type])
            f = np.array((a,b,c))
            display_type_locs = np.all(f, axis=0)
            configuration_id[display_type_locs] = i_type
        
        return (df_config, configuration_id)


class JudgedTrials(SimilarityTrials):
    """Object that encapsulates judged similarity trials.
    
    Attributes:
        n_trial: An integer indicating the number of trials.
        stimulus_set: An integer matrix representing the set of stimuli 
            in each trial. Each row indicates the stimuli used in one 
            trial. The shape of the matrix implies the number of 
            references used for each trial. The first column is the 
            query stimulus, then the selected references (in order of 
            selection), and then any remaining unselected references.
            shape = [n_trial, max(n_reference) + 1]
        n_reference: An integer array indicating the number of 
            references in each trial.
            shape = [n_trial, 1]
        n_selected: An integer array indicating the number of 
            references selected in each trial.
            shape = [n_trial, 1]
        is_ranked:  Boolean array indicating which trials had selected
            references that were ordered.
            shape = [n_trial, 1]
        group_id: An integer array indicating the group membership of 
            each trial. It is assumed that group is composed of 
            integers from [0,N] where N is the total number of groups.
            shape = [n_trial, 1]
        configuration_id: An integer array indicating the 
            configuration of each trial.
            shape = [n_trial, 1]
        configurations: A DataFrame object describing the unique trial
            configurations.
    
    Methods:
        subset: Return a subset of trials given an index.
    """

    def __init__(self, stimulus_set, n_selected=None, is_ranked=None, group_id=None):
        """Initialize.

        Args:
            stimulus_set:
            n_selected (optional):
            is_ranked (optional):
            group_id (optional):
        """
        SimilarityTrials.__init__(self, stimulus_set, n_selected, is_ranked)

        # Handle default settings.
        if group_id is None:
            group_id = np.zeros((self.n_trial))
        self.group_id = group_id

        # Determine unique display configurations.
        (configurations, configuration_id) = self._generate_configuration_id(
            self.n_reference, self.n_selected, self.is_ranked, group_id)
        self.configuration_id = configuration_id
        self.configurations = configurations
    
    def subset(self, index):
        """Return subset of trials as new JudgedTrials object.

        Args:
            index: The indices corresponding to the subset.

        Returns:
            A new JudgedTrials object.
        """
        return JudgedTrials(self.stimulus_set[index,:], self.n_selected[index], self.is_ranked[index], self.group_id[index])
        
    def _generate_configuration_id(self, n_reference, n_selected, is_ranked, 
        group_id, assignment_id=None):
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
                integers from [0,N] where N is the total number of 
                groups. Separate attention weights are inferred for 
                each group.
                shape = [n_trial, 1]
            assignment_id: An integer array indicating the assignment 
                ID of the trial. It is assumed that observations with a 
                given assignment ID were judged by a single person 
                although a single person may have completed multiple 
                assignments (e.g., Amazon Mechanical Turk).
                shape = [n_trial, 1]
        
        Returns:
            df_config: A DataFrame containing all the unique 
                trial configurations.
            configuration_id: A unique ID for each type of trial 
                configuration.
        """
        n_trial = len(n_reference)

        if assignment_id is None:
            assignment_id = np.ones((n_trial))
        
        # Determine unique display configurations.
        d = {'n_reference': n_reference, 'n_selected': n_selected, 
        'is_ranked': is_ranked, 'group_id': group_id, 
        'assignment_id': assignment_id}
        df_config = pd.DataFrame(d)
        df_config = df_config.drop_duplicates()
        n_config = len(df_config)

        # Assign display configuration ID for every observation.
        configuration_id = np.empty(n_trial)
        for i_type in range(n_config):
            a = (n_reference == df_config['n_reference'].iloc[i_type])
            b = (n_selected == df_config['n_selected'].iloc[i_type])
            c = (is_ranked == df_config['is_ranked'].iloc[i_type])
            d = (group_id == df_config['group_id'].iloc[i_type])
            e = (assignment_id == df_config['assignment_id'].iloc[i_type])
            f = np.array((a,b,c,d,e))
            display_type_locs = np.all(f, axis=0)
            configuration_id[display_type_locs] = i_type
        
        return (df_config, configuration_id)
