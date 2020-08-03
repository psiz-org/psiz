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
"""Module for generating unjudged similarity judgment trials.

Classes:
    DocketGenerator: Base class for generating a docket of unjudged
        similarity trials.
    RandomRank: Concrete class for generating random Rank similarity
        trials.
    ActiveRank: Concrete class that produces Rank similarity trials
        that are expected to maxize expected information gain.

Functions:
    expected_information_gain: A sample-based function for computing
        expected information gain.

"""

from abc import ABCMeta, abstractmethod
import copy
# from functools import partial
import itertools
# import multiprocessing
import time

import numpy as np
import numpy.ma as ma
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

from psiz.trials import stack, RankDocket, RateDocket
from psiz.preprocess import remove_catch_trials
from psiz.utils import ProgressBarRe, choice_wo_replace


class DocketGenerator(object):
    """Abstract base class for generating similarity judgment trials.

    Methods:
        generate: Generate unjudged trials.

    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """Initialize."""

    @abstractmethod
    def generate(self, args):
        """Return generated trials based on provided arguments.

        Arguments:
            n_stimuli

        Returns:
            A Docket object.

        """
        pass


class RandomRank(DocketGenerator):
    """A trial generator that blindly samples trials."""

    def __init__(self, n_stimuli, n_reference=2, n_select=1):
        """Initialize.

        Arguments:
            n_stimuli: A scalar indicating the total number of unique
                stimuli.
            n_reference (optional): A scalar indicating the number of
                references for each trial.
            n_select (optional): A scalar indicating the number of
                selections an agent must make.

        """
        DocketGenerator.__init__(self)

        self.n_stimuli = n_stimuli

        # Sanitize inputs.
        # TODO re-use sanitize methods from elsewhere
        self.n_reference = np.int32(n_reference)
        self.n_select = np.int32(n_select)
        self.is_ranked = True

    def generate(self, n_trial):
        """Return generated trials based on provided arguments.

        Arguments:
            n_trial: A scalar indicating the number of trials to
                generate.

        Returns:
            A RankDocket object.

        """
        n_reference = self.n_reference
        n_select = np.repeat(self.n_select, n_trial)
        is_ranked = np.repeat(self.is_ranked, n_trial)
        idx_eligable = np.arange(self.n_stimuli, dtype=np.int32)
        prob = np.ones([self.n_stimuli]) / self.n_stimuli
        stimulus_set = choice_wo_replace(
            idx_eligable, (n_trial, n_reference + 1), prob
        )

        return RankDocket(
            stimulus_set, n_select=n_select, is_ranked=is_ranked
        )


class ActiveRank(DocketGenerator):
    """A trial generator that uses approximate information gain."""

    def __init__(
            self, n_stimuli, n_reference=2, n_select=1, n_sample=1000,
            max_unique_query=None, n_candidate=1000):
        """Initialize.

        Arguments:
            n_stimuli: A scalar indicating the total number of unique
                stimuli.
            n_reference (optional): An integer indicating the number of
                references for each trial.
            n_select (optional): An integer indicating the number of
                selections an agent must make.
            n_sample (optional): The number of samples to draw from the
                posterior distribution in order to estimate information
                gain.
            max_unique_query (optional): A scalar parameter that
                governs heuristic behavior. The value indicates the
                maximum number of unique query stimuli that should be
                chosen. By default, this is equal to `n_stimuli`.
            n_candidate (optional): A scalar parameter that governs
                heuristic behavior. Given a query stimulus, this
                parameter determines how many candidate trials will be
                considered. In general, more is better, but you may be
                limited by time and RAM. Must be greater than zero.

        """
        DocketGenerator.__init__(self)

        self.n_stimuli = n_stimuli

        # Set trial configuration parameters.
        self.n_reference = np.int32(n_reference)
        self.n_select = np.int32(n_select)
        self.is_ranked = True

        # Set heuristic parameters.
        self.n_sample = n_sample
        if max_unique_query is None:
            max_unique_query = n_stimuli
        else:
            max_unique_query = np.minimum(max_unique_query, n_stimuli)
        self.max_unique_query = max_unique_query

        # TODO MAYBE np.minimum(max_candidate, n_candidate)
        self.n_candidate = n_candidate

    def generate(
            self, n_trial, model, priority=None, mask=None, group_id=0,
            verbose=0):
        """Return a docket of trials based on provided arguments.

        Trials are selected in order to maximize expected information
        gain given a specific group. Expected information gain is
        approximated using posterior samples.

        The docket is assembled in two steps, first the query stimuli
        are chosen. The values in `priority` are used to select query
        stimuli that the user considers promising. Second, reference
        stimuli for the queries are chosen. References are selected by
        favoring the current neighbors of the query.

        Arguments:
            n_trial: A scalar indicating the number of trials to
                generate.
            model: A PsychologicalEmbedding object.
            priority (optional): An array indicating the priority of
                sampling each stimulus. Priority is used as a heuristic
                to guide the active selection procedure in the
                typically vast search space. It is assumed that
                priorities are nonnegative and sum to one. If you want
                a particular stimulus to be excluded from the generated
                trials, set its corresponding priority to zero. The
                default behavior allocates equal priority to each
                stimulus.
                shape=[n_stimuli,]
            mask (optional): A Boolean array indicating which
                references are eligable for each query. Element
                `mask_ij` indicates if stimulus `j` can be used as a
                references for query stimulus `i`. If you provide a
                mask, the diagonal elements must be `False`.
                shape=[n_stimuli, n_stimuli]
            group_id (optional): An integer indicating the group ID to
                target for active selection.
            verbose (optional): An integer specifying the verbosity of
                printed output. If zero, nothing is printed. Increasing
                integers display an increasing amount of information.

        Returns:
            docket: A RankDocket object with trials that maximize expected
                information gain for the requested group.
            data: A dictionary containing additional information about
                the docket.
                ig_trial: A numpy.ndarray containing the expected
                information gain for each trial in the docket.
                shape = (n_trial,)

        """
        # Normalize priorities.
        if priority is None:
            priority = np.ones([n_stimuli]) / self.n_stimuli
        if np.sum(np.less(priority, 0)) > 0:
            raise ValueError(
                "The `priority` argument must only contain non-negative"
                " values."
            )

        # Handle mask.
        if mask is None:
            # Make all stimuli (except self) eligable to be a reference.
            mask = np.logical_not(np.eye(self.n_stimuli, dtype=bool))
        else:
            mask = mask.astype(bool)

        # Determine number of unique query stimuli to use in the docket.
        n_unique_query = np.minimum(n_trial, self.max_unique_query)

        # Assemble docket in two stages.
        (query_idx_arr, query_idx_count_arr) = self._select_query(
            n_trial, priority, n_unique_query
        )
        (docket, expected_ig) = self._select_references(
            model, group_id, query_idx_arr, query_idx_count_arr,
            priority, mask, verbose
        )
        data = {
            'docket' : {'expected_ig': expected_ig},
            'meta': {}
        }

        return docket, data

    def _select_query(self, n_trial, priority, n_unique_query):
        """Select which stimuli should serve as queries and how often.

        Arguments:
            n_trial: Integer indicating the total number of trials.
            priority: An array indicating stimulus priorities.
            n_unique_query: Scalar indicating the number of unique queries.

        Returns:
            query_idx_arr: An array of selected query indices.
            query_idx_count_arr: An array indicating the corresponding
                number of times a query is used in a trial.

        """
        # Create an index list for all selected query stimuli.
        # Initialize index with all stimuli.
        query_idx_arr = np.arange(0, self.n_stimuli)

        # If necessary, stochastically select subset of query stimuli
        # based on priority.
        query_priority = priority / np.sum(priority)
        if n_unique_query < self.n_stimuli:
            query_idx_arr = np.random.choice(
                query_idx_arr, n_unique_query, replace=False, p=query_priority
            )
        else:
            query_idx_arr = np.random.permutation(query_idx_arr)

        # Determine how many times each query stimulus should be used.
        query_idx_count_arr = np.zeros((n_unique_query), dtype=np.int32)
        for i_trial in range(n_trial):
            query_idx_count_arr[np.mod(i_trial, n_unique_query)] = (
                query_idx_count_arr[np.mod(i_trial, n_unique_query)] + 1
            )

        return query_idx_arr, query_idx_count_arr

    def _select_references(
            self, model, group_id, query_idx_arr, query_idx_count_arr,
            priority, mask, verbose):
        """Determine references for all requested query stimuli."""
        n_query = query_idx_arr.shape[0]

        if verbose > 0:
            progbar = ProgressBarRe(
                n_query, prefix='Progress:', length=50
            )
            progbar.update(0)

        docket = None
        expected_ig = None
        for i_query in range(n_query):
            mask_q = mask[query_idx_arr[i_query]]
            docket_q, expected_ig_q = _select_query_references(
                i_query, model, group_id, query_idx_arr,
                query_idx_count_arr,
                self.n_reference, self.n_select, self.n_candidate,
                self.n_sample, priority, mask_q
            )

            if verbose > 0:
                progbar.update(i_query + 1)

            # Add to dynamic list.
            if expected_ig is None:
                expected_ig = expected_ig_q
            else:
                expected_ig = np.hstack((expected_ig, expected_ig_q))

            if docket is None:
                docket = docket_q
            else:
                docket = stack((docket, docket_q))

        return docket, expected_ig


class RandomRate(DocketGenerator):
    """A trial generator that blindly samples trials."""

    def __init__(self, n_stimuli, n_reference=2, n_select=1):
        """Initialize.

        Arguments:
            n_stimuli: A scalar indicating the total number of unique
                stimuli.

        """
        DocketGenerator.__init__(self)

        self.n_stimuli = n_stimuli

    def generate(self, n_trial):
        """Return generated trials based on provided arguments.

        Arguments:
            n_trial: A scalar indicating the number of trials to
                generate.

        Returns:
            A RateDocket object.

        """
        idx_eligable = np.arange(self.n_stimuli, dtype=np.int32)
        prob = np.ones([self.n_stimuli]) / self.n_stimuli
        stimulus_set = choice_wo_replace(
            idx_eligable, (n_trial, 2), prob
        )
        return RateDocket(stimulus_set)


def _select_query_references(
        i_query, model, group_id, query_idx_arr, query_idx_count_arr,
        n_reference, n_select, n_candidate, n_sample, priority, mask_q):
    """Determine query references."""
    query_idx = query_idx_arr[i_query]
    n_trial_q = query_idx_count_arr[i_query]

    ref_idx_eligable = np.arange(len(priority))
    ref_idx_eligable = ref_idx_eligable[mask_q]
    ref_prob = priority[mask_q]
    ref_prob = ref_prob / np.sum(ref_prob)

    # Create a docket full of candidate trials.
    n_select = np.repeat(n_select, n_candidate)
    stimulus_set = np.empty(
        (n_candidate, n_reference + 1), dtype=np.int32
    )
    stimulus_set[:, 0] = query_idx
    stimulus_set[:, 1:] = choice_wo_replace(
        ref_idx_eligable, (n_candidate, n_reference), ref_prob
    )
    docket = RankDocket(stimulus_set, n_select=n_select)
    group = group_id * np.ones(n_candidate)
    ds_docket = docket.as_dataset(group)

    # Compute expected information gain from prediction samples.
    y_pred = tf.stack([
        model(ds_docket, training=False) for _ in range(n_sample)
    ], axis=0)
    expected_ig = expected_information_gain(y_pred).numpy()

    # Grab the top trials as requested.
    top_indices = np.argsort(-expected_ig)
    docket = docket.subset(
        top_indices[0:n_trial_q]
    )
    expected_ig = expected_ig[top_indices[0:n_trial_q]]

    return docket, expected_ig


def expected_information_gain(y_pred):
    """Return expected information gain of each discrete outcome trial.

    A sample-based approximation of information gain is determined by
    computing the mutual information between the candidate trial(s)
    and the existing set of observations (implied by the current model
    state).

    This sample-based approximation is intended for trials that have
    multiple discrete outcomes.

    This function is designed to be agnostic to the manner in which
    `y_pred` samples are drawn. For example, these could be dervied
    using MCMC or by sampling output predictions from a model fit using
    variational inference.

    NOTE: This function works with placeholder elements as long as
    `y_pred` is zero for those elements.

    Arguments:
        y_pred: A tf.Tensor of model predictions.
            shape=(n_sample, n_trial, n_outcome)

    Returns:
        A tf.Tensor object representing the expected information gain
        of the candidate trial(s).
        shape=(n_trial,)

    """
    # First term of mutual information.
    # H(Y | obs, c) = - sum P(y_i | obs, c) log P(y_i | obs, c),
    # where `c` indicates a candidate trial that we want to compute the
    # expected information gain for.
    # Take mean over samples to approximate p(y_i | obs, c).
    term0 = tf.reduce_mean(y_pred, axis=0)  # shape=(n_trial, n_outcome)
    term0 = term0 * tf.math.log(
        tf.math.maximum(term0, tf.keras.backend.epsilon())
    )
    # NOTE: At this point we would need to zero out place-holder outcomes,
    # but placeholder elements will always have a value of zero  since
    # y_pred will be zero for placeholder elements.
    # Sum over possible outcomes.
    term0 = -tf.reduce_sum(term0, axis=1)  # shape=(n_trial,)

    # Second term of mutual information.
    # E[H(Y | Z, D, x)]
    term1 = y_pred * tf.math.log(
        tf.math.maximum(y_pred, tf.keras.backend.epsilon())
    )
    # Take the sum over the possible outcomes.
    # NOTE: At this point we would need to zero out place-holder outcomes,
    # but placeholder elements will always have a value of zero since
    # y_pred will be zero for placeholder elements.
    term1 = tf.reduce_sum(term1, axis=2)  # shape=(n_sample, n_trial,)
    # Take the mean over all samples.
    term1 = tf.reduce_mean(term1, axis=0)  # shape=(n_trial,)

    return term0 + term1
