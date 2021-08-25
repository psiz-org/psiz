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
"""Generate rank-type unjudged similarity judgment trials.

Classes:
    ActiveRank: Concrete class that produces Rank similarity trials
        that are expected to maxize expected information gain.

Functions:
    expected_information_gain_rank: A sample-based function for computing
        expected information gain of Rank trials.

"""

import numpy as np
import tensorflow as tf

from psiz.trials.similarity.docket_generator import DocketGenerator
from psiz.trials.similarity.rank.rank_docket import RankDocket
from psiz.trials.stack import stack
from psiz.utils import ProgressBarRe, choice_wo_replace


class ActiveRank(DocketGenerator):
    """A trial generator that uses approximate information gain."""

    def __init__(
            self, n_stimuli, n_reference=2, n_select=1, max_unique_query=None,
            n_candidate=1000, batch_size=128):
        """Initialize.

        Arguments:
            n_stimuli: A scalar indicating the total number of unique
                stimuli.
            n_reference (optional): An integer indicating the number of
                references for each trial.
            n_select (optional): An integer indicating the number of
                selections an agent must make.
            max_unique_query (optional): A scalar parameter that
                governs heuristic behavior. The value indicates the
                maximum number of unique query stimuli that should be
                chosen. By default, this is equal to `n_stimuli`.
            n_candidate (optional): A scalar parameter that governs
                heuristic behavior. Given a query stimulus, this
                parameter determines how many candidate trials will be
                considered. In general, more is better, but you may be
                limited by time and RAM. Must be greater than zero.
            batch_size (optional): The batch size to use when
                iterating over the candidate docket.

        """
        DocketGenerator.__init__(self)

        self.n_stimuli = n_stimuli

        # Set trial configuration parameters.
        self.n_reference = np.int32(n_reference)
        self.n_select = np.int32(n_select)
        self.is_ranked = True

        # Set heuristic parameters.
        if max_unique_query is None:
            max_unique_query = n_stimuli
        else:
            max_unique_query = np.minimum(max_unique_query, n_stimuli)
        self.max_unique_query = max_unique_query

        # TODO MAYBE np.minimum(max_candidate, n_candidate)
        self.n_candidate = n_candidate
        self.batch_size = batch_size

    def generate(
            self, n_trial, model_list, q_priority=None, r_priority=None,
            groups=[0], verbose=0):
        """Return a docket of trials based on provided arguments.

        Trials are selected in order to maximize expected information
        gain given a specific group. Expected information gain is
        approximated using posterior samples.

        The docket is assembled in two steps. First the query stimuli
        are chosen based on the values in `q_priority`. Second, the
        reference stimuli for the queries are chosen based on
        `r_priority`.

        Arguments:
            n_trial: A scalar indicating the number of trials to
                generate.
            model_list: A list of PsychologicalEmbedding objects.
            q_priority (optional): A 1D array indicating the priority
                of sampling each stimulus to serve as a query. Priority
                is used as a heuristic to guide the active selection
                procedure in the typically vast search space. All
                priority values should be nonnegative. If you want a
                particular stimulus to be excluded as query stimulus,
                set its corresponding priority to zero. The default
                behavior allocates equal priority to each stimulus.
                shape=[n_stimuli,]
            r_priority (optional): A 2D array indicating the priority
                of sampling a stimulus to serve as a reference given a
                particular query. Element `r_priority`_{i,j} indicates
                the priority of sampling stimulus `j` as a reference
                given query stimulus `i`. All elements should be
                non-negative. Values on the diagonal elements are
                ignored.
                shape=[n_stimuli, n_stimuli]
            groups (optional): An integer array indicating the group
                membership to target for active selection.
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
        if q_priority is None:
            q_priority = np.ones([self.n_stimuli]) / self.n_stimuli

        # Check that `q_priority` is non-negative.
        if np.sum(np.less(q_priority, 0)) > 0:
            raise ValueError(
                "The `q_priority` argument must only contain non-negative"
                " values."
            )

        # Handle r_priority.
        if r_priority is None:
            # Uniform sampling probability.
            # NOTE: We divide by `n_stimuli - 1` because we will set the
            # diagonal element in each row to zero.
            r_priority = np.ones(
                [self.n_stimuli, self.n_stimuli], dtype=np.float32
            ) / (self.n_stimuli - 1)
        else:
            r_priority = r_priority.astype(np.float32)

        # Set diagonal to zero to prohibit sampling query as reference.
        r_priority[np.eye(self.n_stimuli, dtype=bool)] = 0

        # Check that `r_priority` is non-negative.
        if np.sum(np.less(r_priority, 0)) > 0:
            raise ValueError(
                "The `r_priority` argument must only contain non-negative"
                " values."
            )

        # Determine number of unique query stimuli to use in the docket.
        n_unique_query = np.minimum(n_trial, self.max_unique_query)

        # Assemble docket in two stages.
        (query_idx_arr, query_idx_count_arr) = self._select_query(
            n_trial, q_priority, n_unique_query
        )
        (docket, expected_ig) = self._select_references(
            model_list, groups, query_idx_arr, query_idx_count_arr,
            q_priority, r_priority, verbose
        )
        data = {
            'docket': {'expected_ig': expected_ig},
            'meta': {}
        }

        return docket, data

    def _select_query(self, n_trial, q_priority, n_unique_query):
        """Select which stimuli should serve as queries and how often.

        Arguments:
            n_trial: Integer indicating the total number of trials.
            q_priority: An array indicating stimulus priorities.
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
        # based on `q_priority`.
        query_priority = q_priority / np.sum(q_priority)
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
            self, model_list, groups, query_idx_arr, query_idx_count_arr,
            q_priority, r_priority, verbose):
        """Determine references for all requested query stimuli."""
        n_query = query_idx_arr.shape[0]

        if verbose > 0:
            progbar = ProgressBarRe(
                n_query, prefix='Active Trials:', length=50
            )
            progbar.update(0)

        docket = None
        expected_ig = None
        for i_query in range(n_query):
            r_priority_q = r_priority[query_idx_arr[i_query]]
            docket_q, expected_ig_q = _select_query_references(
                i_query, model_list, groups, query_idx_arr,
                query_idx_count_arr,
                self.n_reference, self.n_select, self.n_candidate,
                r_priority_q, self.batch_size
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


def _select_query_references(
        i_query, model_list, groups, query_idx_arr, query_idx_count_arr,
        n_reference, n_select, n_candidate, r_priority_q, batch_size):
    """Determine query references."""
    query_idx = query_idx_arr[i_query]
    n_trial_q = query_idx_count_arr[i_query]

    ref_idx_eligable = np.arange(len(r_priority_q))
    bidx_eligable = np.greater(r_priority_q, 0)
    ref_idx_eligable = ref_idx_eligable[bidx_eligable]
    ref_prob = r_priority_q[bidx_eligable]
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

    group_matrix = np.expand_dims(groups, axis=0)
    group_matrix = np.repeat(group_matrix, n_candidate, axis=0)

    ds_docket = docket.as_dataset(group_matrix).batch(
        batch_size, drop_remainder=False
    )

    expected_ig = []
    for x in ds_docket:
        batch_expected_ig = []
        # Compute average of ensemble of models.
        for model in model_list:
            # Compute expected information gain from prediction samples.
            batch_expected_ig.append(
                expected_information_gain_rank(
                    tf.transpose(model(x, training=False), perm=[1, 0, 2])
                )
            )
        # TODO Should IG be computed on ensemble samples collectively?
        # for model in model_list:
        #     batch_pred.append(model(x, training=False))
        # batch_pred = tf.stack(batch_pred, axis=TODO)
        # batch_expected_ig = expected_information_gain_rank(batch_pred)

        batch_expected_ig = tf.stack(batch_expected_ig, axis=0)
        batch_expected_ig = tf.reduce_mean(batch_expected_ig, axis=0)
        expected_ig.append(batch_expected_ig)
    expected_ig = tf.concat(expected_ig, 0).numpy()

    # Grab the top trials as requested.
    top_indices = np.argsort(-expected_ig)
    docket = docket.subset(
        top_indices[0:n_trial_q]
    )
    expected_ig = expected_ig[top_indices[0:n_trial_q]]

    return docket, expected_ig


def expected_information_gain_rank(y_pred):
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
    # but placeholder elements will always have a value of zero since
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
