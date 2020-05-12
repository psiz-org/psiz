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
# ==============================================================================

"""Module for generating unjudged similarity judgment trials.

Classes:
    DocketGenerator: Base class for generating a docket of unjudged
        similarity trials.
    RandomGenerator: Concrete class for generating random similarity
        trials.
    ActiveGenerator: Concrete class for generating similarity trials
        using an active selection procedure that leverages expected
        information gain.

Functions:
    information_gain: Compute expected information gain of a docket.
    query_kl_priority: Compute query stimulus priority using KL
        divergence approach for fitted Gaussian.
    normal_kl_divergence: Compute KL divergence for two multi-variate
        Gaussian distributions.
    stimulus_entropy: Compute entropy of stimulus based on Gaussian
        approximation to posterior samples.
    normal_entropy: Compute entropy associated with a Gaussian
        distribution.

Todo:
    - MAYBE Reduce uncertainty on positions AND group-specific
        parameters.

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

from psiz.trials import RankDocket, stack
from psiz.simulate import Agent
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
            A RankDocket object.

        """
        pass


class RandomGenerator(DocketGenerator):
    """A trial generator that independently samples trials."""

    def __init__(self, n_stimuli, n_reference=2, n_select=1, is_ranked=True):
        """Initialize.

        Arguments:
            n_stimuli: A scalar indicating the total number of unique
                stimuli.
            n_reference (optional): A scalar indicating the number of
                references for each trial.
            n_select (optional): A scalar indicating the number of
                selections an agent must make.
            is_ranked (optional): Boolean indicating whether an agent
                must make ranked selections.

        """
        DocketGenerator.__init__(self)

        self.n_stimuli = n_stimuli

        # Sanitize inputs.
        # TODO re-use sanitize methods from elsewhere
        self.n_reference = np.int32(n_reference)
        self.n_select = np.int32(n_select)
        self.is_ranked = bool(is_ranked)

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


class ActiveGenerator(DocketGenerator):
    """A trial generator that leverages expected information gain.

    Since the number of candidate trials is typically enormous, a
    number of heuristics are used to guide the selection processes
    and narrow the search space.

    Attributes:
        n_stimuli:
        max_query:
        max_neighbor:
        max_candidate:

    Methods:
        generate:

    """

    def __init__(
            self, n_stimuli, n_reference=2, n_select=1, is_ranked=True,
            max_query=None, max_neighbor=1000, max_candidate=1000):
        """Initialize.

        Arguments:
            n_stimuli: A scalar indicating the total number of unique
                stimuli.
            n_reference (optional): A scalar indicating the number of
                references for each trial.
            n_select (optional): A scalar indicating the number of
                selections an agent must make.
            is_ranked (optional): Boolean indicating whether an agent
            max_query (optional): A scalar parameter that governs
                heuristic behavior. The value indicates the maximum
                number of unique query stimuli that should be chosen.
                By default, this is equal `n_stimuli` (i.e., no
                heuristic used).
            max_neighbor (optional): A scalar parameter that governs
                heuristic behavior. When selecting references, this
                value determines which stimuli can be selected as
                references for a particular query stimulus. Only the
                stimuli that are no more than max_neighbor away will be
                considered. Increasing the value above the default is
                likely to increase computational time. It is not
                recommended to use a value smaller than the default.
            max_candidate (optional): A scalar parameter that governs
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
        self.is_ranked = bool(is_ranked)

        # Set heuristic parameters.
        if max_query is None:
            max_query = n_stimuli
        self.max_query = np.minimum(max_query, n_stimuli)
        self.max_neighbor = np.minimum(n_stimuli - 1, max_neighbor)
        self.max_candidate = max_candidate

    def generate(self, n_trial, embedding, samples, priority, verbose=0):
        """Return a docket of trials based on provided arguments.

        Trials are selected in order to maximize expected information
        gain given a specific group. Expected information gain is
        approximated using MCMC posterior samples.

        The docket is assembled in two steps, first the query stimuli
        are chosen. The values in `priority` are used to select query
        stimuli that the user considers promising. Second, reference
        stimuli for the queries are chosen. References are selected by
        only considering nearby neighbors.

        Arguments:
            n_trial: A scalar indicating the number of trials to
                generate.
            embedding: A PsychologicalEmbedding object.
            samples: A dictionary containing the posterior
                samples of parameters from a PsychologicalEmbedding
                object. The samples should correspond to group_id=0 in
                the `embedding` object.
            priority: An array indicating the priority of
                each stimulus. Priority is used as a heuristic to guide
                the active selection procedure in the typically vast
                search space.
            verbose (optional): An integer specifying the verbosity of
                printed output. If zero, nothing is printed. Increasing
                integers display an increasing amount of information.

        Returns:
            docket: A RankDocket object with trials that maximize expected
                information gain.
            info: A dictionary containing additional information about
                the docket.
                ig_trial: A numpy.ndarray containing the expected
                information gain for each trial in the docket.
                shape = (n_trial,)

        Notes:
            The variable `samples` must be samples from a particular
                group. It is assumed that group_id=0 in `embedding`
                was used to generate these samples.

        """
        if verbose > 0:
            print('[psiz] Generating docket using active selection...')

        # Ensure priorities sum to one. TODO
        # if np.sum(np.less(priority, 0)) > 0:
        #     raise('ValueError')
        # priority = priority - np.min(priority)
        # priority -= np.min(priority)
        priority = priority / np.sum(priority)

        # Determine number of unique query stimuli in docket.
        n_query = np.minimum(n_trial, self.max_query)

        # Assemble docket in two stages.
        (query_idx_list, n_trial_per_query_list) = self._select_query(
            n_trial, priority, n_query
        )
        (docket, ig) = self._select_references(
            embedding, samples, query_idx_list, n_trial_per_query_list,
            verbose=verbose
        )
        info = {
            "ig_trial": ig
        }

        return (docket, info)

    def _select_query(self, n_trial, priority, n_query):
        """Select which stimuli should by queries and how often.

        Arguments:
            priority:
            n_query: Scalar indicating the number of unique queries.

        Returns:
            query_idx_list: An array of selected query indices.
            n_trial_per_query_list: An array indicating the corresponding
                number of times a query should be shown in a trial.

        """
        n_stimuli = len(priority)

        # Create an index list for all selected query stimuli.
        # Initialize index with all stimuli.
        query_idx_list = np.arange(0, n_stimuli)

        # If necessary, stochastically select subset of query stimuli.
        if n_query < n_stimuli:
            query_idx_list = np.random.choice(
                query_idx_list, n_query, replace=False, p=priority
            )

        # Determine how many times each query stimulus should be used.
        n_trial_per_query_list = np.zeros((n_query), dtype=np.int32)
        for i_trial in range(n_trial):
            n_trial_per_query_list[np.mod(i_trial, n_query)] = (
                n_trial_per_query_list[np.mod(i_trial, n_query)] + 1
            )

        return query_idx_list, n_trial_per_query_list

    def _select_references(
            self, embedding, samples, query_idx_list, n_trial_per_query_list,
            verbose=0):
        """Determine references for all requested query stimuli.

        Notes:
            This function assumes that group_id=0 of the embedding
                should be used.

        """
        n_query = query_idx_list.shape[0]

        if verbose > 0:
            progbar = ProgressBarRe(
                n_query, prefix='Progress:', length=50
            )
            progbar.update(0)

        best_docket = None
        ig_best = None
        for i_query in range(n_query):
            top_candidate, curr_best_ig = _select_query_references(
                i_query, embedding, samples, query_idx_list,
                n_trial_per_query_list,
                self.n_reference, self.n_select, self.is_ranked,
                self.max_candidate, self.max_neighbor
            )

            if verbose > 0:
                progbar.update(i_query + 1)

            # Add to dynamic list.
            if ig_best is None:
                ig_best = curr_best_ig
            else:
                ig_best = np.hstack((ig_best, curr_best_ig))

            if best_docket is None:
                best_docket = top_candidate
            else:
                best_docket = stack((best_docket, top_candidate))

        return (best_docket, ig_best)


def _select_query_references(
        i_query, embedding, samples, query_idx_list, n_trial_per_query_list,
        n_reference, n_select, is_ranked, max_candidate, max_neighbor):
    """Determine query references.

    Arguments:
        max_neighbor (optional): Scalar indicating the number of query
            neighbors to consider as references. This parameter
            primarily acts as a bound on the computational cost of the
            sampling without replacement function.

    Notes:
        This function assumes that group_id=0 of the embedding
            should be used.

    """
    query_idx = query_idx_list[i_query]
    n_trial_q = n_trial_per_query_list[i_query]

    n_stimuli = embedding.n_stimuli
    z = embedding.z
    z_q = z[query_idx, :]
    s_qr = embedding.similarity(np.expand_dims(z_q, axis=0), z)

    # Determine eligable reference stimuli and their draw probability
    # based on similarity. TODO remove self
    ref_idx_eligable = np.argsort(-s_qr)
    # Remove query from eligable list. Note that we don't just pop the
    # first element since the user may be using an exotic similarity
    # function.
    ref_idx_eligable = ref_idx_eligable[
        np.not_equal(ref_idx_eligable, query_idx)
    ]
    # Limit eligable references to nearest neighbors.
    ref_idx_eligable = ref_idx_eligable[0:max_neighbor]
    # Renormalize probability of drawing a reference based on similarity
    # of the eligable options.
    ref_prob = s_qr[ref_idx_eligable] / np.sum(s_qr[ref_idx_eligable])

    # Create a docket full of candidate trials.
    n_select = np.repeat(n_select, max_candidate)
    is_ranked = np.repeat(is_ranked, max_candidate)
    stimulus_set = np.empty(
        (max_candidate, n_reference + 1), dtype=np.int32
    )
    stimulus_set[:, 0] = query_idx
    stimulus_set[:, 1:] = choice_wo_replace(
        ref_idx_eligable, (max_candidate, n_reference), ref_prob
    )
    docket = RankDocket(
        stimulus_set, n_select=n_select, is_ranked=is_ranked
    )

    # Compute information gain of candidate trials.
    ig = information_gain(embedding, samples, docket)

    # Grab the top trials as requested.
    top_indices = np.argsort(-ig)
    docket_top = docket.subset(
        top_indices[0:n_trial_q]
    )
    ig_top = ig[top_indices[0:n_trial_q]]

    return docket_top, ig_top


def information_gain(embedding, samples, docket, group_id=None):
    """Return expected information gain of trial(s) in docket.

    Information gain is determined by computing the mutual
    mutual information between the candidate trial(s) and the
    existing set of observations.

    Arguments:
        embedding: A PsychologicalEmbedding object.
        samples: Dictionary of sampled parameters.
            'z': shape = (n_stimuli, n_dim, n_sample)
        docket: A RankDocket object.
        group_id (optional): A scalar or an array with
            shape = (n_trial,).

    Returns:
        A numpy.ndarray representing the expected information gain
        of the candidate trial(s).
        shape = (n_trial,)

    """
    cap = 2.2204e-16

    # Note: z_samples has shape = (n_stimuli, n_dim, n_sample)
    z_samples = samples['z']
    # Note: prob_all has shape = (n_trial, n_outcome, n_sample)
    prob_all = embedding.outcome_probability(
        docket, group_id=group_id, z=z_samples
    )

    # First term of mutual information.
    # H(Y | obs, c) = - sum P(y_i | obs, c) log P(y_i | obs, c)
    # Take mean over samples to approximate p(y_i | obs, c).
    # shape = (n_trial, n_outcome)
    first_term = ma.mean(prob_all, axis=2)
    # Use threshold to avoid log(0) issues (unlikely to happen).
    first_term = ma.maximum(cap, first_term)
    first_term = first_term * ma.log(first_term)
    # Sum over possible outcomes.
    first_term = -1 * ma.sum(first_term, axis=1)

    # Second term of mutual information.
    # E[H(Y | Z, D, x)]
    # Use threshold to avoid log(0) issues (likely to happen).
    # shape = (n_trial, n_outcome, n_sample)
    prob_all = ma.maximum(cap, prob_all)

    # shape = (n_trial, n_outcome, n_sample)
    second_term = prob_all * ma.log(prob_all)

    # Take the sum over the possible outcomes.
    # shape = (n_trial, n_sample)
    second_term = ma.sum(second_term, axis=1)

    # Take the sum over all samples.
    # shape = (n_trial)
    second_term = ma.mean(second_term, axis=1)

    info_gain = first_term + second_term

    # Convert to normal numpy.ndarray.
    info_gain = info_gain.data
    return info_gain


def determine_stimulus_priority(embedding, samples, mode='kl'):
    """Determine stimulus priority.

    The priority of all stimuli must sum to one.

    Arguments:
        embedding:
        samples:
        mode (optional): Can be 'random', 'entropy' or 'kl'.

    Returns:
        priority:

    """
    n_stimuli = embedding.n_stimuli

    if mode is "random":
        # Completely random.
        priority = np.ones([n_stimuli]) / n_stimuli
    elif mode is "entropy":
        # Based on entropy.
        entropy = stimulus_entropy(samples)
        rel_entropy = entropy - np.min(entropy)
        priority = rel_entropy / np.sum(rel_entropy)
    elif mode is "kl":
        # Based on KL divergence.
        stim_kl = query_kl_priority(embedding, samples)
        priority = stim_kl / np.sum(stim_kl)

    return priority


def stimulus_entropy(samples):
    """Return the approximate entropy associated with every stimulus.

    Arguments:
        samples: Posterior samples.
            shape: (n_stimuli, n_dim, n_sample)

    Returns:
        The approximate entropy associated with each stimulus.
            shape: (n_stimuli,)

    Notes:
        The computation is specific to a particular group.

    """
    # Unpack.
    z_samp = samples['z']
    n_stimuli = z_samp.shape[0]

    # Fit multi-variate normal for each stimulus.
    z_samp = np.transpose(z_samp, axes=[2, 0, 1])
    entropy = np.empty((n_stimuli))
    for i_stim in range(n_stimuli):
        gmm = GaussianMixture(
            n_components=1, covariance_type='full'
        )
        gmm.fit(z_samp[:, i_stim, :])
        entropy[i_stim] = normal_entropy(gmm.covariances_[0])
    return entropy


def normal_entropy(cov):
    """Return entropy of multivariate normal distribution."""
    n_dim = cov.shape[0]
    h = (
        (n_dim / 2) +
        (n_dim / 2 * np.log(2 * np.pi)) +
        (1 / 2 * np.log(np.linalg.det(cov)))
    )
    return h


def query_kl_priority(embedding, samples):
    """Return a priority score for every stimulus.

    The score indicates the priority each stimulus should serve as
    a query stimulus in a trial.

    Arguments:
        embedding:
        samples:

    Returns:
        A priority score.
            shape: (n_stimuli,)

    Notes:
        The priority scores are specific to a particular group.

    """
    # Unpack.
    # z = embedding.z
    z = np.median(samples['z'], axis=2)
    try:
        rho = embedding.rho
    except AttributeError:
        rho = 2.
    z_samp = samples['z']
    # z_median = np.median(z_samp, axis=2)
    (n_stimuli, n_dim, _) = z_samp.shape

    # Fit multi-variate normal for each stimulus.
    z_samp = np.transpose(z_samp, axes=[2, 0, 1])
    mu = np.empty((n_stimuli, n_dim))
    cov = np.empty((n_stimuli, n_dim, n_dim))
    for i_stim in range(n_stimuli):
        gmm = GaussianMixture(
            n_components=1, covariance_type='full'
        )
        gmm.fit(z_samp[:, i_stim, :])
        mu[i_stim, :] = gmm.means_[0]
        cov[i_stim, :, :] = gmm.covariances_[0]

    # Determine nearest neighbors.
    # TODO maybe convert to faiss
    k = np.minimum(10, n_stimuli-1)
    nbrs = NearestNeighbors(
        n_neighbors=k+1, algorithm='auto', p=rho
    ).fit(z)
    (_, nn_idx) = nbrs.kneighbors(z)
    nn_idx = nn_idx[:, 1:]  # Drop self index.

    # Compute KL divergence for nearest neighbors.
    kl_div = np.empty((n_stimuli, k))
    for query_idx in range(n_stimuli):
        for j_nn in range(k):
            idx_j = nn_idx[query_idx, j_nn]
            kl_div[query_idx, j_nn] = normal_kl_divergence(
                mu[idx_j], cov[idx_j], mu[query_idx], cov[query_idx]
            )
    score = np.sum(kl_div, axis=1)
    score = 1 / score

    return score


def normal_kl_divergence(mu_a, cov_a, mu_b, cov_b):
    """Return the Kullback-Leibler divergence between two normals."""
    mu_diff = mu_b - mu_a
    mu_diff = np.expand_dims(mu_diff, 1)
    n_dim = len(mu_a)
    cov_b_inv = np.linalg.inv(cov_b)
    kl = .5 * (
        np.log(np.linalg.det(cov_b) / np.linalg.det(cov_a)) - n_dim +
        np.matrix.trace(np.matmul(cov_b_inv, cov_a)) +
        np.matmul(
            np.matmul(np.transpose(mu_diff), cov_b_inv), mu_diff
        )
    )
    return kl
