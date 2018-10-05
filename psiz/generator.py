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

"""Module for generating unjudged similarity judgment trials.

Classes:
    TrialGenerator: Base class for generating unjudged similarity
        trials.
    RandomGenerator: Concrete class for generating random similarity
        trials.
    ActiveGenerator: Concrete class for generating similarity trials
        using an active selection procedure that leverages expected
        informatin gain.

Todo:
    - implement ActiveGenerator
    - config_list should automatically generate number of outcomes,
        user should not need to provide this. Perhaps
        should be encapsulated as a class and class re-used within
        trials classes?
    - MAYBE change RandomGenerator API to take a config_list, generated
        trials are then randomly chosen from config list.
    - MAYBE document stimulus index formatting [0,N[

"""

from abc import ABCMeta, abstractmethod
import copy
import itertools

import numpy as np
import numpy.ma as ma
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from psiz.trials import Docket, stack
from psiz.simulate import Agent


class TrialGenerator(object):
    """Abstract base class for generating similarity judgment trials.

    Methods:
        generate: Generate unjudged trials.

    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """Initialize.

        Arguments:
        """

    @abstractmethod
    def generate(self, args):
        """Return generated trials based on provided arguments.

        Arguments:
            n_stimuli

        Returns:
            A Docket object.

        """
        pass


class RandomGenerator(TrialGenerator):
    """A trial generator that independently samples trials."""

    def __init__(self, n_reference=2, n_select=1, is_ranked=True):
        """Initialize.

        Arguments:
            n_reference (optional): A scalar indicating the number of
                references for each trial.
            n_select (optional): A scalar indicating the number of
                selections an agent must make.
            is_ranked (optional): Boolean indicating whether an agent
                must make ranked selections.
            config_list: TODO
        """
        TrialGenerator.__init__(self)

        # TODO sanitize (re-use methods from elsewhere; move those methods
        # out of classes and create method for sanitizing incoming
        # config_list)
        self.n_reference = np.int32(n_reference)
        self.n_select = np.int32(n_select)
        self.is_ranked = bool(is_ranked)

    def generate(self, n_trial, n_stimuli):
        """Return generated trials based on provided arguments.

        Arguments:
            n_trial: A scalar indicating the number of trials to
                generate.
            n_stimuli: An integer indicating the total number of unique
                stimuli.

        Returns:
            A Docket object.

        """
        n_reference = self.n_reference
        n_select = np.repeat(self.n_select, n_trial)
        is_ranked = np.repeat(self.is_ranked, n_trial)
        stimulus_set = np.empty((n_trial, n_reference + 1), dtype=np.int32)
        for i_trial in range(n_trial):
            stimulus_set[i_trial, :] = np.random.choice(
                n_stimuli, (1, n_reference + 1), False
            )
        # Sort indices corresponding to references.
        stimulus_set[:, 1:] = np.sort(stimulus_set[:, 1:])
        return Docket(
            stimulus_set, n_select=n_select, is_ranked=is_ranked
        )


# TODO
# API should be conceptually consistent for RandomGenerator and ActiveGenerator
#   trial config given at init
#   all embedding information given to generate (n_stimuli, etc.)
# 
# Should update_samples be included? probably not.
class ActiveGenerator(TrialGenerator):
    """A trial generator that leverages expected information gain.

    Attributes:
        config_list: TODO
        n_neighbor: A scalar that influences the greedy behavior of the
            active selection procedure.
        placeholder_docket: TODO

    Methods:
        generate: TODO
        update: TODO

    """

    def __init__(self, config_list=None, n_neighbor=10):
        """Initialize.

        Arguments:
            config_list (optional): A DataFrame indicating the trial
                configurations to consider.
            n_neighbor (optional): A scalar that influences the greedy
                behavior of the active selection procedure. An
                exhaustive search requires 'n_neighbor' to be equal to
                'n_stimuli' - 1. For problems with more than ~10
                stimuli, this becomes computationally infeasible.
        """
        TrialGenerator.__init__(self)

        self.n_neighbor = n_neighbor

        # Default trial configurations (2c1, 8c2).
        if config_list is None:
            config_list = pd.DataFrame({
                'n_reference': np.array([2, 8], dtype=np.int32),
                'n_select': np.array([1, 2], dtype=np.int32),
                'is_ranked': [True, True],
                'n_outcome': np.array([2, 56], dtype=np.int32)
            })
        self.config_list = config_list

        self.placeholder_docket = self._placeholder_docket(
            config_list, n_neighbor)

    def generate(
            self, n_trial, embedding, samples, group_id=None, n_query=3,
            verbose=0):
        """Return a docket of trials based on provided arguments.

        Trials are selected in order to maximize expected information
        gain given a specific group. Expected information gain is
        approximated using MCMC posterior samples.

        Arguments:
            n_trial: A scalar indicating the number of trials to
                generate.
            embedding: A PsychologicalEmbedding object.
            samples: A dictionary containing the posterior
                samples of parameters from a PsychologicalEmbedding
                object.
            group_id (optional): Scalar indicating which group to
                target.
            n_query (optional): Scalar indicating the number of unique
                query stimuli that should be chosen.
            verbose (optional): An integer specifying the verbosity of
                printed output. If zero, nothing is printed. Increasing
                integers display an increasing amount of information.

        Returns:
            best_docket: A Docket object.
            best_ig: A numpy.ndarray containing the expected
                information gain for each trial.
                shape = (n_trial,)

        Todo:
            - Reduce uncertainty on positions AND group-specific
                parameters.

        """
        n_stimuli = embedding.n_stimuli
        # Check if there are more stimuli than the requested number of
        # neighbors.
        self.n_neighbor = np.minimum(self.n_neighbor, n_stimuli - 1)

        if group_id is None:
            group_id = 0
        n_query = np.minimum(n_query, n_stimuli)
        n_query = np.minimum(n_query, n_trial)

        # Prioritize query stimulus based on corresponding entropy.
        # embdding.convert(samples, group_id) TODO
        h = stimulus_entropy(samples)
        query_idx = np.argsort(-h)
        query_idx = query_idx[0:n_query]

        (best_docket, best_ig) = self._determine_references(
            embedding, samples, self.placeholder_docket, query_idx,
            n_trial, group_id
        )
        return (best_docket, best_ig)

    def update_samples(self, embedding, obs, trs, samples):
        """Update posterior samples."""
        # Simulate judgments for trials.
        agent = Agent(embedding)
        obs_new = agent.simulate(trs)
        # Combine new obs with old obs.
        obs = stack((obs, obs_new))
        # Set embedding to last samples.
        embedding.z['values'] = samples['z'][:, :, -1]
        # Run sampler.
        samples = embedding.posterior_samples(obs, n_sample=1000, n_burn=10)

        # Return updated samples.
        return samples

    def _query_kl_priority(self, samples):
        """Return a priority score for every stimulus.

        The score indicates the priority each stimulus should serve as
        a query stimulus in a trial.

        Arguments:
            samples:

        Returns:
            A priority score.
                shape: (n_stimuli,)

        Notes:
            The priority scores are specific to a particular group.

        """
        # Unpack.
        z_samp = samples['z']
        z_median = np.median(z_samp, axis=2)
        (n_stimuli, n_dim, _) = z_samp.shape

        # Fit multi-variate normal for each stimulus.
        z_samp = np.transpose(z_samp, axes=[2, 0, 1])
        mu = np.empty((n_stimuli, n_dim))
        sigma = np.empty((n_stimuli, n_dim, n_dim))
        for i_stim in range(n_stimuli):
            gmm = GaussianMixture(
                n_components=1, covariance_type='full')
            gmm.fit(z_samp[:, i_stim, :])
            mu[i_stim, :] = gmm.means_[0]
            sigma[i_stim, :, :] = gmm.covariances_[0]

        # Determine nearest neighbors.
        # TODO maybe convert to faiss
        rho = 2  # TODO
        k = np.minimum(10, n_stimuli-1)
        nbrs = NearestNeighbors(
            n_neighbors=k+1, algorithm='auto', p=rho).fit(z_median)
        (_, idx) = nbrs.kneighbors(z_median)
        idx = idx[:, 1:]

        # Compute KL divergence for nearest neighbors.
        kl_div = np.empty((n_stimuli, k))
        for i_stim in range(n_stimuli):
            for j_stim in range(k):
                idx_j = idx[i_stim, j_stim]
                kl_div[i_stim, j_stim] = normal_kl_divergence(
                    mu[i_stim], sigma[i_stim], mu[idx_j], sigma[idx_j]
                )
        score = np.sum(kl_div, axis=1)
        return score

    def _placeholder_docket(self, config_list, n_neighbor):
        """Return placeholder trials.

        These trials use dummy indices so that the stimulus set can
        easily be replaced with actual indices to relevant stimuli.
        """
        # Precompute possible stimulus_set combinations.
        config_n_ref = np.unique(config_list['n_reference'].values)
        combos = {}
        for n_ref in config_n_ref:
            combos[n_ref] = stimulus_set_combos(n_neighbor, n_ref)

        # Fill in candidate trials.
        placeholder_docket = None
        for i_config in config_list.itertuples():
            stimulus_set = combos[i_config.n_reference]
            n_candidate = stimulus_set.shape[0]
            n_select = i_config.n_select * np.ones(
                n_candidate, dtype=np.int32)
            is_ranked = np.full(n_candidate, i_config.is_ranked, dtype=bool)
            if placeholder_docket is None:
                placeholder_docket = Docket(
                    stimulus_set, n_select=n_select, is_ranked=is_ranked)
            else:
                placeholder_docket = stack((
                    placeholder_docket,
                    Docket(
                        stimulus_set, n_select=n_select,
                        is_ranked=is_ranked)
                ))
        return placeholder_docket

    def _determine_references(
            self, embedding, samples, placeholder_docket, query_idx, n_trial,
            group_id):
        n_query = query_idx.shape[0]

        # Determine how many times each query stimulus should be used.
        n_trial_per_query = np.zeros((n_query), dtype=np.int32)
        for i_trial in range(n_trial):
            n_trial_per_query[np.mod(i_trial, n_query)] = (
                n_trial_per_query[np.mod(i_trial, n_query)] + 1
            )

        # TODO convert using group_specific "view".
        z_median = np.median(samples['z'], axis=2)
        rho = embedding.theta['rho']['value']
        # TODO MAYBE faiss nearest neighbors
        nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbor+1, algorithm='auto', p=rho
        ).fit(z_median)
        (_, nn_idx) = nbrs.kneighbors(z_median[query_idx])

        best_docket = None
        best_ig = None
        placeholder_stimulus_set = copy.copy(placeholder_docket.stimulus_set)
        # MAYBE parallelize this for loop.
        for i_query in range(n_query):
            # Substitute actual indices in candidate trials. The addition and
            # substraction of 1 is to handle the -1 used as a placeholder
            # in 'stimulus_set'.
            r = nn_idx[i_query] + 1
            r = np.hstack((0, r))
            placeholder_docket.stimulus_set = r[placeholder_stimulus_set + 1]
            placeholder_docket.stimulus_set = (
                placeholder_docket.stimulus_set - 1
            )

            ig = self._information_gain(
                embedding, samples, placeholder_docket, group_id=group_id)
            top_indices = np.argsort(-ig)
            # Grab the top N trials as specified by 'n_trial_per_query'.
            top_candidate = placeholder_docket.subset(
                top_indices[0:n_trial_per_query[i_query]]
            )
            curr_best_ig = ig[top_indices[0:n_trial_per_query[i_query]]]
            # Add to dynamic list.
            if best_ig is None:
                best_ig = curr_best_ig
            else:
                best_ig = np.hstack((best_ig, curr_best_ig))
            if best_docket is None:
                best_docket = top_candidate
            else:
                best_docket = stack((best_docket, top_candidate))
        # Reset placeholder. TODO Verify.
        placeholder_docket.stimulus_set = placeholder_stimulus_set
        return (best_docket, best_ig)

    def _information_gain(
            self, embedding, samples, docket, group_id=None):
        """Return expected information gain of trial(s) in docket.

        Information gain is determined by computing the mutual
        mutual information between the candidate trial(s) and the
        existing set of observations.

        Arguments:
            embedding: A PsychologicalEmbedding object.
            samples: Dictionary of sampled parameters.
                'z': shape = (n_stimuli, n_dim, n_sample)
            docket: A Docket object.
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
            docket, group_id=group_id, z=z_samples)

        # First term of mutual information.
        # H(Y | obs, c) = - sum P(y_i | obs, c) log P(y_i | obs, c)
        # Take mean over samples to approximate p(y_i | obs, c).
        first_term = ma.mean(prob_all, axis=2)
        # Use threshold to avoid log(0) issues (unlikely to happen).
        first_term = ma.maximum(cap, first_term)
        first_term = first_term * ma.log(first_term)
        # Sum over possible outcomes.
        first_term = -1 * ma.sum(first_term, axis=1)

        # Second term of mutual information.
        # E[H(Y | Z, D, x)]
        # Use threshold to avoid log(0) issues (likely to happen).
        prob_all = ma.maximum(cap, prob_all)
        second_term = prob_all * ma.log(prob_all)
        # Take the sum over the possible outcomes.
        second_term = ma.sum(second_term, axis=1)
        # Take the sum over all samples.
        second_term = ma.mean(second_term, axis=1)

        info_gain = first_term + second_term

        # Convert to normal numpy.ndarray.
        info_gain = info_gain.data
        return info_gain


def stimulus_set_combos(n_neighbor, n_reference):
    """Determine all possible stimulus set combinations.

    Assumes that index zero corresponds to the query stimulus and
    indices [1, n_reference] inclusive correspond to the reference
    stimuli.
    """
    eligable_list = np.arange(1, n_neighbor + 1, dtype=np.int32)

    stimulus_set = np.empty([0, n_reference + 1], dtype=np.int32)
    iterator = itertools.combinations(eligable_list, n_reference)
    for item in iterator:
        item = np.hstack((0, item))
        stimulus_set = np.vstack((stimulus_set, item))
    stimulus_set = stimulus_set.astype(dtype=np.int32)
    return stimulus_set


def normal_kl_divergence(mu_a, sigma_a, mu_b, sigma_b):
    """Return the Kullback-Leiler divergence between two normals."""
    mu_diff = mu_b - mu_a
    mu_diff = np.expand_dims(mu_diff, 1)
    n_dim = len(mu_a)
    sigma_b_inv = np.linalg.inv(sigma_b)
    kl = .5 * (
        np.log(np.linalg.det(sigma_b) / np.linalg.det(sigma_a)) - n_dim +
        np.matrix.trace(np.matmul(sigma_b_inv, sigma_a)) +
        np.matmul(
            np.matmul(np.transpose(mu_diff), sigma_b_inv), mu_diff
        )
    )
    return kl


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
            n_components=1, covariance_type='full')
        gmm.fit(z_samp[:, i_stim, :])
        entropy[i_stim] = normal_entropy(gmm.covariances_[0])
    return entropy


def normal_entropy(sigma):
    """Return entropy of multivariate normal distribution."""
    n_dim = sigma.shape[0]
    h = (
        (n_dim / 2) +
        (n_dim / 2 * np.log(2 * np.pi)) +
        (1 / 2 * np.log(np.linalg.det(sigma)))
    )
    return h

# How should I prioritize queries and references?
# TODO When prioritizing, must prioritize taking into account the
# group_id. Maybe by adjusting samples and z to take into account
# attention weights?

# Want to sample trials that also minimize uncertainty associated
# with attention weights. If they are just tacked on to the existing
# sample calculation, will the gain from the relatively high number
# of embedding points swamp the gain from the attention weights?
# Is there actually any benefit in including the attention weights,
# or is the uncertainty already well captures by the information
# gain computation which takes into account group_id.

# THOUGHTS: The current query prioritization strategy may be wrong.
# Not sure assymmetrical KL divergence is a good approach.

# Evaluate all combos of nearest neighbors.

# For selecting trials, select trial, continue sampler (100 samples,
# 0 burn-in), select next trial, repeat.

# Does query entropy correlate with expected information gain.
