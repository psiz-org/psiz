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
    - MAYBE document stimulus index formatting [0,N[

"""

from abc import ABCMeta, abstractmethod

import numpy as np

from psiz.trials import UnjudgedTrials


class TrialGenerator(object):
    """Abstract base class for generating similarity judgment trials.

    Methods:
        generate: Generate trials.

    Attributes:
        n_stimuli: An integer indicating the total number of unique
            stimuli.

    """

    __metaclass__ = ABCMeta

    def __init__(self, n_stimuli):
        """Initialize.

        Arguments:
            n_stimuli: An integer indicating the total number of unique
                stimuli.
        """
        self.n_stimuli = n_stimuli

    @abstractmethod
    def generate(self, args):
        """Return generated trials based on provided arguments.

        Arguments:
            n_stimuli

        Returns:
            An UnjudgedTrials object.

        """
        pass


class RandomGenerator(TrialGenerator):
    """A random similarity trial generator."""

    def __init__(self, n_stimuli):
        """Initialize.

        Arguments:
            n_stimuli: An integer indicating the total number of unique
                stimuli.
        """
        TrialGenerator.__init__(self, n_stimuli)

    def generate(self, n_trial, n_reference=2, n_selected=1, is_ranked=True):
        """Return generated trials based on provided arguments.

        Arguments:
            n_trial: A scalar indicating the number of trials to
                generate.
            n_reference (optional): A scalar indicating the number of
                references for each trial.
            n_selected (optional): A scalar indicating the number of
                selections an agent must make.
            is_ranked (optional): Boolean indicating whether an agent
                must make ranked selections.

        Returns:
            An UnjudgedTrials object.

        """
        n_reference = np.int32(n_reference)
        n_selected = np.repeat(np.int32(n_selected), n_trial)
        is_ranked = np.repeat(bool(is_ranked), n_trial)
        stimulus_set = np.empty((n_trial, n_reference + 1), dtype=np.int32)
        for i_trial in range(n_trial):
            stimulus_set[i_trial, :] = np.random.choice(
                self.n_stimuli, (1, n_reference + 1), False
            )
        # Sort indices corresponding to references.
        stimulus_set[:, 1:] = np.sort(stimulus_set[:, 1:])
        return UnjudgedTrials(
            stimulus_set, n_selected=n_selected, is_ranked=is_ranked
        )


class ActiveGenerator(TrialGenerator):
    """A trial generator that leverages expected information gain."""

    def __init__(self, n_stimuli):
        """Initialize.

        Arguments:
            embedding:
        """
        TrialGenerator.__init__(self, n_stimuli)

    def generate(
            self, n_trial, samples, embedding, n_reference=None,
            n_selected=None, is_ranked=True, verbose=0):
        """Return generated trials based on provided arguments.

        Trials are selected in order to maximize expected information
        gain. Information gain is estimated using posterior samples.

        Arguments:
            n_trial: A scalar indicating the number of trials to
                generate.
            samples: A dictionary containing the posterior samples of
                parameters from a PsychologicalEmbedding object.
            n_reference (optional):
            n_selected (optional):
            is_ranked (optional):
            verbose (optional): An integer specifying the verbosity of
                printed output. If zero, nothing is printed. Increasing
                integers display an increasing amount of information.

        Returns:
            An UnjudgedTrials object.

        """
        # Goal: Reduce uncertainty on positions and group-specific tunings.

        # Point estimates of posterior samples.
        # z_central = np.median(samples['z'], axis=0)
        # utils.similarity_matrix(similarity_fn, z_central)

        return None  # TODO

    def _information_gain(self, embedding, samples, candidate_trial):
        """Return expected information gain of candidate trial(s).

        Information gain is determined by computing the mutual
        mutual information between the candidate trial(s) and the
        existing set of observations.

        Arguments:
            embedding:
            samples:
            candidate_trials:

        Returns:
            Expected information gain of candidate trial.
            shape = (n_trial,)

        """
        # TODO important that no placeholder outcomes are passed in. Probably 
        # solve problem by passing in the same configuraiton OR have outcome
        # probability return number of outcomes, OR list.
        cap = 2.2204e-16

        # Note z_samples has shape = (n_stimuli, n_dim, n_sample)
        z_samples = samples['z']
        # group_id = 0  # TODO
        # Note: prob_all has shape = (n_trial, n_outcome, n_sample)
        prob_all = embedding.outcome_probability(
            candidate_trial, group_id=None, z=z_samples)

        # First term of mutual information.
        # H(Y | obs, c) = - sum P(y_i | obs, c) log P(y_i | obs, c)
        # Take mean over samples to approximate p(y_i | obs, c).
        first_term = np.mean(prob_all, axis=2)
        # Use threshold to avoid log(0) issues (unlikely to happen).
        first_term = np.maximum(cap, first_term)
        first_term = first_term * np.log(first_term)
        # Sum over possible outcomes.
        first_term = -1 * np.sum(first_term, axis=1)

        # Second term of mutual information.
        # E[H(Y | Z, D, x)]
        # Use threshold to avoid log(0) issues (likely to happen).
        prob_all = np.maximum(cap, prob_all)
        second_term = prob_all * np.log(prob_all)
        # Take the sum over the possible outcomes.
        second_term = np.sum(second_term, axis=1)
        # Take the sum over all samples.
        second_term = np.mean(second_term, axis=1)

        info_gain = first_term + second_term
        return info_gain
