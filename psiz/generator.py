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

        Args:
            n_stimuli: An integer indicating the total number of unique
                stimuli.
        """
        self.n_stimuli = n_stimuli

    @abstractmethod
    def generate(self, args):
        """Return generated trials based on provided arguments.

        Args:
            n_stimuli

        Returns:
            An UnjudgedTrials object.

        """
        pass


class RandomGenerator(TrialGenerator):
    """A random similarity trial generator."""

    def __init__(self, n_stimuli):
        """Initialize.

        Args:
            n_stimuli: An integer indicating the total number of unique
                stimuli.
        """
        TrialGenerator.__init__(self, n_stimuli)

    def generate(self, n_trial, n_reference=2, n_selected=1, is_ranked=True):
        """Return generated trials based on provided arguments.

        Args:
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

        Args:
            n_stimuli:
        """
        TrialGenerator.__init__(self, n_stimuli)

    def generate(
            self, n_trial, samples, n_reference=None, n_selected=None, is_ranked=True, verbose=0):
        """Return generated trials based on provided arguments.

        Args:
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
        z_central = np.median(samples['z'], axis=0)
        utils.similarity_matrix(similarity_fn, z_central)

        return None  # TODO
