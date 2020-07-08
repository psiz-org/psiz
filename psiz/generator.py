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
