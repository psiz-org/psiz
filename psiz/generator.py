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
    RandomGenerator: Concrete class for generating random similarity
        trials.

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
