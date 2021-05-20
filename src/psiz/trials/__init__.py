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
"""Trials initialization."""

from psiz.trials.stack import stack
from psiz.trials.load_trials import load_trials
from psiz.trials.similarity.docket_generator import DocketGenerator
from psiz.trials.similarity.rank.active_rank import ActiveRank
from psiz.trials.similarity.rank.active_rank import expected_information_gain_rank
from psiz.trials.similarity.rank.random_rank import RandomRank
from psiz.trials.similarity.rank.rank_trials import RankTrials
from psiz.trials.similarity.rank.rank_docket import RankDocket
from psiz.trials.similarity.rank.rank_observations import RankObservations
from psiz.trials.similarity.rate.random_rate import RandomRate
from psiz.trials.similarity.rate.rate_trials import RateTrials
from psiz.trials.similarity.rate.rate_docket import RateDocket
from psiz.trials.similarity.rate.rate_observations import RateObservations
from psiz.trials.experimental.trial_component import TrialComponent
from psiz.trials.experimental.trial_dataset import TrialDataset
from psiz.trials.experimental.contents.content import Content
from psiz.trials.experimental.contents.rank_similarity import RankSimilarity
from psiz.trials.experimental.contents.rate_similarity import RateSimilarity
from psiz.trials.experimental.outcomes.outcome import Outcome
from psiz.trials.experimental.outcomes.continuous import Continuous
from psiz.trials.experimental.outcomes.sparse_categorical import SparseCategorical
from psiz.trials.experimental.unravel_timestep import unravel_timestep
