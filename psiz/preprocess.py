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
"""Module for preprocessing observations.

Functions:
    identify_catch_trials: Identify catch trials.
    grade_catch_trials: Identify and grade catch trials.
    quality_control: Remove observations belonging to agents that do
        not meet quality control standards.
    remove_catch_trials: Remove catch trials.

"""

from pathlib import Path

import numpy as np
import pandas as pd


def identify_catch_trials(obs):
    """Identify catch trials.

    Catch trials are assumed to be any trial where at least one of the
    references is the same stimulus as the query.

    Arguments:
        obs: A psiz.trials.RankObservations object.

    Returns:
        is_catch: Boolean array indicating catch trial locations.
            shape = (n_trial,)

    """
    n_trial = obs.n_trial
    is_catch = np.zeros([n_trial], dtype=bool)
    for i_trial in range(n_trial):
        # Determine which references are identical to the query.
        is_identical = np.equal(
            obs.stimulus_set[i_trial, 0], obs.stimulus_set[i_trial, 1:]
        )
        if np.sum(is_identical) > 0:
            is_catch[i_trial] = True
    return is_catch


def grade_catch_trials(obs, grade_mode='lenient'):
    """Grade catch trials.

    Catch trials are assumed to be any trial where at least one of the
    references is the same stimulus as the query. A catch trial is
    graded as correct depending on the grade_mode.

    Arguments:
        obs: A psiz.trials.RankObservations object.
        grade_mode (optional): Determines the manner in which responses
            are graded. Can be either 'strict', 'partial', or
            'lenient'. The options 'strict' and 'partial' are only
            relevant for trials where participants provide ranked
            responses, otherwise they are equivalent to 'lenient'. If
            'lenient', then one of the selected references must include
            the copy of the query. If 'strict', then the first choice
            must be the copy of the query. If 'partial', then full
            credit is given if the first choice is the copy of the
            query and half credit is given if a choice other than the
            first choice includes a copy of the query.

    Returns:
        avg_grade: Scalar indicating average grade on all catch trials.
            Is np.nan if there are no catch trials.
        grade: Array indicating grade of catch trial. The value can be
            between 0 and 1, where 1 is a perfect score.
            shape = (n_catch_trial,)
        is_catch: Boolean array indicating catch trial locations.
            shape = (n_trial,)

    """
    n_trial = obs.n_trial
    is_catch = identify_catch_trials(obs)
    grade = np.zeros([n_trial])

    for i_trial in range(n_trial):
        if is_catch[i_trial]:
            # Determine which references are identical to the query.
            is_identical = np.equal(
                obs.stimulus_set[i_trial, 0], obs.stimulus_set[i_trial, 1:]
            )
            # Grade response.
            if grade_mode == 'lenient':
                is_identical_selected = is_identical[0:obs.n_select[i_trial]]
                if np.sum(is_identical_selected) > 0:
                    grade[i_trial] = 1
            elif grade_mode == 'strict':
                if obs.is_ranked[i_trial]:
                    is_identical_selected = is_identical[0]
                else:
                    is_identical_selected = is_identical[
                        0:obs.n_select[i_trial]
                    ]
                if np.sum(is_identical_selected) > 0:
                    grade[i_trial] = 1
            elif grade_mode == 'partial':
                if obs.is_ranked[i_trial]:
                    is_identical_selected = is_identical[0]
                    if is_identical_selected:
                        grade[i_trial] = 1
                    else:
                        is_identical_selected = is_identical[
                            0:obs.n_select[i_trial]
                        ]
                        if np.sum(is_identical_selected) > 0:
                            grade[i_trial] = .5
                else:
                    is_identical_selected = is_identical[
                        0:obs.n_select[i_trial]
                    ]
                    if np.sum(is_identical_selected) > 0:
                        grade[i_trial] = 1
            else:
                raise ValueError((
                    "The argument `grade_mode` must be 'strict', 'partial' or"
                    " 'lenient'."
                ))

    # Compute average grade.
    grade = grade[is_catch]
    if len(grade) > 0:
        n_catch = np.sum(is_catch)
        avg_grade = np.sum(grade) / n_catch
    else:
        avg_grade = np.nan

    return (avg_grade, grade, is_catch)


def quality_control(obs, grade_thresh=1.0, grade_mode='lenient'):
    """Remove agents that do not meet quality control standards.

    Arguments:
        obs: A psiz.trials.RankObservations object.
        grade_thresh (optional): The threshold for dropping
            an agent's data.
        grade_mode (optional): Determines the manner in which responses
            are graded. See function grade_catch_trials.

    Returns:
        obs: An psiz.trials.RankObservations object with bad data removed.

    """
    agent_list = np.unique(obs.agent_id)
    grade_record = {
        'agent_id': agent_list,
        'grade': np.zeros(len(agent_list)),
        'is_retained': np.ones(len(agent_list), dtype=bool)
    }

    keep_locs = np.ones([obs.n_trial], dtype=bool)
    for idx, i_agent in enumerate(agent_list):
        agent_locs = np.equal(obs.agent_id, i_agent)
        obs_agent = obs.subset(agent_locs)
        (avg_grade, _, _) = grade_catch_trials(
            obs_agent, grade_mode=grade_mode
        )
        grade_record['grade'][idx] = avg_grade

        # Drop agent if below required grade threshold.
        # Note that this retains observations with zero catch trials.
        if avg_grade < grade_thresh:
            keep_locs[agent_locs] = False
            grade_record['is_retained'][idx] = False

    obs_new = obs.subset(keep_locs)
    df_grade = pd.DataFrame.from_dict(grade_record)
    return obs_new, df_grade


def remove_catch_trials(obs):
    """Remove all catch trials.

    Arguments:
        obs: A psiz.trials.RankObservations object.

    Returns:
        obs: A psiz.trials.RankObservations object.

    """
    is_catch = identify_catch_trials(obs)
    obs = obs.subset(np.logical_not(is_catch))
    return obs
