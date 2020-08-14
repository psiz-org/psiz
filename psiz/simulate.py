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
"""Module for simulating agent behavior.

Classes:
    Agent: An object that can be initialized using a psychological
        embedding and used to simulate similarity judgments.

"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
import tensorflow_probability as tfp

from psiz.trials import RankObservations


class Agent(object):
    """Agent that simulates similarity judgments.

    Attributes:
        embedding: A PsychologicalEmbedding object that supplies a
            similarity function and embedding points.
        group_id: An integer indicating which set of attention weights
            to use when simulating judgments.

    Methods:
        simulate: Stochastically simulate similarity judgments.

    """

    def __init__(self, model, group_id=0, agent_id=0):
        """Initialize.

        Arguments:
            model: A concrete instance of a PsychologicalEmedding
                object.
            group_id (optional): If the provided embedding was inferred
                for more than one group, an index can be provided to
                indicate which set of attention weights should be used.
            agent_id: An integer array indicating the agent ID of a
                trial. It is assumed that all IDs are non-negative and
                that observations with the same agent ID were judged by
                a single agent.
                shape = (n_trial,)

        """
        self.model = model
        self.group_id = group_id
        self.agent_id = agent_id

    def simulate(self, docket, session_id=None, batch_size=None):
        """Stochastically simulate similarity judgments.

        Arguments:
            docket: A RankDocket object representing the
                to-be-judged trials. The order of the stimuli in the
                stimulus set is ignored for the simulations.
            session_id: An integer array indicating the session ID of a
                trial. It is assumed that all IDs are non-negative.
                Trials with different session IDs were obtained during
                different sessions.
            batch_size (optional): If None, `batch_size` is equal to
                the total number of trials.

        Returns:
            RankObservations object representing the judged trials. The
                order of the stimuli is now informative.

        """
        if batch_size is None:
            batch_size = docket.n_trial

        agent_id = self.agent_id * np.ones((docket.n_trial), dtype=np.int32)
        group_id = self.group_id * np.ones((docket.n_trial), dtype=np.int32)

        # Create TF dataset.
        group = np.stack((group_id, agent_id), axis=-1)
        ds_docket = docket.as_dataset(group, all_outcomes=True).batch(
            batch_size, drop_remainder=False
        )

        # Call model with TensorFlow formatted docket and
        # stochastically sample an outcome.
        stimulus_set = None
        for data in ds_docket:
            data = data_adapter.expand_1d(data)
            x, _, _ = data_adapter.unpack_x_y_sample_weight(data)

            batch_stimulus_set = _rank_sample(
                x['stimulus_set'], self.model(x, training=False)
            )
            if stimulus_set is None:
                stimulus_set = [batch_stimulus_set]
            else:
                stimulus_set.append(batch_stimulus_set)
        stimulus_set = tf.concat(stimulus_set, axis=0).numpy() - 1

        obs = RankObservations(
            stimulus_set,
            n_select=docket.n_select,
            is_ranked=docket.is_ranked,
            group_id=group_id, agent_id=agent_id, session_id=session_id
        )
        return obs


def _rank_sample(stimulus_set, probs):
    """Stochasatically select outcome.
    
    Arguments:
        stimulus_set:
            shape=(batch_size, n_reference + 1, n_outcome)
        probs:
            shape=(batch_size, n_outcome)
    
    Returns:
        stimulus_set_selected:
            shape=(batch_size, n_reference + 1)

    """
    probs = tf.reduce_mean(probs, axis=0)
    outcome_distribution = tfp.distributions.Categorical(
        probs=probs
    )
    idx_sample = outcome_distribution.sample()
    idx_batch = tf.range(tf.shape(idx_sample)[0])
    idx_batch_sample = tf.stack([idx_batch, idx_sample], axis=1)

    # Retrieve stimulus set associated with particular outcome.
    stimulus_set = tf.transpose(stimulus_set, perm=[0, 2, 1])
    stimulus_set_selected = tf.gather_nd(stimulus_set, idx_batch_sample)
    return stimulus_set_selected