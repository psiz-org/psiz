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
"""Example that reproduces Figure 5 and Figure 14 from Kruschke, 1992.

Results are saved in the directory specified by `fp_project`. By
default, a `psiz_examples` directory is created in your home directory.

Figure 5 Description (pg. 27):
Each datum shows the probability of selecting the correct category,
averaged across the eight exemplars within an epoch. For both graphs,
the response mapping constant was set to phi = 2.0, the specificity was
fixed at c = 6.5, and the learning rate for association weights was
lambda_w = 0.03. In Figure 5A, there was no attention learning
(lambda_a = 0.0), and it can be seen that Type II is learned much too
slowly. In Figure 5B, the attention-learning rate was raised to
lambda_a = 0.0033, and consequently Type II was learned second fastest,
as observed in human data.

Figure 14 Description (pg. 37-38):
It is now shown that ALCOVE can exhibit three-stage learning of high-
frequency exceptions to rules in a highly simplified abstract analogue
of the verb-acquisition situation. For this demonstration, the input
stimuli are distributed over two continuously varying dimensions as
shown in Figure 13.
...
ALCOVE was applied to the structure in Figure 13, using 14 hidden nodes
and parameter values near the values used to fit the Medin et al.
(1982) data: phi = 1.00, lambda_w = 0.025, c = 3.50, and
lambda_a = 0.010. Epoch updating was used, with each rule exemplar
occurring once per epoch and each exceptional case occurring four times
per epoch, for a total of 20 patterns per epoch. (The same qualitative
effects are produced with trial-by-trial updating, with superimposed
trial-by-trial "sawteeth," what Plunket and Marchman, 1991, called
micro U-shaped learning.) The results are shown in Figure 14.

References:
    [1] Kruschke, J. K. (1992). ALCOVE: an exemplar-based connectionist
        model of category learning. Psychological Review, 99(1), 22-44.
        http://dx.doi.org/10.1037/0033-295X.99.1.22.

"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import psiz

# Uncomment the following line to force eager execution.
# tf.config.run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """Run script."""
    # Settings.
    fp_project = Path.home() / Path(
        'psiz_examples', 'categorize', 'kruschke_1992'
    )
    fp_fig5 = Path(fp_project, 'kruschke_1992_fig5.pdf')
    fp_fig14 = Path(fp_project, 'kruschke_1992_fig14.pdf')

    # Directory preparation.
    fp_project.mkdir(parents=True, exist_ok=True)

    simulation_fig5(fp_fig5)
    simulation_fig14(fp_fig14)


def build_model(
        n_stimuli=None, n_dim=None, n_output=None, rho=None, tau=None,
        beta=None, lr_attention=None, lr_association=None, temperature=None,
        feature_matrix=None):
    """Build model for Kruschke simulation experiments."""
    similarity = psiz.keras.layers.ExponentialSimilarity(
        beta_initializer=tf.keras.initializers.Constant(beta),
        tau_initializer=tf.keras.initializers.Constant(tau),
        gamma_initializer=tf.keras.initializers.Constant(0.0),
        trainable=False,
    )
    alcove_embedding = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.Constant(feature_matrix),
        trainable=False,
    )
    cell = psiz.keras.layers.ALCOVECell(
        n_output, percept=alcove_embedding, similarity=similarity,
        rho_initializer=tf.keras.initializers.Constant(rho),
        temperature_initializer=tf.keras.initializers.Constant(temperature),
        lr_attention_initializer=tf.keras.initializers.Constant(lr_attention),
        lr_association_initializer=tf.keras.initializers.Constant(
            lr_association
        ),
        trainable=False
    )
    categorize = tf.keras.layers.RNN(
        cell, return_sequences=True, stateful=False
    )

    model = psiz.keras.models.Backbone(net=categorize)

    # Compile model.
    compile_kwargs = {
        # TODO it appears `SparseCategoricalCrossentropy` is necessary to
        # prevent shape complaints at compile.
        # TODO now that I changed `.output_size`, does non-sparse loss work?
        # 'loss': tf.keras.losses.SparseCategoricalCrossentropy(),
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy')
        ]
    }
    model.compile(**compile_kwargs)

    return model


def simulation_fig5(fp_fig5):
    """Create figure 5."""
    n_sequence = 10
    n_epoch = 50
    n_trial_per_epoch = 8

    # Define stimuli.
    feature_matrix = np.array([
        [0, 0, 0],  # mask_zero
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    n_stimuli = feature_matrix.shape[0] - 1
    n_dim = feature_matrix.shape[1]

    # Define tasks.
    # TODO taken from psixy.task.shepard_hovland_jenkins_1961()
    task_name_list = ['I', 'II', 'III', 'IV', 'V', 'VI']
    class_id = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 1]
    ])
    n_task = len(task_name_list)
    n_output = 2

    accuracy_epoch_0 = np.zeros([n_task, n_epoch])
    accuracy_epoch_attn = np.zeros([n_task, n_epoch])
    for i_task in range(n_task):
        ds, y = generate_fig5_stimulus_sequences(
            n_output, class_id[i_task], n_sequence, n_epoch
        )

        # Model without attention.
        model_0 = build_model(
            n_stimuli=n_stimuli, n_dim=n_dim, n_output=n_output, rho=1.0,
            tau=1.0, beta=6.5, lr_attention=0.0, lr_association=0.03,
            temperature=2.0, feature_matrix=feature_matrix
        )
        # Model with attention.
        model_attn = build_model(
            n_stimuli=n_stimuli, n_dim=n_dim, n_output=n_output, rho=1.0,
            tau=1.0, beta=6.5, lr_attention=0.0033, lr_association=0.03,
            temperature=2.0, feature_matrix=feature_matrix
        )

        # Predict behavior.
        behav_predict_0 = model_0.predict(ds)
        behav_predict_attn = model_attn.predict(ds)

        # Process predictions into per-epoch accuracy (i.e., probability
        # correct).
        accuracy_0 = np.sum(behav_predict_0 * y, axis=2)
        accuracy_epoch_0[i_task, :] = epoch_analysis_correct(
            accuracy_0, n_trial_per_epoch
        )
        accuracy_attn = np.sum(behav_predict_attn * y, axis=2)
        accuracy_epoch_attn[i_task, :] = epoch_analysis_correct(
            accuracy_attn, n_trial_per_epoch
        )

    plot_fig5(task_name_list, accuracy_epoch_0, accuracy_epoch_attn)
    plt.savefig(
        os.fspath(fp_fig5), format='pdf', bbox_inches='tight', dpi=400
    )


def plot_fig5(task_name_list, accuracy_epoch_0, accuracy_epoch_attn):
    """Create Figure 5."""
    n_task = len(task_name_list)

    # Plot figure.
    # Create color map of green and red shades, but don't use middle
    # yellow.
    cmap = matplotlib.cm.get_cmap('RdYlGn')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=n_task + 2)
    color_array = cmap(norm(np.flip(range(n_task + 2))))
    # Drop middle yellow colors.
    locs = np.array([1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
    color_array = color_array[locs, :]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax = plt.subplot(1, 2, 1)
    for i_task in range(n_task):
        ax.plot(
            accuracy_epoch_0[i_task, :],
            marker='o', markersize=3,
            c=color_array[i_task, :],
            label='{0}'.format(task_name_list[i_task])
        )
    ax.set_xlabel('epoch')
    ax.set_ylabel('Pr(correct)')
    ax.set_title('Without Attention')
    ax.legend(title='Category Type')

    ax = plt.subplot(1, 2, 2)
    for i_task in range(n_task):
        ax.plot(
            accuracy_epoch_attn[i_task, :],
            marker='o', markersize=3,
            c=color_array[i_task, :],
            label='{0}'.format(task_name_list[i_task])
        )
    ax.set_xlabel('epoch')
    ax.set_ylabel('Pr(correct)')
    ax.set_title('With Attention')
    ax.legend(title='Category Type')

    plt.tight_layout()


def simulation_fig14(fp_fig14):
    """Create figure 14."""
    n_sequence = 50
    n_epoch = 50
    n_trial_per_epoch = 20

    # Define stimuli.
    feature_matrix = np.array([
        [0, 0],  # mask_zero
        [1, 1],
        [2, 1],
        [3, 1],
        [1, 3],
        [2, 3],
        [3, 3],
        [1, 4.4],
        [3, 4.6],
        [1, 6],
        [2, 6],
        [3, 6],
        [1, 8],
        [2, 8],
        [3, 8],
    ])
    stimulus_label = np.array([
        'A_1', 'A_2', 'A_3', 'A_4', 'A_5', 'A_6', 'B_e',
        'A_e', 'B_6', 'B_5', 'B_4', 'B_3', 'B_2', 'B_1',
    ])
    n_stimuli = feature_matrix.shape[0] - 1
    stimulus_id = np.arange(n_stimuli)
    n_dim = feature_matrix.shape[1]

    # Define task.
    class_id = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    n_output = 2

    ds, x_stimulus, y_onehot = generate_fig14_stimulus_sequences(
        n_output, class_id, n_sequence, n_epoch
    )

    # Model with attention.
    model_attn = build_model(
        n_stimuli=n_stimuli, n_dim=n_dim, n_output=n_output, rho=1.0, tau=1.0,
        beta=3.5, lr_attention=0.010, lr_association=0.025, temperature=1.0,
        feature_matrix=feature_matrix
    )

    # Predict behavior.
    behav_predict_attn = model_attn.predict(ds)

    # Process predictions into stimulus-level epoch accuracy.
    accuracy_attn = np.sum(behav_predict_attn * y_onehot, axis=2)
    accuracy_epoch_attn = epoch_analysis_stimulus(
        x_stimulus, accuracy_attn, stimulus_id, n_trial_per_epoch
    )

    plot_fig14(accuracy_epoch_attn, stimulus_label, n_stimuli)
    plt.savefig(
        os.fspath(fp_fig14), format='pdf', bbox_inches='tight', dpi=400
    )


def plot_fig14(accuracy_epoch_attn, stimulus_label, n_stimuli):
    """Plot Figure 14."""
    # Plot figure.
    marker_list = [
        'o', 'o', 'o', 'o', 'o', 'o', 's',
        'o', 's', 's', 's', 's', 's', 's',
    ]
    color_array = np.vstack([
        np.repeat(
            np.array([[0.07197232, 0.54071511, 0.28489043, .1]]), 6, axis=0
        ),
        np.array([[0.8899654, 0.28673587, 0.19815456, 1.]]),
        np.array([[0.4295271, 0.75409458, 0.39146482, 1.]]),
        np.repeat(np.array([[0.64705882, 0., 0.14901961, .1]]), 6, axis=0),
    ])

    fig, ax = plt.subplots(figsize=(4, 4))

    ax = plt.subplot(1, 1, 1)
    for i_stim in range(n_stimuli):
        ax.plot(
            accuracy_epoch_attn[i_stim, :],
            marker=marker_list[i_stim], markersize=3,
            c=color_array[i_stim, :],
            label='{0}'.format(stimulus_label[i_stim])
        )
    ax.set_xlabel('epoch')
    ax.set_ylabel('Pr(correct)')
    ax.set_title('Rules and Exceptions')
    ax.legend()
    plt.tight_layout()


def epoch_analysis_correct(prob_correct, n_trial_per_epoch):
    """Epoch analysis."""
    n_sequence = prob_correct.shape[0]
    n_epoch = int(prob_correct.shape[1] / n_trial_per_epoch)

    # Use reshape in order to get epoch-level accuracy averages.
    prob_correct_2 = np.reshape(
        prob_correct, (n_sequence, n_trial_per_epoch, n_epoch), order='F'
    )
    seq_epoch_avg = np.mean(prob_correct_2, axis=1)

    # Average over all sequences.
    epoch_avg = np.mean(seq_epoch_avg, axis=0)
    return epoch_avg


def epoch_analysis_stimulus(
        x_stimulus, prob_response, stim_id_list, n_trial_per_epoch):
    """Epoch analysis."""
    n_sequence = prob_response.shape[0]
    n_epoch = int(prob_response.shape[1] / n_trial_per_epoch)
    n_stimuli = len(stim_id_list)

    stimulus_id_2 = np.reshape(
        x_stimulus, (n_sequence, n_trial_per_epoch, n_epoch),
        order='F'
    )
    # Use reshape in order to get epoch-level accuracy averages.
    prob_response_2 = np.reshape(
        prob_response, (n_sequence, n_trial_per_epoch, n_epoch), order='F'
    )
    # prob_response_2: (n_seq, n_trial, n_epoch)

    epoch_avg = np.zeros([n_stimuli, n_epoch])
    for i_epoch in range(n_epoch):
        prob_response_2_epoch = prob_response_2[:, :, i_epoch]
        stimulus_id_2_epoch = stimulus_id_2[:, :, i_epoch]
        for i_stim in range(n_stimuli):
            locs = np.equal(stimulus_id_2_epoch, stim_id_list[i_stim])
            epoch_avg[i_stim, i_epoch] = np.mean(prob_response_2_epoch[locs])

    return epoch_avg


def generate_fig5_stimulus_sequences(
        n_output, class_id_in, n_sequence, n_epoch):
    """Generate stimulus sequences."""
    n_stimuli = len(class_id_in)
    cat_idx = np.arange(n_stimuli, dtype=int)

    cat_idx_all = np.zeros([n_sequence, n_epoch * n_stimuli], dtype=int)
    for i_seq in range(n_sequence):
        curr_cat_idx = np.array([], dtype=int)
        for i_epoch in range(n_epoch):
            curr_cat_idx = np.hstack(
                [curr_cat_idx, np.random.permutation(cat_idx)]
            )
        cat_idx_all[i_seq, :] = curr_cat_idx

    y = class_id_in[cat_idx_all]

    sequence_length = n_epoch * n_stimuli
    groups = np.zeros([n_sequence, sequence_length, 1])
    # Must be at least rank 3 for RNN cell.
    y_onehot = tf.one_hot(y, n_output, on_value=1.0, off_value=0.0)
    x = {
        'stimulus_set': tf.constant(cat_idx_all + 1),
        # TODO `correct_label` added dim should be inside class for sample axis
        # TODO `correct_label` should we pass in one-hot rep instead?
        'correct_label': tf.constant(np.expand_dims(y, axis=2)),
        'groups': tf.constant(groups),
    }
    w = tf.constant(np.ones_like(y))
    ds = tf.data.Dataset.from_tensor_slices((x, y_onehot, w)).batch(
        n_sequence, drop_remainder=False
    )
    return ds, y_onehot.numpy()


def generate_fig14_stimulus_sequences(
        n_output, class_id_in, n_sequence, n_epoch):
    """Generate stimulus sequences."""
    n_stimuli = len(class_id_in)
    epoch_cat_idx = np.hstack([
        np.arange(n_stimuli, dtype=int),
        np.array([6, 6, 6, 7, 7, 7], dtype=int)
    ])

    n_trial_per_epoch = len(epoch_cat_idx)

    cat_idx_all = np.zeros(
        [n_sequence, n_epoch * n_trial_per_epoch], dtype=int
    )
    for i_seq in range(n_sequence):
        curr_cat_idx = np.array([], dtype=int)
        for i_epoch in range(n_epoch):
            curr_cat_idx = np.hstack(
                [curr_cat_idx, np.random.permutation(epoch_cat_idx)]
            )
        cat_idx_all[i_seq, :] = curr_cat_idx

    class_id = class_id_in[cat_idx_all]

    sequence_length = class_id.shape[1]
    groups = np.zeros([n_sequence, sequence_length, 1])

    y = class_id
    y_onehot = tf.one_hot(
        class_id, n_output, on_value=1.0, off_value=0.0
    ).numpy()
    x = {
        'stimulus_set': cat_idx_all + 1,
        'correct_label': np.expand_dims(class_id, axis=2),
        'groups': groups,
    }
    w = np.ones_like(y)
    ds = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(
        n_sequence, drop_remainder=False
    )
    return ds, cat_idx_all, y_onehot


if __name__ == "__main__":
    main()
