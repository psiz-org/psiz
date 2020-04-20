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

"""Module of psychological embedding models.

Classes:
    PsychologicalEmbedding: Abstract base class for embedding model.
    Rate: Class that uses ratio observations between unanchored sets
        of stimulus (typically two stimuli).
    Rank: Class that uses ordinal observations that are anchored by a
        designated query stimulus.
    Sort: Class that uses ordinal observations that are unanchored by
        a designated query stimulus.

Functions:
    load_model: Load a hdf5 file, that was saved with the `save`
        class method, as a PsychologicalEmbedding object.

TODO:
    * Implement RateModel class.
    * Implement SortModel class.

"""

from abc import ABCMeta, abstractmethod
import copy
import datetime
from random import randint
import sys
import time
import warnings

import h5py
import numpy as np
import numpy.ma as ma
import pandas as pd
from sklearn import mixture
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.initializers import Initializer
import tensorflow.keras.optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History
from tensorflow.python.keras.callbacks import configure_callbacks
from tensorflow.keras.constraints import Constraint

import psiz.keras.layers
import psiz.trials
import psiz.utils


class PsychologicalEmbedding(metaclass=ABCMeta):
    """Abstract base class for a psychological embedding model.

    The embedding procedure jointly infers three components. First, the
    embedding algorithm infers a stimulus representation denoted by the
    variable z. Second, the embedding algorithm infers the variables
    governing the similarity kernel, denoted theta. Third, the
    embedding algorithm infers a set of attention weights if there is
    more than one group.

    Methods:
        reset_weights: Reset all trainable variables.
        compile: Assign a optimizer, loss and regularization function
            for the optimization procedure.
        fit: Fit the embedding model using the provided observations.
        evaluate: Evaluate the embedding model using the provided
            observations.
        similarity: Return the similarity between provided points.
        distance: Return the (weighted) minkowski distance between
            provided points.
        view: Returns a view-specific embedding.
        trainable: Get or set which parameters are trainable.
        outcome_probability: Return the probability of the possible
            outcomes for each trial.
        posterior_samples: Sample from the posterior distribution.
        set_log: Adjust the TensorBoard logging behavior.
        save: Save the embedding model as an hdf5 file.

    Attributes:
        n_stimuli: The number of unique stimuli in the embedding.
        n_dim: The dimensionality of the embedding.
        n_group: The number of distinct groups in the embedding.
        z: A dictionary with the keys 'value', 'trainable'. The key
            'value' contains the actual embedding points. The key
            'trainable' is a boolean flag that determines whether
            the embedding points are inferred during inference.
        theta: Dictionary containing data about the parameter values
            governing the similarity kernel. The dictionary contains
            the variable names as keys at the first level. For each
            variable, there is an additional dictionary containing
            the keys 'value', 'trainable', and 'bounds'. The key
            'value' indicates the actual value of the parameter. The
            key 'trainable' is a boolean flag indicating whether the
            variable is trainable during inferene. The key 'bounds'
            indicates the bounds of the parameter during inference. The
            bounds are specified using a list of two items where the
            first item indicates the lower bound and the second item
            indicates the upper bound. Use None to indicate no bound.
        phi: Dictionary containing data about the group-specific
            parameter values. These parameters are only trainable if
            there is more than one group. The dictionary contains the
            parameter names as keys at the first level. For each
            parameter name, there is an additional dictionary
            containing the keys 'value' and 'trainable'. The key
            'value' indicates the actual value of the parameter. The
            key 'trainable' is a boolean flag indicating whether the
            variable is trainable during inference. The free parameter
            `w` governs dimension-wide weights.
        do_log: A boolean variable that controls whether gradient
            decent progress is logged. By default, this is initialized
            to False.
        log_dir: The location of the logs. The default location is
            `/tmp/psiz/tensorboard_logs/`.
        log_freq: The number of epochs to wait between log entries.

    """

    def __init__(
            self, n_stimuli, n_dim=2, n_group=1, embedding=None,
            attention=None, kernel=None):
        """Initialize.

        Arguments:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded. This must be equal to or
                greater than three.
            n_dim (optional): An integer indicating the dimensionality
                of the embedding. Must be equal to or greater than one.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group. Must be equal to or greater than one.
            embedding (optional): An embedding layer.
            attention (optional): An attention layer.
            kernel (optional): A similarity kernel layer.

        Raises:
            ValueError: If arguments are invalid.

        """
        super().__init__()

        if (n_stimuli < 3):
            raise ValueError("There must be at least three stimuli.")
        self.n_stimuli = n_stimuli
        if (n_dim < 1):
            raise ValueError(
                "The dimensionality (`n_dim`) must be an integer "
                "greater than 0."
            )
        self.n_dim = n_dim
        if (n_group < 1):
            raise ValueError(
                "The number of groups (`n_group`) must be an integer greater "
                "than 0."
            )
            n_group = 1
        self.n_group = n_group

        # Initialize model components.
        if embedding is None:
            embedding = psiz.keras.layers.Embedding(
                n_stimuli=self.n_stimuli, n_dim=self.n_dim
            )
        self.embedding = embedding

        if attention is None:
            attention = psiz.keras.layers.Attention(
                n_dim=self.n_dim, n_group=self.n_group
            )
        self.attention = attention

        if kernel is None:
            kernel = psiz.keras.layers.ExponentialKernel()
        self.kernel = kernel

        # Unsaved attributes.
        # TensorBoard log attributes.
        self.do_log = False
        self.log_dir = '/tmp/psiz/tensorboard_logs/'
        self.log_freq = 10
        # Compile attributes.
        self.optimizer = None
        self.loss = None

    def reset_weights(self):
        """Reinitialize trainable model parameters."""
        self.embedding.reset_weights()
        self.attention.reset_weights()
        self.kernel.reset_weights()

    @property
    def z(self):
        """Getter method for z."""
        return self.embedding.z.numpy()

    @z.setter
    def z(self, z):
        """Setter method for z."""
        self.embedding.z.assign(z)

    @property
    def w(self):
        """Getter method for phi."""
        return self.attention.w.numpy()

    @w.setter
    def w(self, w):
        """Setter method for w."""
        self.attention.w.assign(w)

    @property
    def phi(self):
        """Getter method for phi."""
        d = {
            'w': self.w
        }
        return d

    @phi.setter
    def phi(self, phi):
        """Setter method for w."""
        for k, v in phi.items():
            setattr(self, k, v)

    @property
    def theta(self):
        """Getter method for theta."""
        d = {}
        for k, v in self.kernel.theta.items():
            d[k] = v.numpy()
        return d

    @theta.setter
    def theta(self, theta):
        """Setter method for w."""
        for k, v in theta.items():
            self.kernel.theta[k].assign(v)

    def get_weights(self):
        """Get weights for all layers.

        Returns all weights as a dictionary of layers with each layer's
        weights as a single-level dictionary of weights.

        """
        weights = {
            'embedding': {'z': self.z},
            'attention': self.phi,
            'kernel': self.theta
        }
        return weights

    def set_weights(self, weights):
        """Setter method for all weights."""
        for layer_name, layer_dict in weights.items():
            layer = getattr(self, layer_name)
            for var_name, var_value in layer_dict.items():
                var = getattr(layer, var_name)
                var.assign(var_value)

    def _broadcast_for_similarity(
            self, z_q, z_r, group_id=None):
        """Return similarity between two lists of points.

        Similarity is determined using the similarity kernel and the
        current similarity parameters. This method implements the
        logic for handling arguments of different shapes.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim, [1, n_sample])
            z_r: A set of embedding points.
                shape = (n_trial, n_dim, [n_reference, n_sample])
            group_id (optional): The group ID for each sample. Can be a
                scalar or an array of shape = (n_trial,).

        Returns:
            The corresponding similarity between rows of embedding
                points.

        """
        n_trial = z_q.shape[0]

        # Handle group_id.
        if group_id is None:
            group_id = np.zeros((n_trial), dtype=np.int32)
        else:
            if np.isscalar(group_id):
                group_id = group_id * np.ones((n_trial), dtype=np.int32)
            else:
                group_id = group_id.astype(dtype=np.int32)

        attention = self.phi['w'][group_id, :]

        # Make sure z_q and attention have an appropriate singleton
        # dimensions.
        if z_r.ndim > 2:
            if z_q.ndim == 2:
                z_q = np.expand_dims(z_q, axis=2)
            if attention.ndim == 2:
                attention = np.expand_dims(attention, axis=2)
        if z_r.ndim == 4:
            # A fourth dimension means there are samples for each point.
            if z_q.ndim == 3:
                z_q = np.expand_dims(z_q, axis=3)
            if attention.ndim == 3:
                attention = np.expand_dims(attention, axis=3)

        return (z_q, z_r, attention)

    def similarity(self, z_q, z_r, group_id=None):
        """Return similarity between two lists of points.

        Similarity is determined using the similarity kernel and the
        current similarity parameters. This method implements the
        logic for handling arguments of different shapes.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim, [1, n_sample])
            z_r: A set of embedding points.
                shape = (n_trial, n_dim, [n_reference, n_sample])
            group_id (optional): The group ID for each sample. Can be a
                scalar or an array of shape = (n_trial,).

        Returns:
            The corresponding similarity between rows of embedding
                points.

        """
        (z_q, z_r, attention) = self._broadcast_for_similarity(
            z_q, z_r, group_id=group_id
        )
        sim_qr = self.kernel([
            tf.constant(z_q, dtype=K.floatx()),
            tf.constant(z_r, dtype=K.floatx()),
            tf.constant(attention, dtype=K.floatx())
        ]).numpy()
        return sim_qr

    def distance(self, z_q, z_r, group_id=None):
        """Return distance between two lists of points.

        Distance is determined using the weighted Minkowski metric.
        This method implements the logic for handling arguments of
        different shapes.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim, [1, n_sample])
            z_r: A set of embedding points.
                shape = (n_trial, n_dim, [n_reference, n_sample])
            group_id (optional): The group ID for each sample. Can be a
                scalar or an array of shape = (n_trial,).

        Returns:
            The corresponding similarity between rows of embedding
                points.

        """
        n_trial = z_q.shape[0]
        # Handle group_id.
        if group_id is None:
            group_id = np.zeros((n_trial), dtype=np.int32)
        else:
            if np.isscalar(group_id):
                group_id = group_id * np.ones((n_trial), dtype=np.int32)
            else:
                group_id = group_id.astype(dtype=np.int32)

        attention = self.w[group_id, :]

        # Make sure z_q and attention have an appropriate singleton
        # dimensions.
        if z_r.ndim > 2:
            if z_q.ndim == 2:
                z_q = np.expand_dims(z_q, axis=2)
            if attention.ndim == 2:
                attention = np.expand_dims(attention, axis=2)
        if z_r.ndim == 4:
            # A fourth dimension means there are samples for each point.
            if z_q.ndim == 3:
                z_q = np.expand_dims(z_q, axis=3)
            if attention.ndim == 3:
                attention = np.expand_dims(attention, axis=3)

        d_qr = self.kernel.distance_layer([
            tf.constant(z_q, dtype=K.floatx()),
            tf.constant(z_r, dtype=K.floatx()),
            tf.constant(attention, dtype=K.floatx())
        ]).numpy()
        return d_qr

    def set_log(self, do_log, log_dir=None, log_freq=None, delete_prev=True):
        """State changing method that sets TensorBoard logging.

        Arguments:
            do_log: Boolean that indicates whether logs should be
                recorded.
            log_dir (optional): A string indicating the file path for
                the logs.
            delete_prev (optional): Boolean indicating whether the
                directory should be cleared of previous files first.

        """
        if do_log:
            self.do_log = True

        if log_dir is not None:
            self.log_dir = log_dir

        if log_freq is not None:
            self.log_freq = log_freq

        if delete_prev:
            if tf.io.gfile.exists(self.log_dir):
                tf.io.gfile.rmtree(self.log_dir)
        tf.io.gfile.makedirs(self.log_dir)

    @abstractmethod
    def _build_model(self):
        """Build TensorFlow model.

        Returns:
            model: A tf.Model object.

        """
        pass

    def _check_obs(self, obs):
        """Check observerations.

        Arguments:
            obs: A psiz.trials.RankObservations object.

        Raises:
            ValueError

        """
        n_group_obs = np.max(obs.group_id) + 1
        if n_group_obs > self.n_group:
            raise ValueError(
                "The provided observations contain data from at least {0}"
                " groups. The present model only supports {1}"
                " group(s).".format(
                    n_group_obs, self.n_group
                )
            )

    def compile(self, optimizer=None, loss=None):
        """Configure the model for training.

        Arguments:
            optimizer: A tf.keras.optimizer object.
            loss: A loss function.

        Raises:
            ValueError: If arguments are invalid.

        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        self.optimizer = optimizer

        if loss is None:
            loss = _observation_loss
        self.loss = loss

    def fit(
            self, obs_train, batch_size=None, obs_val=None, epochs=1000,
            initial_epoch=0, callbacks=None, seed=None, verbose=0):
        """Fit the free parameters of the embedding model.

        Arguments:
            obs_train: An RankObservations object representing the observed
                data used to train the model.
            batch_size: The batch size to use for the training step.
            obs_val (optional): An RankObservations object representing the
                observed data used to validate the model.
            epochs (optional): The number of epochs to perform.
            initial_epoch (optional): The initial epoch.
            callbacks (optional): A list of TensorFlow callbacks.
            seed (optional): An integer to be used to seed the random number
                generator.
            verbose (optional): An integer specifying the verbosity of
                printed output. If zero, nothing is printed. Increasing
                integers display an increasing amount of information.

        Returns:
            history: A tf.callbacks.History object.

        """
        fit_start_time_s = time.time()

        # Determine batch size.
        n_obs_train = obs_train.n_trial
        if batch_size is None:
            batch_size_train = n_obs_train
        else:
            batch_size_train = np.minimum(batch_size, n_obs_train)

        # Create TensorFlow training Dataset.
        self._check_obs(obs_train)
        # Format as TensorFlow dataset.
        ds_obs_train = obs_train.as_dataset()
        ds_obs_train = ds_obs_train.shuffle(
            buffer_size=n_obs_train, reshuffle_each_iteration=True
        )
        ds_obs_train = ds_obs_train.batch(
            batch_size_train, drop_remainder=False
        )

        # Create TensorFlow validation Dataset (if necessary).
        if obs_val is not None:
            do_validation = True
            self._check_obs(obs_val)
            ds_obs_val = obs_val.as_dataset()
            n_obs_val = obs_val.n_trial
            # Format as TensorFlow dataset.
            ds_obs_val = ds_obs_val.batch(
                n_obs_val, drop_remainder=False
            )

        # Build TensorFlow model.
        model = self._build_model()

        callback_list = configure_callbacks(
            callbacks,
            model,
            do_validation=False,
            batch_size=None,
            epochs=None,
            steps_per_epoch=None,
            samples=None,
            verbose=0,
            count_mode='steps'
        )

        logs = {}
        metric_train_loss = tf.keras.metrics.Mean(name='loss')
        metric_val_loss = tf.keras.metrics.Mean(name='val_loss')
        summary_writer = tf.summary.create_file_writer(self.log_dir)

        # NOTE: Must bring into local scope in order for optimizer state
        # to update appropriately.
        optimizer = self.optimizer

        # NOTE: Trainable attention weights does not work with eager
        # execution.
        @tf.function
        def train_step(inputs):
            # Compute training loss and gradients.
            with tf.GradientTape() as grad_tape:
                probs = model(inputs)
                # Loss value for this minibatch.
                loss_value = self.loss(probs, inputs['weight'])
                # Add extra losses created during this forward pass.
                loss_value += sum(model.losses)

            gradients = grad_tape.gradient(
                loss_value, model.trainable_variables
            )
            # NOTE: This assumes equal number of samples for each minibatch.
            # The computed mean will deviate from correct if minibatch sizes
            # vary.
            metric_train_loss.update_state(loss_value)

            # NOTE: There is an open issue for using constraints with
            # tf.keras.layers.Embedding and psiz.keras.layers.Embedding (see:
            # https://github.com/tensorflow/tensorflow/issues/33755). There
            # are also issues when using Eager Execution. A work-around is
            # to convert the problematic gradients, which are returned as
            # tf.IndexedSlices, into dense tensors.
            for var_idx, var in enumerate(model.trainable_variables):
                if var.name == 'z:0':
                    gradients[var_idx] = tf.convert_to_tensor(
                        gradients[var_idx]
                    )

            # Apply gradients (subject to constraints).
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )

        @tf.function
        def eval_val_step(inputs):
            probs = model(inputs)
            # Loss value for this minibatch.
            loss_value = self.loss(probs, inputs['weight'])
            metric_val_loss.update_state(loss_value)

        @tf.function
        def eval_train_step(inputs):
            probs = model(inputs)
            # Loss value for this minibatch.
            loss_value = self.loss(probs, inputs['weight'])
            # Add extra losses created during this forward pass.
            loss_value += sum(model.losses)
            metric_train_loss.update_state(loss_value)

        callback_list.on_train_begin(logs=None)

        with summary_writer.as_default():
            epoch_start_time_s = time.time()
            for epoch in range(initial_epoch, epochs):
                if callback_list.model.stop_training:
                    epoch = epoch - 1
                    break
                else:
                    callback_list.on_epoch_begin(epoch, logs=None)

                    # Reset metrics at the start of each epoch.
                    metric_train_loss.reset_states()
                    metric_val_loss.reset_states()

                    # Compute training loss and update variables.
                    # NOTE: During computation of gradients, IndexedSlices are
                    # created which generates a TensorFlow warning. I cannot
                    # find an implementation that avoids IndexedSlices. The
                    # following catch environment silences the offending
                    # warning.
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            'ignore', category=UserWarning,
                            module=r'.*indexed_slices'
                        )
                        for batch_idx, batch_train in enumerate(ds_obs_train):
                            # callback_list.on_train_batch_begin(
                            #     batch_idx, logs=None
                            # )
                            train_step(batch_train)
                    logs['loss'] = metric_train_loss.result().numpy()

                    # Compute validation loss.
                    if do_validation:
                        for batch_val in ds_obs_val:
                            eval_val_step(batch_val)
                        logs['val_loss'] = metric_val_loss.result().numpy()

                    if verbose > 3:
                        if epoch % self.log_freq == 0:
                            _print_epoch_logs(epoch, logs)

                    # Add logs to summary for TensorBoard.
                    if self.do_log:
                        if epoch % self.log_freq == 0:
                            for k, v in logs.items():
                                tf.summary.scalar(k, v, step=epoch)

                    callback_list.on_epoch_end(epoch, logs=logs)
                summary_writer.flush()
            epoch_stop_time_s = time.time() - epoch_start_time_s
        callback_list.on_train_end(logs=None)

        # Add final model metrics to a `final_logs` dictionary.
        # NOTE: If there is an early stopping callback with
        # restore_best_weights, then the final evaluation will use
        # the best weights.
        final_logs = {}

        # Reset metrics.
        metric_train_loss.reset_states()
        metric_val_loss.reset_states()

        for batch_train in ds_obs_train:
            eval_train_step(batch_train)
        final_logs['loss'] = metric_train_loss.result().numpy()

        if do_validation:
            for batch_val in ds_obs_val:
                eval_val_step(batch_val)
            final_logs['val_loss'] = metric_val_loss.result().numpy()

        # Add time information.
        total_duration = time.time() - fit_start_time_s
        final_logs['total_duration_s'] = int(total_duration)
        final_logs['ms_per_epoch'] = int(1000 * epoch_stop_time_s / epoch)

        if (verbose > 2):
            _print_epoch_logs(epoch, final_logs)
            print('')

        # Add epoch after print statement.
        final_logs['epoch'] = epoch

        # Piggy-back on History object.
        model.history.final = final_logs

        return model.history

    def evaluate(self, obs, batch_size=None):
        """Evaluate observations using the current state of the model.

        Notes:
            Observations are evaluated in test mode. This means that
            regularization terms are not included in the loss.

        Arguments:
            obs: A RankObservations object representing the observed data.
            batch_size (optional): Integer indicating the batch size.

        Returns:
            loss: The average loss per observation. Loss is defined as
                the negative loglikelihood.

        """
        self._check_obs(obs)
        ds_obs = obs.as_dataset()

        model = self._build_model()

        if batch_size is None:
            batch_size = obs.n_trial

        ds_obs = ds_obs.batch(
            batch_size, drop_remainder=False
        )

        metric_loss = tf.keras.metrics.Mean(name='loss')

        @tf.function
        def eval_step(inputs):
            # Compute validation loss.
            probs = model(inputs)
            loss_value = self.loss(probs, inputs['weight'])
            metric_loss.update_state(loss_value)

        for batch in ds_obs:
            eval_step(batch)
        loss = metric_loss.result()

        return loss

    def outcome_probability(
            self, docket, group_id=None, z=None, unaltered_only=False):
        """Return probability of each outcome for each trial.

        Arguments:
            docket: A docket of unjudged similarity trials. The indices
                used must correspond to the rows of z.
            group_id (optional): The group ID for which to compute the
                probabilities.
            z (optional): A set of embedding points. If no embedding
                points are provided, the points associated with the
                object are used.
                shape=(n_stimuli, n_dim, [n_sample])
            unaltered_only (optional): Flag that determines whether
                only the unaltered ordering is evaluated and returned.

        Returns:
            prob_all: A MaskedArray representing the probabilities
                associated with the different outcomes for each
                unjudged trial. In general, different trial
                configurations have a different number of possible
                outcomes. The mask attribute of the MaskedArray
                indicates which elements are actual outcome
                probabilities.
                shape = (n_trial, n_max_outcome, [n_sample])

        Notes:
            The first outcome corresponds to the original order of the
                trial data.

        """
        n_trial_all = docket.n_trial

        if z is None:
            z = self.z

        n_config = docket.config_list.shape[0]

        outcome_idx_list = docket.outcome_idx_list
        n_outcome_list = docket.config_list['n_outcome'].values
        max_n_outcome = np.max(n_outcome_list)

        if unaltered_only:
            max_n_outcome = 1

        if z.ndim == 2:
            z = np.expand_dims(z, axis=2)
        n_sample = z.shape[2]

        # Compute similarity between query and references.
        z_stimulus_set = _inflate_points(docket.stimulus_set, z)
        z_q = z_stimulus_set[:, :, 0, :]
        z_q = np.expand_dims(z_q, axis=2)
        z_r = z_stimulus_set[:, :, 1:, :]
        z_q, z_r, attention = self._broadcast_for_similarity(
            z_q, z_r, group_id=group_id
        )

        sim_qr = self.kernel([
            tf.constant(z_q, dtype=K.floatx()),
            tf.constant(z_r, dtype=K.floatx()),
            tf.constant(attention, dtype=K.floatx())
        ]).numpy()

        prob_all = -1 * np.ones((n_trial_all, max_n_outcome, n_sample))
        for i_config in range(n_config):
            config = docket.config_list.iloc[i_config]
            outcome_idx = outcome_idx_list[i_config]
            trial_locs = docket.config_idx == i_config
            n_trial = np.sum(trial_locs)
            n_reference = config['n_reference']

            sim_qr_config = sim_qr[trial_locs]
            sim_qr_config = sim_qr_config[:, 0:n_reference]

            n_outcome = n_outcome_list[i_config]
            if unaltered_only:
                n_outcome = 1

            # Compute probability of each possible outcome.
            probs_config = np.ones(
                (n_trial, n_outcome, n_sample), dtype=np.float64
            )
            # TODO (maybe faster) stack permutations, run
            # _ranked_sequence_probability once and then reshape.
            for i_outcome in range(n_outcome):
                s_qr_perm = sim_qr_config[:, outcome_idx[i_outcome, :], :]
                probs_config[:, i_outcome, :] = _ranked_sequence_probability(
                    s_qr_perm, config['n_select']
                )
            prob_all[trial_locs, 0:n_outcome, :] = probs_config
        prob_all = ma.masked_values(prob_all, -1)

        # Correct for any numerical inaccuracy.
        if not unaltered_only:
            prob_all = ma.divide(
                prob_all, ma.sum(prob_all, axis=1, keepdims=True))

        # Reshape prob_all as necessary.
        if n_sample == 1:
            prob_all = prob_all[:, :, 0]

        return prob_all

    def posterior_samples(
            self, obs, n_final_sample=1000, n_burn=100, thin_step=5,
            z_init=None, verbose=0):
        """Sample from the posterior of the embedding.

        Samples are drawn from the posterior holding theta constant. A
        variant of Elliptical Slice Sampling (Murray & Adams 2010) is
        used to estimate the posterior for the embedding points. Since
        the latent embedding variables are translation and rotation
        invariant, generic sampling will artificially inflate the
        entropy of the samples. To compensate for this issue, the
        points are split into two groups, holding one set constant
        while sampling the other set.

        Arguments:
            obs: A RankObservations object representing the observed data.
                There must be at least one observation in order to
                sample from the posterior distribution.
            n_final_sample (optional): The number of samples desired
                after removing the "burn in" samples and applying
                thinning.
            n_burn (optional): The number of samples to remove from the
                beginning of the sampling sequence.
            thin_step (optional): The interval to use in order to thin
                (i.e., de-correlate) the samples.
            z_init (optional): Initialization of z. If not provided,
                the current embedding values associated with the object
                are used.
            verbose (optional): An integer specifying the verbosity of
                printed output. If zero, nothing is printed. Increasing
                integers display an increasing amount of information.

        Returns:
            A dictionary of posterior samples for different parameters.
                The samples are stored as a NumPy array.
                'z' : shape = (n_stimuli, n_dim, n_total_sample).

        Notes:
            The step_size of the Hamiltonian Monte Carlo procedure is
                determined by the scale of the current embedding.

        References:
            Murray, I., & Adams, R. P. (2010). Slice sampling
            covariance hyperparameters of latent Gaussian models. In
            Advances in Neural Information Processing Systems (pp.
            1732-1740).

        """
        # Timer attributes. TODO
        # posterior_duration: The duration (in seconds) of the last
        #     called posterior sampling procedure.
        # self.posterior_duration = 0.0
        # change references to .posterior_duration

        start_time_s = time.time()
        n_final_sample = int(n_final_sample)
        n_total_sample = n_burn + (n_final_sample * thin_step)
        n_stimuli = self.n_stimuli
        n_dim = self.n_dim
        if z_init is None:
            z = copy.copy(self.z)
        else:
            z = z_init

        if verbose > 0:
            print('[psiz] Sampling from posterior...')
            progbar = psiz.utils.ProgressBar(
                n_total_sample, prefix='Progress:', length=50
            )
            progbar.update(0)

        if (verbose > 1):
            print('    Settings:')
            print('    n_total_sample: ', n_total_sample)
            print('    n_burn:         ', n_burn)
            print('    thin_step:      ', thin_step)
            print('    --------------------------')
            print('    n_final_sample: ', n_final_sample)

        # Prior
        # p(z_k | Z_negk, theta) ~ N(mu, sigma)
        # Approximate prior of z_k using all embedding points to reduce
        # computational burden.
        gmm = mixture.GaussianMixture(
            n_components=1, covariance_type='spherical'
        )
        gmm.fit(z)
        mu = np.expand_dims(gmm.means_[0], axis=0)
        sigma = gmm.covariances_[0] * np.identity(n_dim)
        # NOTE: Since the covariance is spherical, we just need one element.
        chol_element = np.linalg.cholesky(sigma)[0, 0]

        # Center embedding to satisfy assumptions of elliptical slice sampling.
        z = z - mu

        # Define log-likelihood for elliptical slice sampler.
        def log_likelihood(z_part, part_idx, z_full, obs):
            # Assemble full z.
            z_full[part_idx, :] = z_part
            cap = 2.2204e-16
            probs = self.outcome_probability(
                obs, group_id=obs.group_id, z=z_full,
                unaltered_only=True
            )
            probs = ma.maximum(cap, probs[:, 0])
            ll = ma.sum(ma.log(probs))
            return ll

        # Initialize sampler.
        z_full = copy.copy(z)
        samples = np.empty((n_stimuli, n_dim, n_total_sample))

        # Make first partition.
        n_partition = 2
        part_idx, n_stimuli_part = self._make_partition(
            n_stimuli, n_partition
        )

        for i_round in range(n_total_sample):

            if np.mod(i_round, 100) == 0:
                if verbose > 0:
                    progbar.update(i_round + 1)

            # Partition stimuli into two groups to fix rotation invariance.
            if np.mod(i_round, 10) == 0:
                part_idx, n_stimuli_part = self._make_partition(
                    n_stimuli, n_partition
                )

            for i_part in range(n_partition):
                z_part = z_full[part_idx[i_part], :]
                # Sample.
                (z_part, _) = _elliptical_slice(
                    z_part, chol_element, log_likelihood,
                    pdf_params=[part_idx[i_part], copy.copy(z), obs]
                )
                # Update.
                z_full[part_idx[i_part], :] = z_part

            samples[:, :, i_round] = z_full

        # Add back in mean.
        mu = np.expand_dims(mu, axis=2)
        samples = samples + mu

        samples_all = samples[:, :, n_burn::thin_step]
        samples_all = samples_all[:, :, 0:n_final_sample]
        samples = dict(z=samples_all)

        if verbose > 0:
            progbar.update(n_total_sample)

        self.posterior_duration = time.time() - start_time_s
        return samples

    @staticmethod
    def _make_partition(n_stimuli, n_partition):
        """Partition stimuli.

        Arguments:
            n_stimuli: Scalar indicating the total number of stimuli.
            n_partition: Scalar indicating the number of partitions.

        Returns:
            part_idx: A boolean array indicating partition membership.
                shape = (n_partition, n_stimuli)
            n_stimuli_part: An integer array indicating the number of
                stimuli in each partition.
                shape = (n_partition)

        """
        n_stimuli_part = np.floor(n_stimuli / n_partition)
        n_stimuli_part = n_stimuli_part * np.ones([n_partition])
        n_stimuli_part[1] = n_stimuli_part[1] + (
            n_stimuli - (n_stimuli_part[1] * n_partition)
        )
        n_stimuli_part = n_stimuli_part.astype(np.int32)

        partition = np.empty([0])
        for i_part in range(n_partition):
            partition = np.hstack(
                (partition, i_part * np.ones([n_stimuli_part[i_part]]))
            )
        partition = np.random.choice(partition, n_stimuli, replace=False)

        part_idx = np.zeros((n_partition, n_stimuli), dtype=np.int32)
        for i_part in range(n_partition):
            locs = np.equal(partition, i_part)
            part_idx[i_part, locs] = 1
        part_idx = part_idx.astype(bool)

        return part_idx, n_stimuli_part

    def save(self, filepath):
        """Save the PsychologialEmbedding model as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the model.

        """
        f = h5py.File(filepath, "w")
        f.create_dataset("embedding_type", data=type(self).__name__)
        f.create_dataset("n_stimuli", data=self.n_stimuli)
        f.create_dataset("n_dim", data=self.n_dim)
        f.create_dataset("n_group", data=self.n_group)

        # Save model architecture.
        grp_arch = f.create_group('architecture')
        # Create group for embedding layer.
        grp_coord = grp_arch.create_group('embedding')
        _add_layer_to_save_architecture(grp_coord, self.embedding)
        # Create group for attention layer.
        grp_attention = grp_arch.create_group('attention')
        _add_layer_to_save_architecture(grp_attention, self.attention)
        # Create group for kernel layer.
        grp_kernel = grp_arch.create_group('kernel')
        _add_layer_to_save_architecture(grp_kernel, self.kernel)

        # Save weights.
        weights = self.get_weights()
        grp_weights = f.create_group("weights")
        for layer_name, layer_dict in weights.items():
            grp_layer = grp_weights.create_group(layer_name)
            for k, v in layer_dict.items():
                grp_layer.create_dataset(k, data=v)

        f.close()

    def subset(self, idx):
        """Return subset of embedding."""
        emb = copy.deepcopy(self)
        emb.z = emb.z[idx, :]
        emb.n_stimuli = emb.z.shape[0]
        return emb

    def view(self, group_id):
        """Return a view-specific embedding.

        The returned embedding contains information only about the
        requested group. The embedding is appropriately adjusted such
        that the group-specific parameters are rolled into the other
        parameters. Specifically the embedding points are adjusted to
        account for the attention weights, and the attention weights
        are returned to ones. This function is useful if you would like
        to visualize and compare how group-specific embeddings differ
        in terms of perceived similarity.

        Arguments:
            group_id: Scalar or list. If scale, indicates the group_id
                to use. If a list, should be a list of group_id's to
                average.

        Returns:
            emb: A group-specific embedding.

        """
        emb = copy.deepcopy(self)
        z = self.z
        rho = self.rho
        if np.isscalar(group_id):
            attention_weights = self.w[group_id, :]
        else:
            group_id = np.asarray(group_id)
            attention_weights = self.w[group_id, :]
            attention_weights = np.mean(attention_weights, axis=0)

        z_group = z * np.expand_dims(attention_weights**(1/rho), axis=0)
        emb.z = z_group
        emb.n_group = 1
        emb.w = np.ones([1, self.n_dim])
        return emb

    def __deepcopy__(self, memodict={}):
        """Override deepcopy method."""
        # TODO CRITICAL test
        # embedding = copy.deepcopy(self.embedding, memodict)
        cpyobj = type(self)(
            self.n_stimuli, n_dim=self.n_dim, n_group=self.n_group,
            embedding=self.embedding, attention=self.attention,
            kernel=self.kernel
        )

        cpyobj.do_log = self.do_log
        cpyobj.log_dir = self.log_dir
        cpyobj.log_freq = self.log_freq

        return cpyobj


class Rank(PsychologicalEmbedding):
    """An embedding model that uses anchored, ordinal judgments."""

    def __init__(
            self, n_stimuli, n_dim=2, n_group=1, embedding=None,
            attention=None, kernel=None):
        """Initialize."""
        PsychologicalEmbedding.__init__(
            self, n_stimuli, n_dim=n_dim, n_group=n_group,
            embedding=embedding,
            attention=attention, kernel=kernel
        )

    def _build_model(self):
        """Build TensorFlow model.

        Returns:
            model: A tf.Model object.

        """
        model = RankModel(
            self.embedding, self.attention, self.kernel, name='rank_model'
        )
        return model


class Rate(PsychologicalEmbedding):
    """An embedding model that uses pair-wise ratings."""

    def __init__(
            self, n_stimuli, n_dim=2, n_group=1, embedding=None,
            attention=None, kernel=None):
        """Initialize."""
        PsychologicalEmbedding.__init__(
            self, n_stimuli, n_dim=n_dim, n_group=n_group,
            embedding=embedding,
            attention=attention, kernel=kernel
        )

    def _build_model(self):
        """Build TensorFlow model.

        Returns:
            model: A tf.Model object.

        """
        model = RateModel(
            self.embedding, self.attention, self.kernel, name='rank_model'
        )
        return model


class Sort(PsychologicalEmbedding):
    """An embedding model that sorted similarity judgments."""

    def __init__(
            self, n_stimuli, n_dim=2, n_group=1, embedding=None,
            attention=None, kernel=None):
        """Initialize."""
        PsychologicalEmbedding.__init__(
            self, n_stimuli, n_dim=n_dim, n_group=n_group,
            embedding=embedding,
            attention=attention, kernel=kernel
        )

    def _build_model(self):
        """Build TensorFlow model.

        Returns:
            model: A tf.Model object.

        """
        model = SortModel(
            self.embedding, self.attention, self.kernel, name='rank_model'
        )
        return model


class RankModel(tf.keras.Model):
    """Model based on ranked similarity judgments."""

    def __init__(self, embedding, attention, kernel, **kwargs):
        """Initialize.

        Arguments:
            embedding: An embedding layer.
            attention: An attention layer.
            kernel: A kernel layer.

        """
        super().__init__(**kwargs)

        self.embedding = embedding
        self.attention = attention
        self.kernel = kernel

    @tf.function
    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A dictionary of inputs:
                stimulus_set: dtype=tf.int32, consisting of the
                    integers on the interval [0, n_stimuli[
                    shape=(batch_size, n_max_reference + 1)
                group_id: dtype=tf.int32, consisting of the
                    integers on the interval [0, n_group[
                    shape=(batch_size,)
                is_present: dtype=tf.bool
                    shape=(batch_size, n_max_reference + 1)
                is_select: dtype=tf.bool, the shape implies the
                    maximum number of selected stimuli in the data
                    shape=(batch_size, n_max_select)

        """
        # Grab inputs.
        obs_stimulus_set = inputs['stimulus_set']
        obs_group_id = inputs['group_id']
        is_present = inputs['is_present']
        is_select = inputs['is_select']

        # Expand attention weights.
        attention = self.attention(obs_group_id)

        # Inflate cooridnates.
        z_stimulus_set = self.embedding(obs_stimulus_set)
        max_n_reference = tf.shape(z_stimulus_set)[2] - 1
        z_q, z_r = tf.split(z_stimulus_set, [1, max_n_reference], 2)

        # Compute similarity between query and references.
        sim_qr = self.kernel([z_q, z_r, attention])

        # Zero out similarities involving placeholder.
        sim_qr = sim_qr * tf.cast(is_present[:, 1:], dtype=K.floatx())

        # Compute the observation likelihood.
        likelihood = _tf_ranked_sequence_probability(sim_qr, is_select)
        return likelihood

    def reset_weights(self):
        """Reset trainable variables."""
        self.embedding.reset_weights()
        self.attention.reset_weights()
        self.kernel.reset_weights()


# class RateModel(tf.keras.Model):


# class SortModel(tf.keras.Model):


def load_model(filepath, custom_objects={}):
    """Load embedding model saved via the save method.

    The loaded data is instantiated as a concrete class of
    psiz.trials.Trials.

    Arguments:
        filepath: The location of the hdf5 file to load.
        custom_objects (optional): A dictionary mapping the string
            class name to the Python class

    Returns:
        Loaded embedding model.

    Raises:
        ValueError

    """
    f = h5py.File(filepath, 'r')
    # Common attributes.
    embedding_type = f['embedding_type'][()]
    n_stimuli = f['n_stimuli'][()]
    n_dim = f['n_dim'][()]
    n_group = f['n_group'][()]

    if embedding_type == 'Rank':
        grp_architecture = f['architecture']
        # Instantiate embedding layer.
        embedding = _load_layer(
            grp_architecture['embedding'], custom_objects
        )
        # Instantiate attention layer.
        attention = _load_layer(
            grp_architecture['attention'], custom_objects
        )
        # Instantiate kernel layer.
        kernel = _load_layer(
            grp_architecture['kernel'], custom_objects
        )

        emb = Rank(
            n_stimuli, n_dim=n_dim, n_group=n_group,
            embedding=embedding, attention=attention,
            kernel=kernel
        )

        # Set weights.
        grp_weights = f['weights']
        # Assemble dictionary of weights.
        weights = {}
        for layer_name, grp_layer in grp_weights.items():
            layer_weights = {}
            for var_name in grp_layer:
                layer_weights[var_name] = grp_weights[layer_name][var_name][()]
            weights[layer_name] = layer_weights
        emb.set_weights(weights)

    else:
        # Create embedding layer.
        z = f['z']['value'][()]
        fit_z = f['z']['trainable'][()]
        embedding = Embedding(
            n_stimuli=n_stimuli, n_dim=n_dim, fit_z=fit_z
        )

        # Create attention layer.
        if 'phi_1' in f['phi']:
            fit_group = f['phi']['phi_1']['trainable']
            w = f['phi']['phi_1']['value']
        else:
            fit_group = f['phi']['w']['trainable']
            w = f['phi']['w']['value']
        attention = Attention(
            n_dim=n_dim, n_group=n_group, fit_group=fit_group
        )
        # OLD code for reference.
        # for p_name in f['phi']:
        #     # Patch for older models using `phi_1` variable name.
        #     if p_name == 'phi_1':
        #         p_name_new = 'w'
        #     else:
        #         p_name_new = p_name
        #     for name in f['phi'][p_name]:
        #         embedding._phi[p_name_new][name] = f['phi'][p_name][name][()]

        # Create kernel layer.
        theta_config = {}
        theta_value = {}
        for p_name in f['theta']:
            theta_config['fit_' + p_name] = f['theta'][p_name]['trainable'][()]
            theta_value[p_name] = f['theta'][p_name]['value'][()]
            # for name in f['theta'][p_name]:
            #     embedding._theta[p_name][name] = f['theta'][p_name][name][()]

        if embedding_type == 'Exponential':
            kernel = ExponentialKernel(**theta_config)
        elif embedding_type == 'HeavyTailed':
            kernel = HeavyTailedKernel(**theta_config)
        elif embedding_type == 'StudentsT':
            kernel = StudentsTKernel(**theta_config)
        elif embedding_type == 'Inverse':
            kernel = InverseKernel(**theta_config)
        else:
            raise ValueError(
                'No class found matching the provided `embedding_type`.'
            )

        emb = Rank(
            n_stimuli, n_dim=n_dim, n_group=n_group,
            embedding=embedding, attention=attention,
            kernel=kernel
        )

        # Set weights.
        emb.z = z
        emb.w = w
        emb.theta = theta_value

    f.close()
    return emb


def _add_layer_to_save_architecture(grp_layer, layer):
    """Add layer information to layer group.

    Arguments:
        grp_layer: An HDF5 group.
        layer: A TensorFlow layer with a `get_config` method.

    """
    grp_layer.create_dataset(
        'class_name', data=type(layer).__name__
    )
    grp_config = grp_layer.create_group('config')
    layer_config = layer.get_config()
    for k, v in layer_config.items():
        if v is not None:
            grp_config.create_dataset(k, data=v)


def _load_layer(grp_layer, custom_objects):
    """Load a configured layer.

    Arguments:
        grp_layer: An HDF5 group.
        custom_objects: A list of custom classes.

    Returns:
        layer: An instantiated and configured TensorFlow layer.

    """
    layer_class_name = grp_layer['class_name'][()]
    layer_config = {}
    for k in grp_layer['config']:
        layer_config[k] = grp_layer['config'][k][()]

    if layer_class_name in custom_objects:
        layer_class = custom_objects[layer_class_name]
    else:
        layer_class = getattr(psiz.keras.layers, layer_class_name)
    return layer_class.from_config(layer_config)


def _elliptical_slice(
        initial_theta, chol_element, lnpdf, pdf_params=(), angle_range=None):
    """Return samples from elliptical slice sampler.

    Markov chain update for a distribution with a Gaussian "prior"
    factored out. This slice function assumes a spherical Gaussian and
    takes advantage of the simplified math in order to compute `nu`.

    Arguments:
        initial_theta: initial vector
        chol_element: The diagonal element of the cholesky decomposition of
            the covariance matrix (like what numpy.linalg.cholesky
            returns).
        lnpdf: function evaluating the log of the pdf to be sampled
        pdf_params: parameters to pass to the pdf
        angle_range: Default 0: explore whole ellipse with break point
            at first rejection. Set in (0,2*pi] to explore a bracket of
            the specified width centred uniformly at random.

    Returns:
        new_theta, new_lnpdf

    History:
        Originally written in MATLAB by Iain Murray
        (http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m)
        2012-02-24 - Written - Bovy (IAS)

    """
    cur_lnpdf = lnpdf(initial_theta, *pdf_params)

    # Compute nu.
    theta_shape = initial_theta.shape
    nu = chol_element * np.random.normal(size=theta_shape)

    # Set up slice threshold.
    hh = np.log(np.random.uniform()) + cur_lnpdf

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range is None or angle_range == 0.:
        # Bracket whole ellipse with both edges at first proposed point
        phi = np.random.uniform() * 2. * np.pi
        phi_min = phi - 2. * np.pi
        phi_max = phi
    else:
        # Randomly center bracket on current point
        phi_min = -1 * angle_range * np.random.uniform()
        phi_max = phi_min + angle_range
        phi = np.random.uniform() * (phi_max - phi_min) + phi_min

    # Slice sampling loop.
    while True:
        # Compute theta for proposed angle difference and check if it's on the
        # slice.
        theta_prop = initial_theta * np.cos(phi) + nu * np.sin(phi)
        cur_lnpdf = lnpdf(theta_prop, *pdf_params)
        if cur_lnpdf > hh:
            # New point is on slice, ** EXIT LOOP **
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise RuntimeError(
                'BUG DETECTED: Shrunk to current position and still not',
                ' acceptable.'
            )
        # Propose new angle difference.
        phi = np.random.uniform() * (phi_max - phi_min) + phi_min

    return (theta_prop, cur_lnpdf)


def _inflate_points(stimulus_set, z):
    """Inflate stimulus set into embedding points.

    Arguments:
        stimulus_set: Array of integers indicating the stimuli used
            in each trial.
            shape = (n_trial, input_length)
        z: shape = (n_stimuli, n_dim, n_sample)

    Returns:
        z_q: shape = (n_trial, n_dim, 1, n_sample)
        z_r: shape = (n_trial, n_dim, n_reference, n_sample)

    """
    n_trial = stimulus_set.shape[0]
    input_length = stimulus_set.shape[1]
    n_dim = z.shape[1]
    n_sample = z.shape[2]

    # Increment stimuli indices and add placeholder stimulus.
    stimulus_set_temp = (stimulus_set + 1).ravel()
    z_placeholder = np.zeros((1, n_dim, n_sample))
    z_temp = np.concatenate((z_placeholder, z), axis=0)

    # Inflate points.
    z_stimulus_set = z_temp[stimulus_set_temp, :, :]
    z_stimulus_set = np.transpose(
        np.reshape(z_stimulus_set, (n_trial, input_length, n_dim, n_sample)),
        (0, 2, 1, 3)
    )

    return z_stimulus_set


def _ranked_sequence_probability(sim_qr, n_select):
    """Return probability of a ranked selection sequence.

    Arguments:
        sim_qr: A 3D tensor containing pairwise similarity values.
            Each row (dimension 0) contains the similarity between
            a trial's query stimulus and reference stimuli. The
            tensor is arranged such that the first column
            corresponds to the first selection in a sequence, and
            the last column corresponds to the last selection
            (dimension 1). The third dimension indicates
            different samples.
            shape = (n_trial, n_reference, n_sample)
        n_select: Scalar indicating the number of selections made
            by an agent.

    Returns:
        A 2D tensor of probabilities.
        shape = (n_trial, n_sample)

    Notes:
        For example, given query Q and references A, B, and C, the
        probability of selecting reference A then B (in that order)
        would be:

        P(A)P(B|A) = s_QA/(s_QA + s_QB + s_QC) * s_QB/(s_QB + s_QC)

        where s_QA denotes the similarity between the query and
        reference A.

        The probability is computed by starting with the last
        selection for efficiency and numerical stability. In the
        provided example, this corresponds to first computing the
        probability of selecting B second, given that A was
        selected first.

    """
    n_trial = sim_qr.shape[0]
    n_sample = sim_qr.shape[2]

    # Initialize.
    seq_prob = np.ones((n_trial, n_sample), dtype=np.float64)
    selected_idx = n_select - 1
    denom = np.sum(sim_qr[:, selected_idx:, :], axis=1)

    for i_selected in range(selected_idx, -1, -1):
        # Compute selection probability.
        prob = np.divide(sim_qr[:, i_selected], denom)
        # Update sequence probability.
        # seq_prob = np.multiply(seq_prob, prob)
        seq_prob *= prob
        # Update denominator in preparation for computing the probability
        # of the previous selection in the sequence.
        if i_selected > 0:
            # denom = denom + sim_qr[:, i_selected-1, :]
            denom += sim_qr[:, i_selected-1, :]
    return seq_prob


def _print_epoch_logs(epoch, logs):
    """Print epoch logs."""
    msg = (
        '        '
        'epoch {0:5d}'.format(epoch)
    )
    for k, v in logs.items():
        msg += ' | {0}: {1}'.format(k, str(v))
    print(msg)


@tf.function(experimental_relax_shapes=True)
def _observation_loss(y_pred, sample_weight):
    """Compute model loss given observation probabilities."""
    # Convert to (weighted) log probabilities.
    y_pred = _safe_neg_log_prob(y_pred)
    y_pred = tf.multiply(sample_weight, y_pred)

    # Get trial mean.
    loss = tf.reduce_mean(y_pred, axis=0)
    return loss


def _safe_neg_log_prob(probs):
    """Convert to safe log probabilites."""
    cap = tf.constant(2.2204e-16, dtype=K.floatx())
    return tf.negative(tf.math.log(tf.maximum(probs, cap)))


def _tf_ranked_sequence_probability(sim_qr, is_select):
    """Return probability of a ranked selection sequence.

    See: _ranked_sequence_probability

    Arguments:
        sim_qr: A tensor containing the precomputed similarities
            between the query stimuli and corresponding reference
            stimuli.
            shape = (batch_size, n_max_reference)
        is_select: A Boolean tensor indicating if a reference was
            selected.
            shape = (batch_size, n_max_select)

    """
    # Determine batch_size.
    batch_size = tf.shape(sim_qr)[0]
    # Determine max_select_idx (i.e, max_n_select - 1).
    max_select_idx = tf.shape(is_select)[1] - 1

    # Pre-allocate
    seq_prob = tf.ones([batch_size], dtype=K.floatx())

    # Compute denominator of Luce's choice rule.
    # Start by computing denominator of last selection
    denom = tf.reduce_sum(sim_qr[:, max_select_idx:], axis=1)

    # Pre-compute masks for handling non-existent selections.
    does_exist = tf.cast(is_select, dtype=K.floatx())
    does_not_exist = tf.cast(tf.math.logical_not(is_select), dtype=K.floatx())

    # Compute remaining denominators in reverse order for numerical
    # stability.
    for select_idx in tf.range(max_select_idx, -1, -1):
        # Compute selection probability.
        # Use safe divide since some denominators may be zero.
        prob = tf.math.divide_no_nan(sim_qr[:, select_idx], denom)
        # Zero out non-existent selections.
        prob = prob * does_exist[:, select_idx]
        # Add one to non-existent selections.
        prob = prob + does_not_exist[:, select_idx]

        # Update sequence probability.
        # NOTE: Non-existent selections will have a probability of 1, which
        # results in an idempotent multiplication operation.
        seq_prob = tf.multiply(seq_prob, prob)

        # Update denominator in preparation for computing the probability
        # of the previous selection in the sequence.
        if select_idx > tf.constant(0, dtype=tf.int32):
            denom = tf.add(denom, sim_qr[:, select_idx - 1])
        # seq_prob.set_shape([None])  # Not necessary any more?

    return seq_prob
