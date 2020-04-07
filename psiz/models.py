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
    Exponential: Embedding model using an exponential family similarity
        kernel.
    HeavyTailed: Embedding model using a heavy-tailed similarity
        kernel.
    StudentsT: Embedding model using a Student's t similarity kernel.

Functions:
    load_embedding: Load a hdf5 file, that was saved with the `save`
        class method, as a PsychologicalEmbedding object.
    load: An alias for load_embedding.

"""

from abc import ABCMeta, abstractmethod
import ast
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
from tensorflow.keras.layers import Layer
import tensorflow.keras.optimizers
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import BaseLogger, History
from tensorflow.python.keras.callbacks import configure_callbacks
from tensorflow.keras.constraints import Constraint

import psiz.trials
import psiz.utils


class PsychologicalEmbedding(metaclass=ABCMeta):
    """Abstract base class for a psychological embedding.

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
        posterior_duration: The duration (in seconds) of the last
            called posterior sampling procedure.

    """

    def __init__(
            self, n_stimuli, n_dim=2, n_group=1, coordinate=None,
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
            coordinate (optional): A coordinate layer.
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

        # TODO CRITICAL to save/load functionality of below.

        # Initialize model components.
        if coordinate is None:
            coordinate = Coordinate(n_stimuli=self.n_stimuli, n_dim=self.n_dim)
        self.coordinate = coordinate

        if attention is None:
            attention = Attention(n_dim=self.n_dim, n_group=self.n_group)
        self.attention = attention

        if kernel is None:
            kernel = ExponentialKernel()
        self.kernel = kernel

        # Default TensorBoard log attributes.
        self.do_log = False
        self.log_dir = '/tmp/psiz/tensorboard_logs/'
        self.log_freq = 10

        # Timer attributes. TODO
        self.posterior_duration = 0.0

        # Optimizer attributes.
        self.optimizer = None
        self.loss = None

    def reset_weights(self):
        """Reinitialize trainable model parameters."""
        self.coordinate.reset_weights()
        self.attention.reset_weights()
        self.kernel.reset_weights()

    @property
    def z(self):
        """Getter method for z."""
        return self.coordinate.z.numpy()

    @z.setter
    def z(self, z):
        """Setter method for z."""
        self.coordinate.z.assign(z)

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
        """Getter method for all weights."""
        d = {
            'z': self.z,
            'phi': self.phi,
            'theta': self.theta
        }
        return d

    def set_weights(self, w):
        """Setter method for all weights."""
        for k, v in w.items():
            setattr(self, k, v)

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

    def _prepare_and_build(self, obs):
        """Prepare inputs and build TensorFlow model.

        Arguments:
            obs: A psiz.trials.Observations object.

        Returns:
            ds_obs: The observations formatted as a tf.data.Dataset
                object.
            model: A tf.Model object.

        Notes:
            Ideally preparing the inputs and building a model would
            be decoupled. However, construction of the static graph
            requires knowledge of the different trial configurations
            present in the data.

        """
        # Create dataset.
        ds_obs = tf.data.Dataset.from_tensor_slices({
            'stimulus_set': obs.stimulus_set,
            'config_idx': obs.config_idx,
            'group_id': obs.group_id,
            'weight': tf.constant(obs.weight, dtype=K.floatx()),
            'is_present': obs.is_present()
        })

        # Initialize model likelihood layer.
        likelihood_layer = QueryReference(
            self.coordinate, self.attention, self.kernel,
            obs.config_list
        )

        # Define model.
        inputs = [
            tf.keras.Input(
                shape=[None], name='stimulus_set', dtype=tf.int32,
            ),
            tf.keras.Input(
                shape=[], name='config_idx', dtype=tf.int32,
            ),
            tf.keras.Input(
                shape=[], name='group_id', dtype=tf.int32,
            ),
            tf.keras.Input(
                shape=[], name='weight', dtype=K.floatx(),
            ),
            tf.keras.Input(
                shape=[None], name='is_present', dtype=tf.bool
            )
        ]
        output = likelihood_layer(inputs)
        model = tf.keras.models.Model(inputs, output, name='anchored_ordinal')

        return ds_obs, model

    def _check_obs(self, obs):
        """Check observerations.

        Arguments:
            obs: A psiz.trials.Observations object.

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
            loss = observation_loss
        self.loss = loss

    def fit(
            self, obs_train, batch_size=None, obs_val=None, epochs=1000,
            initial_epoch=0, callbacks=None, seed=None, verbose=0):
        """Fit the free parameters of the embedding model.

        Arguments:
            obs_train: An Observations object representing the observed
                data used to train the model.
            batch_size: The batch size to use for the training step.
            obs_val (optional): An Observations object representing the
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

        self._check_obs(obs_train)
        n_obs_train = obs_train.n_trial
        if obs_val is not None:
            do_validation = True
            self._check_obs(obs_val)
            n_obs_val = obs_val.n_trial
        else:
            n_obs_val = 0

        if batch_size is None:
            batch_size_train = n_obs_train
            batch_size_val = n_obs_val
        else:
            batch_size_train = np.minimum(batch_size, n_obs_train)
            batch_size_val = n_obs_val  # TODO

        # NOTE: The stack operation is used to make sure that a consistent
        # trial configuration list is used across train and validation.
        if do_validation:
            obs = psiz.trials.stack([obs_train, obs_val])
        else:
            obs = obs_train

        # Create dataset and build compatible model by examining the
        # different trial configurations used in `obs`.
        ds_obs, model = self._prepare_and_build(obs)

        # Split dataset back into train and validation.
        ds_obs_train = ds_obs.take(n_obs_train)
        ds_obs_train = ds_obs_train.shuffle(
            buffer_size=n_obs_train, reshuffle_each_iteration=True
        )
        ds_obs_train = ds_obs_train.batch(
            batch_size_train, drop_remainder=False
        )
        if do_validation:
            ds_obs_val = ds_obs.skip(n_obs_train)
            ds_obs_val = ds_obs_val.batch(
                batch_size_val, drop_remainder=False
            )

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
        metric_train_loss = tf.keras.metrics.Mean(name='train_loss')
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
            metric_train_loss(loss_value)

            # NOTE: There are problems using constraints with
            # Eager Execution since gradients are returned as
            # tf.IndexedSlices, which in Eager Execution mode
            # cannot be used to update a variable. To solve this
            # problem, use the pattern below on any IndexedSlices.
            # gradients[0] = tf.convert_to_tensor(gradients[0])
            gradients[1] = tf.convert_to_tensor(gradients[1])

            # Apply gradients (subject to constraints).
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )

        @tf.function
        def validation_step(inputs):
            # Compute validation loss.
            probs = model(inputs)
            # Loss value for this minibatch.
            loss_value = self.loss(probs, inputs['weight'])
            # Add extra losses created during this forward pass.
            loss_value += sum(model.losses)
            metric_val_loss(loss_value)

        @tf.function
        def final_step(inputs, metric):
            probs = model(inputs)
            # Loss value for this minibatch.
            loss_value = self.loss(probs, inputs['weight'])
            # Add extra losses created during this forward pass.
            loss_value += sum(model.losses)
            metric(loss_value)

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
                    train_loss = metric_train_loss.result()
                    logs['train_loss'] = train_loss.numpy()

                    # Compute validation loss.
                    if do_validation:
                        for batch_val in ds_obs_val:
                            validation_step(batch_val)
                        val_loss = metric_val_loss.result()
                        logs['val_loss'] = val_loss.numpy()

                    # TODO conditional loss/metric print out:
                    if verbose > 3:
                        if epoch % self.log_freq == 0:
                            print(
                                '        epoch {0:5d} | loss_train: {1: .6f} '
                                '| loss_val: {2: .6f}'.format(
                                    epoch, train_loss, val_loss
                                )
                            )

                    # Record progress for TensorBoard.
                    if self.do_log:
                        if epoch % self.log_freq == 0:
                            tf.summary.scalar(
                                'train_loss', train_loss, step=epoch
                            )
                            tf.summary.scalar(
                                'val_loss', val_loss, step=epoch
                            )
                            tf_theta = model.get_layer(name='core_layer').theta
                            for param_name in tf_theta:
                                tf.summary.scalar(
                                    param_name, tf_theta[param_name],
                                    step=epoch
                                )

                    callback_list.on_epoch_end(epoch, logs=logs)
                summary_writer.flush()
            epoch_stop_time_s = time.time() - epoch_start_time_s
        callback_list.on_train_end(logs=None)

        # Determine time per epoch.
        ms_per_epoch = 1000 * epoch_stop_time_s / epoch
        time_per_epoch_str = '{0:.0f} ms/epoch'.format(ms_per_epoch)

        # Add final model losses to a `final` dictionary.
        # NOTE: If there is an early stopping callback with
        # restore_best_weights, then the final evaluation will use
        # those weights.
        final = {}
        final['epoch'] = epoch

        metric_train_loss.reset_states()
        metric_val_loss.reset_states()
        for batch_train in ds_obs_train:
            final_step(batch_train, metric_train_loss)
        train_loss = metric_train_loss.result()
        final['train_loss'] = train_loss.numpy()

        if do_validation:
            for batch_val in ds_obs_val:
                validation_step(batch_val)
            val_loss = metric_val_loss.result()
            final['val_loss'] = val_loss.numpy()

        # Add time information.
        total_duration = time.time() - fit_start_time_s
        total_duration_str = '{0:.0f} s'.format(total_duration)
        final['total_duration_s'] = total_duration
        final['ms_per_epoch'] = ms_per_epoch

        # Piggy-back on History object.
        model.history.final = final

        # TODO conditional loss/metric print out:
        if (verbose > 2):
            print(
                '        final {0:5d} | loss_train: {1: .6f} | '
                'loss_val: {2: .6f} | {3} | {4}'.format(
                    epoch, train_loss, val_loss,
                    total_duration_str,  time_per_epoch_str
                )
            )
            print('')

        return model.history

    def evaluate(self, obs, batch_size=None):
        """Evaluate observations using the current state of the model.

        Notes:
            Observations are evaluated in test mode. This means that
            regularization terms are not included in the loss.

        Arguments:
            obs: A Observations object representing the observed data.
            batch_size (optional): Integer indicating the batch size.

        Returns:
            loss: The average loss per observation. Loss is defined as
                the negative loglikelihood.

        """
        self._check_obs(obs)
        ds_obs, model = self._prepare_and_build(obs)

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
            metric_loss(loss_value)

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
        (z_q, z_r) = _inflate_points(
            docket.stimulus_set, docket.max_n_reference, z
        )
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
            probs_config = np.ones((n_trial, n_outcome, n_sample), dtype=np.float64)
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
            obs: A Observations object representing the observed data.
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
        # TODO CRITICAL update for new layer-based API
        f = h5py.File(filepath, "w")
        f.create_dataset("embedding_type", data=type(self).__name__)
        f.create_dataset("n_stimuli", data=self.n_stimuli)
        f.create_dataset("n_dim", data=self.n_dim)
        f.create_dataset("n_group", data=self.n_group)

        # Create group for architecture.
        grp_arch = f.create_group('architecture')

        # Create group for coordinate layer.
        grp_coord = grp_arch.create_group('coordinate_layer')
        grp_coord.create_dataset(
            'class_name', data=type(self.coordinate).__name__
        )
        grp_coord.create_dataset(
            'config', data=str(self.coordinate.get_config())
        )

        # Create group for attention layer.
        grp_attention = grp_arch.create_group('attention_layer')
        grp_attention.create_dataset(
            'class_name', data=type(self.attention).__name__
        )
        grp_attention.create_dataset(
            'config', data=str(self.attention.get_config())
        )

        # Create group for kernel layer.
        grp_kernel = grp_arch.create_group('kernel_layer')
        grp_kernel.create_dataset(
            'class_name', data=type(self.kernel).__name__
        )
        grp_kernel.create_dataset(
            'config', data=str(self.kernel.get_config())
        )

        # Save weights.
        weights = self.get_weights()
        grp_weights = f.create_group("weights")
        for k, d in weights.items():
            if isinstance(d, dict):
                for var_name, var_value in d.items():
                    grp_weights.create_dataset(var_name, data=var_value)
            else:
                grp_weights.create_dataset(k, data=d)

        # TODO clean up.
        # Save coordinate layer information.
        # grp_z = f.create_group("z")
        # grp_z.create_dataset("value", data=self.z)
        # grp_z.create_dataset(
        #     "trainable", data=self.coordinate.z.trainable
        # )

        # Save kernel variables (theta).
        # grp_theta = f.create_group("theta")
        # for theta_name, theta_var in self.kernel.theta.items():
        #     grp_theta_param = grp_theta.create_group(theta_name)
        #     grp_theta_param.create_dataset(
        #         "value",
        #         data=theta_var.numpy()
        #     )
        #     # grp_theta_param.create_dataset(
        #     #     "trainable",
        #     #     data=theta_var.trainable
        #     # )

        # # Save phi variables.
        # grp_phi = f.create_group("phi")
        # grp_phi_param = grp_phi.create_group('w')
        # grp_phi_param.create_dataset(
        #     "value", data=self.attention.w.numpy()
        # )
        # # grp_phi_param.create_dataset(
        # #     "trainable", data=self.attention.w.trainable
        # # )

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
        # TODO CRITICAL update and add other necessary attributes: optimizer,
        # etc.
        # Make shallow copy of whole object.
        cpyobj = type(self)(
            self.n_stimuli, n_dim=self.n_dim, n_group=self.n_group
        )
        # Make deepcopy required attributes
        cpyobj.vars['z'] = copy.deepcopy(self.vars['z'], memodict)
        cpyobj.vars['phi'] = copy.deepcopy(self.vars['phi'], memodict)
        cpyobj.vars['theta'] = copy.deepcopy(self.vars['theta'], memodict)

        return cpyobj


class AnchoredOrdinal(PsychologicalEmbedding):
    """An embedding model that uses anchored, ordinal judgments."""

    def __init__(
            self, n_stimuli, n_dim=2, n_group=1, coordinate=None,
            attention=None, kernel=None):
        """Initialize."""
        PsychologicalEmbedding.__init__(
            self, n_stimuli, n_dim=n_dim, n_group=n_group,
            coordinate=coordinate,
            attention=attention, kernel=kernel
        )


class QueryReference(Layer):
    """Model of query reference similarity judgments."""

    def __init__(
            self, coordinate, attention, kernel,
            config_list):
        """Initialize.

        Arguments:
            tf_theta:
            tf_phi:
            tf_z:
            tf_similarity:
            config_list: It is assumed that the indices that will be
                passed in later as inputs will correspond to the
                indices in this data structure.

        """
        super(QueryReference, self).__init__()

        self.coordinate = coordinate
        self.attention = attention
        self.kernel = kernel

        self.n_config = tf.constant(len(config_list))
        self.config_n_select = tf.constant(config_list.n_select.values)
        self.config_is_ranked = tf.constant(config_list.is_ranked.values)
        self.max_n_reference = tf.constant(
            np.max(config_list.n_reference.values)
        )

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A list of inputs:
                stimulus_set: Containing the integers [0, n_stimuli[
                config_idx: Containing the integers [0, n_config[
                group_id: Containing the integers [0, n_group[

        """
        # Inputs.
        obs_stimulus_set = inputs[0]
        obs_config_idx = inputs[1]
        obs_group_id = inputs[2]
        is_present = inputs[4]

        # Expand attention weights.
        attention = self.attention(obs_group_id)

        # Inflate cooridnates.
        outputs = self.coordinate(
            [obs_stimulus_set, self.max_n_reference]
        )
        z_q = outputs[0]
        z_r = outputs[1]

        # Compute similarity between query and references.
        sim_qr = self.kernel([z_q, z_r, attention])

        # Zero out similarities involving placeholder.
        sim_qr = sim_qr * tf.cast(is_present[:, 1:], dtype=K.floatx())

        # Pre-allocate likelihood tensor.
        n_trial = tf.shape(obs_stimulus_set)[0]
        likelihood = tf.zeros([n_trial], dtype=K.floatx())

        # Compute the probability of observations for different trial
        # configurations.
        for i_config in tf.range(self.n_config):
            n_select = self.config_n_select[i_config]
            is_ranked = self.config_is_ranked[i_config]

            # Identify trials belonging to current trial configuration.
            locs = tf.equal(obs_config_idx, i_config)
            trial_idx = tf.squeeze(tf.where(locs))

            # Grab similarities belonging to current trial configuration.
            sim_qr_config = tf.gather(sim_qr, trial_idx)

            # Compute probability of behavior.
            prob_config = _tf_ranked_sequence_probability(
                sim_qr_config, n_select
            )

            # Update master results.
            likelihood = tf.tensor_scatter_nd_update(
                likelihood, tf.expand_dims(trial_idx, axis=1), prob_config
            )

        return likelihood

    def reset_weights(self):
        """Reset trainable variables."""
        pass


class Coordinate(Layer):
    """Embedding coordinates.

    Handles a placeholder stimulus using stimulus ID -1.

    """

    def __init__(
            self, n_stimuli=None, n_dim=None, fit_z=True, z_min=None,
            z_max=None, **kwargs):
        """Initialize a coordinate layer.

        With no constraints, the coordinates are initialized using a
            using a multivariate Gaussian.

        Arguments:
            n_stimuli:
            n_dim:
            fit_z (optional): Boolean
            z_min (optional):
            z_max (optional):

        """
        super(Coordinate, self).__init__(**kwargs)

        self.n_stimuli = n_stimuli
        self.n_dim = n_dim
        self.z_min = z_min
        self.z_max = z_max

        if z_min is not None and z_max is None:
            z_constraint = GreaterEqualThan(min_value=z_min)
        elif z_min is None and z_max is not None:
            z_constraint = LessEqualThan(max_value=z_max)
        elif z_min is not None and z_max is not None:
            z_constraint = MinMax(min_value, max_value)
        else:
            z_constraint = ProjectZ()

        self.fit_z = fit_z
        self.z = tf.Variable(
            initial_value=self.random_z(), trainable=fit_z,
            name="z", dtype=K.floatx(),
            constraint=z_constraint
        )

    def __call__(self, inputs):
        """Call."""
        stimulus_set = inputs[0] + 1  # Add one for placeholder stimulus.
        max_n_reference = inputs[1]

        z_pad = tf.concat(
            [
                tf.zeros([1, self.z.shape[1]], dtype=K.floatx()),
                self.z
            ], axis=0
        )
        (z_q, z_r) = self._tf_inflate_points(
            stimulus_set, max_n_reference, z_pad
        )
        return [z_q, z_r]

    def _tf_inflate_points(
            self, stimulus_set, n_reference, z):
        """Inflate stimulus set into embedding points.

        Note: This method will not gracefully handle placeholder
        stimulus IDs. The stimulus IDs and coordinates must already
        have handled the placeholder.

        """
        n_trial = tf.shape(stimulus_set)[0]
        n_dim = tf.shape(z)[1]

        # Inflate query stimuli.
        z_q = tf.gather(z, stimulus_set[:, 0])
        z_q = tf.expand_dims(z_q, axis=2)

        # Initialize z_r.
        # z_r = tf.zeros([n_trial, n_dim, n_reference], dtype=K.floatx())
        z_r_2 = tf.zeros([n_reference, n_trial, n_dim], dtype=K.floatx())

        for i_ref in tf.range(n_reference):
            z_r_new = tf.gather(
                z, stimulus_set[:, i_ref + tf.constant(1, dtype=tf.int32)]
            )

            i_ref_expand = tf.expand_dims(i_ref, axis=0)
            i_ref_expand = tf.expand_dims(i_ref_expand, axis=0)
            z_r_new_2 = tf.expand_dims(z_r_new, axis=0)
            z_r_2 = tf.tensor_scatter_nd_update(
                z_r_2, i_ref_expand, z_r_new_2
            )

            # z_r_new = tf.expand_dims(z_r_new, axis=2)
            # pre_pad = tf.zeros([n_trial, n_dim, i_ref], dtype=K.floatx())
            # post_pad = tf.zeros([
            #     n_trial, n_dim,
            #     n_reference - i_ref - tf.constant(1, dtype=tf.int32)
            # ], dtype=K.floatx())
            # z_r_new = tf.concat([pre_pad, z_r_new, post_pad], axis=2)
            # z_r = z_r + z_r_new

        z_r_2 = tf.transpose(z_r_2, perm=[1, 2, 0])
        return (z_q, z_r_2)

    def reset_weights(self):
        """Reset trainable variables."""
        if self.fit_z:
            self.z.assign(self.random_z())

    def random_z(self):
        """Random z."""
        # TODO RandomEmbedding should take z_min and z_max argument.
        z = RandomEmbedding(
            mean=tf.zeros([self.n_dim], dtype=K.floatx()),
            stdev=tf.ones([self.n_dim], dtype=K.floatx()),
            minval=tf.constant(-3., dtype=K.floatx()),
            maxval=tf.constant(0., dtype=K.floatx()),
            dtype=K.floatx()
        )(shape=[self.n_stimuli, self.n_dim])
        return z

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'n_stimuli': self.n_stimuli, 'n_dim': self.n_dim,
            'fit_z': self.fit_z, 'z_min': self.z_min, 'z_max': self.z_max
        })
        return config


class WeightedDistance(Layer):
    """Weighted Minkowski distance."""

    def __init__(self, fit_rho=True, **kwargs):
        """Initialize.

        Arguments:
            fit_rho (optional): Boolean

        """
        super(WeightedDistance, self).__init__(**kwargs)
        self.fit_rho = fit_rho
        self.rho = tf.Variable(
            initial_value=self.random_rho(),
            trainable=self.fit_rho, name="rho", dtype=K.floatx(),
            constraint=GreaterThan(min_value=1.0)
        )

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: List of inputs.

        """
        z_q = inputs[0]  # Query.
        z_r = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        # Weighted Minkowski distance.
        d_qr = tf.pow(tf.abs(z_q - z_r), self.rho)
        d_qr = tf.multiply(d_qr, w)
        d_qr = tf.pow(tf.reduce_sum(d_qr, axis=1), 1. / self.rho)

        return d_qr

    def reset_weights(self):
        """Reset trainable variables."""
        if self.fit_rho:
            self.rho.assign(self.random_rho())

    def random_rho(self):
        """Random rho."""
        return tf.random_uniform_initializer(1.01, 3.)(shape=[])

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({'fit_rho': self.fit_rho})
        return config


class SeparateAttention(Layer):
    """Attention Layer."""

    def __init__(self, n_dim, n_group=1, fit_group=None, **kwargs):
        """Initialize.

        Arguments:
            n_dim: Integer
            n_group: Integer
            fit_group: Boolean Array
                shape=(n_group,)

        """
        super(SeparateAttention, self).__init__(**kwargs)

        self.n_dim = n_dim
        self.n_group = n_group

        if fit_group is None:
            if self.n_group == 1:
                fit_group = [False]
            else:
                fit_group = np.ones(n_group, dtype=bool)
        self.fit_group = fit_group

        w_list = []
        for i_group in range(self.n_group):
            w_i_name = "w_{0}".format(i_group)
            if self.n_group == 1:
                initial_value = np.ones([1, self.n_dim])
            else:
                initial_value = self.random_w()

            w_i = tf.Variable(
                initial_value=initial_value,
                trainable=fit_group[i_group], name=w_i_name, dtype=K.floatx(),
                constraint=ProjectAttention()
            )
            setattr(self, w_i_name, w_i)
            w_list.append(w_i)
        self.w_list = w_list
        self.concat_layer = tf.keras.layers.Concatenate(axis=0)

    def call(self, inputs):
        """Call.

        Inflate weights by `group_id`.

        Arguments:
            inputs: group_id

        """
        w_all = self.concat_layer(self.w_list)
        w_expand = tf.gather(w_all, inputs)
        w_expand = tf.expand_dims(w_expand, axis=2)
        return w_expand

    def reset_weights(self):
        """Reset trainable variables."""
        w_list = []
        for i_group in range(self.n_group):
            w_i_name = "w_{0}".format(i_group)
            w_i = getattr(self, w_i_name)
            if self.fit_group[i_group]:
                w_i.assign(self.random_w())
            w_list.append(w_i)
        self.w_list = w_list

    def random_w(self):
        """Random w."""
        scale = tf.constant(self.n_dim, dtype=K.floatx())
        alpha = tf.constant(np.ones((self.n_dim)), dtype=K.floatx())
        return RandomAttention(
            alpha, scale, dtype=K.floatx()
        )(shape=[1, self.n_dim])


class Attention(Layer):
    """Attention Layer."""

    def __init__(self, n_dim=None, n_group=1, fit_group=None, **kwargs):
        """Initialize.

        Arguments:
            n_dim: Integer
            n_group (optional): Integer
            fit_group: Boolean Array
                shape=(n_group,)

        """
        super(Attention, self).__init__(**kwargs)

        self.n_dim = n_dim
        self.n_group = n_group

        if fit_group is None:
            if self.n_group == 1:
                fit_group = False
            else:
                fit_group = True
        self.fit_group = fit_group

        if self.n_group == 1:
            initial_value = np.ones([1, self.n_dim])
        else:
            initial_value = self.random_w()

        self.w = tf.Variable(
            initial_value=initial_value,
            trainable=fit_group, name='w', dtype=K.floatx(),
            constraint=ProjectAttention()
        )

    def call(self, inputs):
        """Call.

        Inflate weights by `group_id`.

        Arguments:
            inputs: group_id

        """
        w_expand = tf.gather(self.w, inputs)
        w_expand = tf.expand_dims(w_expand, axis=2)
        return w_expand

    def reset_weights(self):
        """Reset trainable variables."""
        if self.fit_group:
            self.w.assign(self.random_w())

    def random_w(self):
        """Random w."""
        scale = tf.constant(self.n_dim, dtype=K.floatx())
        alpha = tf.constant(np.ones((self.n_dim)), dtype=K.floatx())
        return RandomAttention(
            alpha, scale, dtype=K.floatx()
        )(shape=[self.n_group, self.n_dim])

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'n_dim': self.n_dim, 'n_group': self.n_group,
            'fit_group': self.fit_group
        })
        return config


class InverseKernel(Layer):
    """Inverse-distance similarity kernel.

    This embedding technique uses the following similarity kernel:
        s(x,y) = 1 / norm(x - y, rho)**tau,
    where x and y are n-dimensional vectors. The similarity kernel has
    three free parameters: rho, tau, and mu.

    """

    def __init__(self, fit_rho=True, fit_tau=True, fit_mu=True, **kwargs):
        """Initialize.

        Arguments:
            fit_tau (optional): Boolean
            fit_gamme (optional): Boolean
            fit_beta (optional): Boolean

        """
        super(InverseKernel, self).__init__(**kwargs)
        self.distance_layer = WeightedDistance(fit_rho=fit_rho)

        self.fit_tau = fit_tau
        self.tau = tf.Variable(
            initial_value=self.random_tau(),
            trainable=fit_tau, name="tau", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=1.0)
        )

        self.fit_mu = fit_mu
        self.mu = tf.Variable(
            initial_value=self.random_mu(),
            trainable=fit_mu, name="mu", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=2.2204e-16)
        )

        self.theta = {
            'rho': self.distance_layer.rho,
            'tau': self.tau,
            'mu': self.mu
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs:

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        z_q = inputs[0]  # Query.
        z_r = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        d_qr = self.distance_layer([z_q, z_r, w])

        # Exponential family similarity function.
        sim_qr = 1 / (tf.pow(d_qr, self.tau) + self.mu)
        return sim_qr

    def reset_weights(self):
        """Reset trainable variables."""
        self.distance_layer.reset_weights()

        if self.fit_tau:
            self.tau.assign(self.random_tau())

        if self.fit_mu:
            self.mu.assign(self.random_mu())

    def random_tau(self):
        """Random tau."""
        return tf.random_uniform_initializer(1., 2.)(shape=[])

    def random_mu(self):
        """Random mu."""
        return tf.random_uniform_initializer(0.0000000001, .001)(shape=[])


class ExponentialKernel(Layer):
    """Exponential family similarity kernel.

    This embedding technique uses the following similarity kernel:
        s(x,y) = exp(-beta .* norm(x - y, rho).^tau) + gamma,
    where x and y are n-dimensional vectors. The similarity kernel has
    four free parameters: rho, tau, gamma, and beta. The exponential
    family is obtained by integrating across various psychological
    theories [1,2,3,4].

    References:
        [1] Jones, M., Love, B. C., & Maddox, W. T. (2006). Recency
            effects as a window to generalization: Separating
            decisional and perceptual sequential effects in category
            learning. Journal of Experimental Psychology: Learning,
            Memory, & Cognition, 32 , 316-332.
        [2] Jones, M., Maddox, W. T., & Love, B. C. (2006). The role of
            similarity in generalization. In Proceedings of the 28th
            annual meeting of the cognitive science society (pp. 405-
            410).
        [3] Nosofsky, R. M. (1986). Attention, similarity, and the
            identification-categorization relationship. Journal of
            Experimental Psychology: General, 115, 39-57.
        [4] Shepard, R. N. (1987). Toward a universal law of
            generalization for psychological science. Science, 237,
            1317-1323.

    """

    def __init__(
            self, fit_rho=True, fit_tau=True, fit_gamma=True, fit_beta=True,
            **kwargs):
        """Initialize.

        Arguments:
            fit_rho (optional): Boolean
            fit_tau (optional): Boolean
            fit_gamme (optional): Boolean
            fit_beta (optional): Boolean

        """
        super(ExponentialKernel, self).__init__(**kwargs)
        self.distance_layer = WeightedDistance(fit_rho=fit_rho)

        self.fit_tau = fit_tau
        self.tau = tf.Variable(
            initial_value=self.random_tau(),
            trainable=self.fit_tau, name="tau", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=1.0)
        )

        self.fit_gamma = fit_gamma
        self.gamma = tf.Variable(
            initial_value=self.random_gamma(),
            trainable=self.fit_gamma, name="gamma", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=0.0)
        )

        self.fit_beta = fit_beta
        self.beta = tf.Variable(
            initial_value=self.random_beta(),
            trainable=self.fit_beta, name="beta", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=1.0)
        )

        self.theta = {
            'rho': self.distance_layer.rho,
            'tau': self.tau,
            'gamma': self.gamma,
            'beta': self.beta
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs:
                z_q: A set of embedding points.
                    shape = (n_trial, n_dim [, n_sample])
                z_r: A set of embedding points.
                    shape = (n_trial, n_dim [, n_sample])
                attention: The weights allocated to each dimension
                    in a weighted minkowski metric.
                    shape = (n_trial, n_dim [, n_sample])

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        z_q = inputs[0]  # Query.
        z_r = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        d_qr = self.distance_layer([z_q, z_r, w])

        # Exponential family similarity function.
        sim_qr = tf.exp(
            tf.negative(self.beta) * tf.pow(d_qr, self.tau)
        ) + self.gamma
        return sim_qr

    def reset_weights(self):
        """Reset trainable variables."""
        self.distance_layer.reset_weights()

        if self.fit_tau:
            self.tau.assign(self.random_tau())

        if self.fit_gamma:
            self.gamma.assign(self.random_gamma())

        if self.fit_beta:
            self.beta.assign(self.random_beta())

    def random_tau(self):
        """Random tau."""
        return tf.random_uniform_initializer(1., 2.)(shape=[])

    def random_gamma(self):
        """Random gamma."""
        return tf.random_uniform_initializer(0., .001)(shape=[])

    def random_beta(self):
        """Random beta."""
        return tf.random_uniform_initializer(1., 30.)(shape=[])

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_rho': self.distance_layer.fit_rho, 'fit_tau': self.fit_tau,
            'fit_gamma': self.fit_gamma, 'fit_beta': self.fit_beta
        })
        return config


class HeavyTailedKernel(Layer):
    """Heavy-tailed family similarity kernel.

    This embedding technique uses the following similarity kernel:
        s(x,y) = (kappa + (norm(x-y, rho).^tau)).^(-alpha),
    where x and y are n-dimensional vectors. The similarity kernel has
    four free parameters: rho, tau, kappa, and alpha. The
    heavy-tailed family is a generalization of the Student-t family.

    """

    def __init__(
            self, fit_rho=True, fit_tau=True, fit_kappa=True, fit_alpha=True,
            **kwargs):
        """Initialize.

        Arguments:
            fit_rho (optional): Boolean
            fit_tau (optional): Boolean
            fit_kappa (optional): Boolean
            fit_alpha (optional): Boolean

        """
        super(HeavyTailedKernel, self).__init__(**kwargs)
        self.distance_layer = WeightedDistance(fit_rho=fit_rho)

        self.fit_tau = fit_tau
        self.tau = tf.Variable(
            initial_value=self.random_tau(),
            trainable=fit_tau, name="tau", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=1.0)
        )

        self.fit_kappa = fit_kappa
        self.kappa = tf.Variable(
            initial_value=self.random_kappa(),
            trainable=fit_kappa, name="kappa", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=0.0)
        )

        self.fit_alpha = fit_alpha
        self.alpha = tf.Variable(
            initial_value=self.random_alpha(),
            trainable=fit_alpha, name="alpha", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=0.0)
        )

        self.theta = {
            'rho': self.distance_layer.rho,
            'tau': self.tau,
            'kappa': self.kappa,
            'alpha': self.alpha
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs:

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        z_q = inputs[0]  # Query.
        z_r = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        d_qr = self.distance_layer([z_q, z_r, w])

        # Heavy-tailed family similarity function.
        sim_qr = tf.pow(
            self.kappa + tf.pow(d_qr, self.tau), (tf.negative(self.alpha))
        )
        return sim_qr

    def reset_weights(self):
        """Reset trainable variables."""
        self.distance_layer.reset_weights()

        if self.fit_tau:
            self.tau.assign(self.random_tau())

        if self.fit_kappa:
            self.kappa.assign(self.random_kappa())

        if self.fit_alpha:
            self.alpha.assign(self.random_alpha())

    def random_tau(self):
        """Random tau."""
        return tf.random_uniform_initializer(1., 2.)(shape=[])

    def random_kappa(self):
        """Random kappa."""
        return tf.random_uniform_initializer(1., 11.)(shape=[])

    def random_alpha(self):
        """Random alpha."""
        return tf.random_uniform_initializer(10., 60.)(shape=[])


class StudentsTKernel(Layer):
    """Student's t-distribution similarity kernel.

    The embedding technique uses the following similarity kernel:
        s(x,y) = (1 + (((norm(x-y, rho)^tau)/alpha))^(-(alpha + 1)/2),
    where x and y are n-dimensional vectors. The similarity kernel has
    three free parameters: rho, tau, and alpha. The original Student-t
    kernel proposed by van der Maaten [1] uses the parameter settings
    rho=2, tau=2, and alpha=n_dim-1. By default, all variables are fit
    to the data. This behavior can be changed by setting the
    appropriate fit_<var_name>=False.

    References:
    [1] van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic
        triplet embedding. In Machine learning for signal processing
        (MLSP), 2012 IEEE international workshop on (p. 1-6).
        doi:10.1109/MLSP.2012.6349720

    """

    def __init__(
            self, n_dim=None, fit_rho=True, fit_tau=True, fit_alpha=True, **kwargs):
        """Initialize.

        Arguments:
            n_dim:  Integer indicating the dimensionality of the embedding.
            fit_rho (optional): Boolean
            fit_tau (optional): Boolean
            fit_alpha (optional): Boolean

        """
        super(StudentsTKernel, self).__init__(**kwargs)
        self.distance_layer = WeightedDistance(fit_rho=fit_rho)
        self.n_dim = n_dim

        self.fit_tau = fit_tau
        self.tau = tf.Variable(
            initial_value=self.random_tau(),
            trainable=fit_tau, name="tau", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=1.0)
        )

        self.fit_alpha = fit_alpha
        self.alpha = tf.Variable(
            initial_value=self.random_alpha(),
            trainable=fit_alpha, name="alpha", dtype=K.floatx(),
            constraint=GreaterEqualThan(min_value=0.000001)
        )

        self.theta = {
            'rho': self.distance_layer.rho,
            'tau': self.tau,
            'alpha': self.alpha
        }

    def call(self, inputs):
        """Call.

        Arguments:
            inputs:

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        z_q = inputs[0]  # Query.
        z_r = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        d_qr = self.distance_layer([z_q, z_r, w])

        # Student-t family similarity kernel.
        sim_qr = tf.pow(
            1 + (tf.pow(d_qr, tau) / alpha), tf.negative(alpha + 1)/2
        )
        return sim_qr

    def reset_weights(self):
        """Reset trainable variables."""
        self.distance_layer.reset_weights()

        if self.fit_tau:
            self.tau.assign(self.random_tau())

        if self.fit_alpha:
            self.alpha.assign(self.random_alpha())

    def random_tau(self):
        """Random tau."""
        return tf.random_uniform_initializer(1., 2.)(shape=[])

    def random_alpha(self):
        """Random alpha."""
        alpha_min = np.max((1, self.n_dim - 2.))
        alpha_max = self.n_dim + 2.
        return tf.random_uniform_initializer(alpha_min, alpha_max)(shape=[])


def _assert_float_dtype(dtype):
    """Validate and return floating point type based on `dtype`.

    `dtype` must be a floating point type.

    Args:
        dtype: The data type to validate.

    Returns:
        Validated type.

    Raises:
        ValueError: if `dtype` is not a floating point type.

    """
    if not dtype.is_floating:
        raise ValueError("Expected floating point type, got %s." % dtype)
    return dtype


class RandomEmbedding(Initializer):
    """Initializer that generates tensors with a normal distribution.

    Arguments:
        mean: A python scalar or a scalar tensor. Mean of the random
            values to generate.
        minval: Minimum value of a uniform random sampler for each
            dimension.
        maxval: Maximum value of a uniform random sampler for each
            dimension.
        seed: A Python integer. Used to create random seeds. See
        `tf.set_random_seed` for behavior.
        dtype: The data type. Only floating point types are supported.

    """

    def __init__(
            self, mean=0.0, stdev=1.0, minval=0.0, maxval=0.0, seed=None,
            dtype=K.floatx()):
        """Initialize."""
        self.mean = mean
        self.stdev = stdev
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        """Call."""
        if dtype is None:
            dtype = self.dtype
        scale = tf.pow(
            tf.constant(10., dtype=dtype),
            tf.random.uniform(
                [1],
                minval=self.minval,
                maxval=self.maxval,
                dtype=dtype,
                seed=self.seed,
                name=None
            )
        )
        stdev = scale * self.stdev
        return tf.random.normal(
            shape, self.mean, stdev, dtype, seed=self.seed)

    def get_config(self):
        """Return configuration."""
        return {
            "mean": self.mean,
            "stdev": self.stdev,
            "min": self.minval,
            "max": self.maxval,
            "seed": self.seed,
            "dtype": self.dtype.name
        }


class RandomAttention(Initializer):
    """Initializer that generates tensors for attention weights.

    Arguments:
        concentration: An array indicating the concentration
            parameters (i.e., alpha values) governing a Dirichlet
            distribution.
        scale: Scalar indicating how the Dirichlet sample should be scaled.
        dtype: The data type. Only floating point types are supported.

    """

    def __init__(self, concentration, scale=1.0, dtype=K.floatx()):
        """Initialize."""
        self.concentration = concentration
        self.scale = scale
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        """Call."""
        if dtype is None:
            dtype = self.dtype
        dist = tfp.distributions.Dirichlet(self.concentration)
        return self.scale * dist.sample([shape[0]])

    def get_config(self):
        """Return configuration."""
        return {
            "concentration": self.concentration,
            "dtype": self.dtype.name
        }


class GreaterThan(Constraint):
    """Constrains the weights to be greater than a value."""

    def __init__(self, min_value=0.):
        """Initialize."""
        self.min_value = min_value

    def __call__(self, w):
        """Call."""
        w = w - self.min_value
        w = w * tf.cast(tf.math.greater(w, 0.), K.floatx())
        w = w + self.min_value
        return w


class LessThan(Constraint):
    """Constrains the weights to be less than a value."""

    def __init__(self, max_value=0.):
        """Initialize."""
        self.max_value = max_value

    def __call__(self, w):
        """Call."""
        w = w - self.max_value
        w = w * tf.cast(tf.math.greater(0., w), K.floatx())
        w = w + self.max_value
        return w


class GreaterEqualThan(Constraint):
    """Constrains the weights to be greater/equal than a value."""

    def __init__(self, min_value=0.):
        """Initialize."""
        self.min_value = min_value

    def __call__(self, w):
        """Call."""
        w = w - self.min_value
        w = w * tf.cast(tf.math.greater_equal(w, 0.), K.floatx())
        w = w + self.min_value
        return w


class LessEqualThan(Constraint):
    """Constrains the weights to be greater/equal than a value."""

    def __init__(self, max_value=0.):
        """Initialize."""
        self.max_value = max_value

    def __call__(self, w):
        """Call."""
        w = w - self.max_value
        w = w * tf.cast(tf.math.greater_equal(0., w), K.floatx())
        w = w + self.max_value
        return w


class MinMax(Constraint):
    """Constrains the weights to be between/equal values."""

    def __init__(self, min_value, max_value):
        """Initialize."""
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        """Call."""
        w = w - self.min_value
        w = w * tf.cast(tf.math.greater_equal(w, 0.), K.floatx())
        w = w + self.min_value

        w = w - self.max_value
        w = w * tf.cast(tf.math.greater_equal(0., w), K.floatx())
        w = w + self.max_value

        return w


class ProjectZ(Constraint):
    """Constrains the embedding to be zero-centered.

    Constraint is used to improve numerical stability.
    """

    def __init__(self):
        """Initialize."""

    def __call__(self, tf_z):
        """Call."""
        tf_mean = tf.reduce_mean(tf_z, axis=0, keepdims=True)
        tf_z_centered = tf_z - tf_mean
        return tf_z_centered


class ProjectAttention(Constraint):
    """Return projection of attention weights."""

    def __init__(self):
        """Initialize."""

    def __call__(self, tf_attention_0):
        """Call."""
        n_dim = tf.shape(tf_attention_0, out_type=K.floatx())[1]
        tf_attention_1 = tf.divide(
            tf.reduce_sum(tf_attention_0, axis=1, keepdims=True), n_dim
        )
        tf_attention_proj = tf.divide(
            tf_attention_0, tf_attention_1
        )
        return tf_attention_proj


def load_embedding(filepath, custom_objects={}):
    """Load embedding model saved via the save method.

    The loaded data is instantiated as a concrete class of
    SimilarityTrials.

    Arguments:
        filepath: The location of the hdf5 file to load.
        custom_objects (optional): A dictionary mapping the string
            class name to the Python class

    Returns:
        Loaded embedding model.

    Raises:
        ValueError

    """
    # TODO CRITICAL Update load for new API.
    f = h5py.File(filepath, 'r')
    # Common attributes.
    embedding_type = f['embedding_type'][()]
    n_stimuli = f['n_stimuli'][()]
    n_dim = f['n_dim'][()]
    n_group = f['n_group'][()]

    if embedding_type == 'AnchoredOrdinal':
        # Instantiate coordinate layer.
        coordinate_class = f['architecture']['coordinate_layer']['class_name'][()]
        coordinate_config = ast.literal_eval(
            f['architecture']['coordinate_layer']['config'][()]
        )
        if coordinate_class in custom_objects:
            coordinate_class_ = custom_objects[coordinate_class]
        else:
            coordinate_class_ = getattr(psiz.models, coordinate_class)
        coordinate_layer = coordinate_class_.from_config(coordinate_config)

        # Instantiate attention layer.
        attention_class = f['architecture']['attention_layer']['class_name'][()]
        attention_config = ast.literal_eval(
            f['architecture']['attention_layer']['config'][()]
        )
        if attention_class in custom_objects:
            attention_class_ = custom_objects[attention_class]
        else:
            attention_class_ = getattr(psiz.models, attention_class)
        attention_layer = attention_class_.from_config(attention_config)

        # Instantiate kernel layer.
        kernel_class = f['architecture']['kernel_layer']['class_name'][()]
        kernel_config = ast.literal_eval(
            f['architecture']['kernel_layer']['config'][()]
        )
        if kernel_class in custom_objects:
            kernel_class_ = custom_objects[kernel_class]
        else:
            kernel_class_ = getattr(psiz.models, kernel_class)
        kernel_layer = kernel_class_.from_config(kernel_config)

        emb = AnchoredOrdinal(
            n_stimuli, n_dim=n_dim, n_group=n_group,
            coordinate=coordinate_layer, attention=attention_layer,
            kernel=kernel_layer
        )

        # Set weights.
        grp_weights = f['weights']
        for var_name in grp_weights:
            setattr(emb, var_name, grp_weights[var_name][()])

    else:
        # Handle old models. TODO
        # Create coordinate layer.
        z = f['z']['value'][()]
        fit_z = f['z']['trainable'][()]
        coordinate_layer = Coordinate(
            n_stimuli=n_stimuli, n_dim=n_dim, fit_z=fit_z
        )

        # Create attention layer.
        if 'phi_1' in f['phi']:
            fit_group = f['phi']['phi_1']['trainable']
            w = f['phi']['phi_1']['value']
        else:
            fit_group = f['phi']['w']['trainable']
            w = f['phi']['w']['value']
        attention_layer = Attention(
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
            kernel_layer = ExponentialKernel(**theta_config)
        elif embedding_type == 'HeavyTailed':
            kernel_layer = HeavyTailedKernel(**theta_config)
        elif embedding_type == 'StudentsT':
            kernel_layer = StudentsTKernel(**theta_config)
        elif embedding_type == 'Inverse':
            kernel_layer = InverseKernel(**theta_config)
        else:
            raise ValueError(
                'No class found matching the provided `embedding_type`.'
            )

        emb = AnchoredOrdinal(
            n_stimuli, n_dim=n_dim, n_group=n_group,
            coordinate=coordinate_layer, attention=attention_layer,
            kernel=kernel_layer
        )

        # Set weights. TODO
        emb.z = z
        emb.w = w
        emb.theta = theta_value

    f.close()
    return emb


load = load_embedding


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


def _inflate_points(stimulus_set, n_reference, z):
    """Inflate stimulus set into embedding points.

    Arguments:
        stimulus_set: Array of integers indicating the stimuli used
            in each trial.
            shape = (n_trial, >= (n_reference + 1))
        n_reference: A scalar indicating the number of references
            in each trial.
        z: shape = (n_stimuli, n_dim, n_sample)

    Returns:
        z_q: shape = (n_trial, n_dim, 1, n_sample)
        z_r: shape = (n_trial, n_dim, n_reference, n_sample)

    """
    n_trial = stimulus_set.shape[0]
    n_dim = z.shape[1]
    n_sample = z.shape[2]

    # Increment stimuli indices and add placeholder stimulus.
    stimulus_set_temp = (stimulus_set + 1).ravel()
    z_placeholder = np.zeros((1, n_dim, n_sample))
    z_temp = np.concatenate((z_placeholder, z), axis=0)

    # Inflate points.
    z_qr = z_temp[stimulus_set_temp, :, :]
    z_qr = np.transpose(
        np.reshape(z_qr, (n_trial, n_reference + 1, n_dim, n_sample)),
        (0, 2, 1, 3)
    )

    z_q = z_qr[:, :, 0, :]
    z_q = np.expand_dims(z_q, axis=2)
    z_r = z_qr[:, :, 1:, :]
    return (z_q, z_r)


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
        For example, given query Q, the probability of selecting
        the references A, B, and C (in that order) would be:

        P(A,B,C) = s_QA/(s_QA + s_QB + s_QC) * s_QB/(s_QB + s_QC)

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


def _tf_ranked_sequence_probability(sim_qr, n_select):
    """Return probability of a ranked selection sequence.

    See: _ranked_sequence_probability

    Arguments:
        sim_qr: A tensor containing the precomputed similarities
            between the query stimuli and corresponding reference
            stimuli.
            shape = (n_trial, n_reference)
        n_select: A scalar indicating the number of selections
            made for each trial.

    """
    # Initialize.
    n_trial = tf.shape(sim_qr)[0]
    seq_prob = tf.ones([n_trial], dtype=K.floatx())
    selected_idx = n_select - 1
    denom = tf.reduce_sum(sim_qr[:, selected_idx:], axis=1)

    for i_selected in tf.range(selected_idx, -1, -1):
        # Compute selection probability.
        prob = tf.divide(sim_qr[:, i_selected], denom)
        # Update sequence probability.
        seq_prob = tf.multiply(seq_prob, prob)
        # Update denominator in preparation for computing the probability
        # of the previous selection in the sequence.
        if i_selected > tf.constant(0, dtype=tf.int32):
            denom = tf.add(denom, sim_qr[:, i_selected - 1])
        seq_prob.set_shape([None])
    return seq_prob


@tf.function(experimental_relax_shapes=True)
def observation_loss(y_pred, sample_weight):
    """Compute model loss given observation probabilities."""
    n_trial = tf.shape(y_pred)[0]
    n_trial = tf.cast(n_trial, dtype=K.floatx())

    # Convert to (weighted) log probabilities.
    cap = tf.constant(2.2204e-16, dtype=K.floatx())
    y_pred = tf.math.log(tf.maximum(y_pred, cap))
    y_pred = tf.multiply(sample_weight, y_pred)

    # Divide by number of trials to make train and test loss
    # comparable.
    loss = tf.negative(tf.reduce_sum(y_pred))
    loss = tf.divide(loss, n_trial)

    return loss
