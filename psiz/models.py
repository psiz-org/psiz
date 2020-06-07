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
    Proxy: Proxy class for embedding model.
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
    * Implement Rate class.
    * Implement Sort class.

"""

import copy
import datetime
import json
import os
from pathlib import Path
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
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

import psiz.keras.layers
import psiz.keras.metrics
import psiz.trials
import psiz.utils


class Proxy(object):
    """Convenient proxy class for a psychological embedding model.

    The embedding procedure jointly infers three components. First, the
    embedding algorithm infers a stimulus representation denoted by the
    variable z. Second, the embedding algorithm infers the variables
    governing the similarity kernel, denoted theta. Third, the
    embedding algorithm infers a set of attention weights if there is
    more than one group.

    Methods: TODO
        compile: Assign a optimizer, loss and regularization function
            for the optimization procedure.
        fit: Fit the embedding model using the provided observations.
        evaluate: Evaluate the embedding model using the provided
            observations.
        similarity: Return the similarity between provided points.
        distance: Return the (weighted) minkowski distance between
            provided points.
        set_log: Adjust the TensorBoard logging behavior.
        save: Save the embedding model as an hdf5 file.

    Attributes: TODO
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
        log_freq: The number of epochs to wait between log entries.

    """

    def __init__(self, model):
        """Initialize.

        Arguments:
            model: A TensorFlow model.

        """
        super().__init__()
        self.model = model

        # Unsaved attributes.
        self.log_freq = 10

    @property
    def n_stimuli(self):
        """Getter method for n_stimuli."""
        return self.model.n_stimuli

    @property
    def n_dim(self):
        """Getter method for n_dim."""
        return self.model.n_dim

    @property
    def n_group(self):
        """Getter method for n_group."""
        return self.model.n_group

    @property
    def z(self):
        """Getter method for `z`."""
        return self.model.embedding.embeddings.numpy()[1:]

    @property
    def w(self):
        """Getter method for `w`."""
        if hasattr(self.model.kernel, 'attention'):
            w = self.model.kernel.attention.w.numpy()
        else:
            w = np.ones([1, self.n_dim])
        return w

    @property
    def phi(self):
        """Getter method for `phi`."""
        d = {
            'w': self.w
        }
        return d

    @property
    def theta(self):
        """Getter method for `theta`."""
        d = {}
        for k, v in self.model.theta.items():
            d[k] = v.numpy()
        return d

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

    def _broadcast_ready_z(self, z):
        """Create necessary singleton dimensions for `z`."""
        if z.ndim == 2:
            z = np.expand_dims(z, axis=2)
            z = np.expand_dims(z, axis=3)
        elif z.ndim == 3:
            z = np.expand_dims(z, axis=3)
        return z

    def similarity(self, z_q, z_r, group_id=None):
        """Return similarity between two lists of points.

        Similarity is determined using the similarity kernel and the
        current similarity parameters. This method implements the
        logic for handling arguments of different shapes.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim, [1, n_sample])  TODO
            z_r: A set of embedding points.
                shape = (n_trial, n_dim, [n_reference, n_sample])  TODO
            group_id (optional): The group ID for each sample. Can be a
                scalar or an array of shape = (n_trial,).

        Returns:
            The corresponding similarity between rows of embedding
                points.

        """
        # Prepare inputs to exploit broadcasting.
        n_trial = z_q.shape[0]
        if group_id is None:
            group_id = np.zeros((n_trial), dtype=np.int32)
        else:
            if np.isscalar(group_id):
                group_id = group_id * np.ones((n_trial), dtype=np.int32)
            else:
                group_id = group_id.astype(dtype=np.int32)
        group_id = np.expand_dims(group_id, axis=1)
        z_q = self._broadcast_ready_z(z_q)
        z_r = self._broadcast_ready_z(z_r)

        # Pass through kernel function.
        sim_qr = self.model.kernel([
            tf.constant(z_q, dtype=K.floatx()),
            tf.constant(z_r, dtype=K.floatx()),
            tf.constant(group_id, dtype=tf.int32)
        ]).numpy()
        return np.squeeze(sim_qr)

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

        attention = self.w[group_id, :]  # TODO brittle assumption

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

        # TODO brittle assumption
        d_qr = self.model.kernel.distance([
            tf.constant(z_q, dtype=K.floatx()),
            tf.constant(z_r, dtype=K.floatx()),
            tf.constant(attention, dtype=K.floatx())
        ]).numpy()
        return d_qr

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

    def compile(self, **kwargs):
        """Configure the model for training.

        Compile defines the loss function, the optimizer and the
        metrics.

        Arguments:
            **kwargs: Key-word arguments passed to the TensorFlow
            model's `compile` method.

        """
        self.model.compile(**kwargs)

    def fit(
            self, obs_train, batch_size=None, validation_data=None,
            n_restart=3, n_record=1, do_init=False, monitor='loss',
            compile_kwargs={}, **kwargs):
        """Fit the free parameters of the embedding model.

        This convenience function formats the observations as
        appropriate Dataset(s) and sets up a Restarter object to track
        restarts.

        Arguments:
            obs_train: An RankObservations object representing the
                observed data used to train the model.
            batch_size (optional): The batch size to use for the
                training step.
            validation_data (optional): An RankObservations object
                representing the observed data used to validate the
                model.
            n_restart (optional): The number of independent restarts to
                perform.
            n_record (optional): The number of top-performing restarts
                to record.
            monitor (optional): The value to monitor and select
                restarts.
            kwargs (optional): Additional key-word arguments to be
                passed to the model's `fit` method.

        Returns:
            restart_record: A psiz.restart.FitTracker object.

        """
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
        if validation_data is not None:
            self._check_obs(validation_data)
            ds_obs_val = validation_data.as_dataset()
            n_obs_val = validation_data.n_trial
            # Format as TensorFlow dataset.
            ds_obs_val = ds_obs_val.batch(
                n_obs_val, drop_remainder=False
            )
        else:
            ds_obs_val = None

        # Handle restarts.
        restarter = psiz.restart.Restarter(
            self.model, compile_kwargs=compile_kwargs, monitor=monitor,
            n_restart=n_restart, n_record=n_record, do_init=do_init
        )
        restart_record = restarter.fit(
            x=ds_obs_train, validation_data=ds_obs_val, **kwargs
        )
        self.model = restarter.model

        return restart_record

    def evaluate(self, obs, batch_size=None, **kwargs):
        """Evaluate observations using the current state of the model.

        This convenience function formats the observations as
        an appropriate Dataset. Observations are evaluated in "test"
        mode. This means that regularization terms are not included in
        the loss.

        Arguments:
            obs: A RankObservations object representing the observed
                data.
            batch_size (optional): Integer indicating the batch size.
            kwargs (optional): Additional key-word arguments for
                evaluate.

        Returns:
            loss: The average loss per observation. Loss is defined as
                the negative loglikelihood.

        """
        self._check_obs(obs)
        ds_obs = obs.as_dataset()

        if batch_size is None:
            batch_size = obs.n_trial

        ds_obs = ds_obs.batch(batch_size, drop_remainder=False)
        metrics = self.model.evaluate(x=ds_obs, **kwargs)
        # TODO First call to evaluate isn't correct. Work-around is to
        # call it twice.
        metrics_2 = self.model.evaluate(x=ds_obs, **kwargs)
        return metrics_2

    # def predict(self, docket, batch_size=None, **kwargs):
    #     """Predict outcomes given current state of model.

    #     Arguments:
    #         docket: A docket of trials.
    #         batch_size (optional): Integer indicating batch size.

    #     Returns:
    #         Numpy array(s) of predictions.

    #     """
    #     ds_docket = docket.as_dataset()

    #     if batch_size is None:
    #         batch_size = docket.n_trial

    #     ds_docket = ds_docket.batch(batch_size, drop_remainder=False)
    #     predictions = self.model.predict(x=ds_docket, **kwargs)
    #     return predictions.numpy()

    def save(self, filepath, overwrite=False):
        """Save the PsychologialEmbedding model as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the model.

        """
        # NOTE: Ideally we would use TensorFlow's save method using the
        # snippet below.
        # self.model.save(filepath, overwrite=overwrite, save_format='tf')

        # Make directory.
        Path(filepath).mkdir(parents=True, exist_ok=overwrite)

        fp_config = os.path.join(filepath, 'config.h5')
        fp_weights = os.path.join(filepath, 'weights')

        # Save configuration.
        json_model_config = json.dumps(self.model.get_config())
        json_loss_config = json.dumps(
            tf.keras.losses.serialize(self.model.loss)
        )
        optimizer_config = tf.keras.optimizers.serialize(self.model.optimizer)
        # HACK: numpy.float32 is not serializable
        for k, v in optimizer_config['config'].items():
            if isinstance(v, np.float32):
                optimizer_config['config'][k] = float(v)
        json_optimizer_config = json.dumps(optimizer_config)
        f = h5py.File(fp_config, "w")
        f.create_dataset('model_type', data='psiz')
        f.create_dataset('psiz_version', data='0.4.0')
        f.create_dataset('config', data=json_model_config)
        f.create_dataset('loss', data=json_loss_config)
        f.create_dataset('optimizer', data=json_optimizer_config)
        f.close()

        # Save weights.
        self.model.save_weights(
            fp_weights, overwrite=overwrite, save_format='tf'
        )

    # def subset(self, idx):  TODO DELETE
    #     """Return subset of embedding."""
    #     emb = self.clone()
    #     raise(NotImplementedError)
    #     # TODO CRITICAL, must handle changes to embedding layer
    #     emb.z = emb.z[idx, :] # must use assign
    #     emb.n_stimuli = emb.z.shape[0]
    #     return emb

    def clone(self, custom_objects={}):
        """Clone model."""
        # TODO Test
        # Create topology.
        with tf.keras.utils.custom_object_scope(custom_objects):
            new_model = self.model.from_config(self.model.get_config())

        # Save weights.
        fp_weights = '/tmp/psiz/clone'
        self.model.save_weights(fp_weights, overwrite=True)

        # Set weights of new model.
        new_model.load_weights(fp_weights)

        # Wrap in proxy Class.
        proxy_model = Proxy(model=new_model)

        # Compile to model. TODO is this too brittle?
        # if self.model.loss is not None:
        #     proxy_model.compile(
        #         loss=self.model.loss, optimizer=self.model.optimizer
        #     )

        # Other attributes.
        proxy_model.log_freq = self.log_freq

        return proxy_model


class Rank(tf.keras.Model):
    """Model based on ranked similarity judgments.

    Attributes:
        n_stimuli: The number of stimuli.
        n_dim: The dimensionality of the embedding.
        n_group: The number of groups.

    """

    def __init__(
            self, embedding=None, kernel=None, behavior=None,
            n_sample_test=1, **kwargs):
        """Initialize.

        Arguments:
            embedding: An embedding layer. Must agree with
                n_stimuli, n_dim, n_group.
            attention (optional): An attention layer. Must agree with
                n_stimuli, n_dim, n_group.
            distance (optional): A distance kernel function layer.
            similarity (optional): A similarity function layer.
            n_sample_test (optional): The number of samples from
                posterior to use during evaluation.
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super().__init__(**kwargs)
        self.embedding = embedding
        self.kernel = kernel

        # Initialize behavioral component.
        if behavior is None:
            behavior = psiz.keras.layers.RankBehavior()
        self.behavior = behavior

        self.n_sample_test = n_sample_test

        # Create convenience pointer to kernel parameters.
        self.theta = self.kernel.theta

    @tf.function(input_signature=[{
        'membership': tf.TensorSpec(
            shape=[None, 2], dtype=tf.int32, name='membership'
        ),
        'is_select': tf.TensorSpec(
            shape=[None, None, None], dtype=tf.bool, name='is_select'
        ),
        'stimulus_set': tf.TensorSpec(
            shape=[None, None, None], dtype=tf.int32, name='stimulus_set'
        )
    }])
    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A dictionary of inputs:
                stimulus_set: dtype=tf.int32, consisting of the
                    integers on the interval [0, n_stimuli[
                    shape=(batch_size, n_max_reference + 1)
                is_select: dtype=tf.bool, the shape implies the
                    maximum number of selected stimuli in the data
                    shape=(batch_size, n_max_select)
                membership: dtype=tf.int32, Integers indicating the
                    group and agent membership of a trial.
                    shape=(batch_size, 2)

        """
        # Grab inputs.
        stimulus_set = inputs['stimulus_set']
        is_select = inputs['is_select'][:, 1:, :]
        membership = inputs['membership']

        # Inflate coordinates.
        # TODO can we always assume this split pattern?
        z_stimulus_set = self.embedding(stimulus_set)
        # TensorShape([batch_size, n_ref + 1, n_outcome, n_dim])
        z_stimulus_set = tf.transpose(z_stimulus_set, perm=[0, 3, 1, 2])
        # TensorShape([batch_size, n_dim, n_ref + 1, n_outcome])
        max_n_reference = tf.shape(z_stimulus_set)[2] - 1
        z_q, z_r = tf.split(z_stimulus_set, [1, max_n_reference], 2)

        # Pass through similarity kernel.
        sim_qr = self.kernel([z_q, z_r, membership])

        # Zero out similarities involving placeholder IDs.
        is_present = tf.math.not_equal(stimulus_set, 0)
        is_present = tf.cast(is_present[:, 1:, :], dtype=K.floatx())
        sim_qr = sim_qr * is_present

        # Compute probability of different behavioral outcomes.
        is_select = tf.cast(is_select, dtype=K.floatx())
        is_outcome = tf.cast(is_present[:, 0, :], dtype=K.floatx())
        probs = self.behavior([sim_qr, is_select, is_outcome])
        return probs

    def train_step(self, data):
        """Logic for one training step.

        Arguments:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically,
            the values of the `Model`'s metrics are returned. Example:
            `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # NOTE: During computation of gradients, IndexedSlices are
        # created which generates a TensorFlow warning. I cannot
        # find an implementation that avoids IndexedSlices. The
        # following catch environment silences the offending
        # warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, module=r'.*indexed_slices'
            )
            with backprop.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(
                    y, y_pred, sample_weight, regularization_losses=self.losses
                )
            # Custom training steps:
            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            # NOTE: There is an open issue for using constraints with
            # embedding-like layers (e.g., tf.keras.layers.Embedding,
            # psiz.keras.layers.GroupAttention), see
            # https://github.com/tensorflow/tensorflow/issues/33755.
            # There are also issues when using Eager Execution. A
            # work-around is to convert the problematic gradients, which
            # are returned as tf.IndexedSlices, into dense tensors.
            for idx, grad in enumerate(gradients):
                if gradients[idx].__class__.__name__ == 'IndexedSlices':
                    gradients[idx] = tf.convert_to_tensor(
                        gradients[idx]
                    )

        self.optimizer.apply_gradients(zip(
            gradients, trainable_variables)
        )

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """The logic for one evaluation step.

        Arguments:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`.
            Typically, the values of the `Model`'s metrics are
            returned.

        """
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # NOTE: The standard prediction is performmed with one sample. To
        # accommodate variational inference, the log prob of the data is
        # computed by averaging samples from the model:
        # p(heldout | train) = int_model p(heldout|model) p(model|train)
        #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
        # where model_i is a draw from the posterior p(model|train).
        y_pred = tf.stack([
            self(x, training=False)
            for _ in range(self.n_sample_test)
        ], axis=0)
        y_pred = tf.reduce_mean(y_pred, axis=0)

        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses
        )

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    @property
    def n_stimuli(self):
        """Getter method for n_stimuli."""
        return self.embedding.input_dim - 1

    @property
    def n_dim(self):
        """Getter method for n_dim."""
        return self.embedding.output_dim

    @property
    def n_group(self):
        """Getter method for n_group."""
        return self.kernel.n_group

    def get_config(self):
        """Return model configuration."""
        layer_configs = {
            'embedding': tf.keras.utils.serialize_keras_object(
                self.embedding
            ),
            'kernel': tf.keras.utils.serialize_keras_object(
                self.kernel
            ),
            'behavior': tf.keras.utils.serialize_keras_object(self.behavior)
        }

        config = {
            'name': self.name,
            'class_name': self.__class__.__name__,
            'n_sample_test': self.n_sample_test,
            'layers': copy.deepcopy(layer_configs)
        }
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration.

        Arguments:
            config: A hierarchical configuration dictionary.
            custom_objects: A dictionary of custom classes.

        Returns:
            layer: An instantiated and configured TensorFlow model.

        """
        model_config = copy.deepcopy(config)
        model_config.pop('class_name', None)

        # Deserialize layers.
        layer_configs = model_config.pop('layers', None)
        built_layers = {}
        for layer_name, layer_config in layer_configs.items():
            layer = tf.keras.layers.deserialize(layer_config)
            built_layers[layer_name] = layer

        model_config.update(built_layers)
        return cls(**model_config)


# class Rate(tf.keras.Model):


# class Sort(tf.keras.Model):


def load_model(filepath, custom_objects={}, compile=False):
    """Load embedding model saved via the save method.

    Arguments:
        filepath: The location of the hdf5 file to load.
        custom_objects (optional): A dictionary mapping the string
            class name to the Python class
        compile: Boolean indicating if model should be compiled.

    Returns:
        Loaded embedding model.

    Raises:
        ValueError

    """
    # Check if directory.
    if os.path.isdir(filepath):
        # NOTE: Ideally we could call TensorFlow's `load_model` method as used
        # in the snippet below, but our sub-classed model does not play
        # nice.
        # with tf.keras.utils.custom_object_scope(custom_objects):
        #     model = tf.keras.models.load_model(filepath, compile=compile)
        # return Proxy(model=model)
        # Storage format for psiz_version >= 0.4.0
        fp_config = os.path.join(filepath, 'config.h5')
        fp_weights = os.path.join(filepath, 'weights')

        # Load configuration.
        f = h5py.File(fp_config, 'r')
        psiz_version = f['psiz_version'][()]
        config = json.loads(f['config'][()])
        loss_config = json.loads(f['loss'][()])
        optimizer_config = json.loads(f['optimizer'][()])

        model_class_name = config.get('class_name')
        # Load model.
        if model_class_name in custom_objects:
            model_class = custom_objects[model_class_name]
        else:
            model_class = getattr(psiz.models, model_class_name)
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = model_class.from_config(config)
            loss = tf.keras.losses.deserialize(loss_config)
            optimizer = tf.keras.optimizers.deserialize(optimizer_config)

        if compile:
            model.compile(loss=loss, optimizer=optimizer)
        # Build lazy layers in order to set weights.
        model.embedding.build(input_shape=[None, None, None])

        # Load weights.
        model.load_weights(fp_weights).expect_partial()
        emb = Proxy(model=model)
    else:
        # Storage format for psiz_version < 0.4.0.
        f = h5py.File(filepath, 'r')
        # Common attributes.
        embedding_type = f['embedding_type'][()]
        n_stimuli = f['n_stimuli'][()]
        n_dim = f['n_dim'][()]
        n_group = f['n_group'][()]

        # Create embedding layer.
        z = f['z']['value'][()]
        z_trainable = f['z']['trainable'][()]
        embedding = psiz.keras.layers.Embedding(
            input_dim=n_stimuli+1, output_dim=n_dim, trainable=z_trainable,
            mask_zero=True
        )

        # Create attention layer.
        if 'phi_1' in f['phi']:
            fit_group = f['phi']['phi_1']['trainable'][()]
            w = f['phi']['phi_1']['value'][()]
        else:
            fit_group = f['phi']['w']['trainable'][()]
            w = f['phi']['w']['value'][()]
        attention = psiz.keras.layers.GroupAttention(
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

        # Create similarity layer.
        theta_config = {}
        theta_value = {}
        for p_name in f['theta']:
            theta_config['fit_' + p_name] = f['theta'][p_name]['trainable'][()]
            theta_value[p_name] = f['theta'][p_name]['value'][()]
            # for name in f['theta'][p_name]:
            #     embedding._theta[p_name][name] = f['theta'][p_name][name][()]

        dist_config = {}
        dist_config.update({'fit_rho': theta_config.pop('fit_rho', None)})
        distance = psiz.keras.layers.WeightedMinkowski(**dist_config)
        if embedding_type == 'Exponential':
            similarity = psiz.keras.layers.ExponentialSimilarity(
                **theta_config
            )
        elif embedding_type == 'HeavyTailed':
            similarity = psiz.keras.layers.HeavyTailedSimilarity(
                **theta_config
            )
        elif embedding_type == 'StudentsT':
            similarity = psiz.keras.layers.StudentsTSimilarity(**theta_config)
        elif embedding_type == 'Inverse':
            similarity = psiz.keras.layers.InverseSimilarity(**theta_config)
        else:
            raise ValueError(
                'No class found matching the provided `embedding_type`.'
            )

        model = Rank(
            embedding=embedding, attention=attention, distance=distance,
            similarity=similarity
        )
        emb = Proxy(model=model)

        # Set weights.
        z_pad = np.vstack(
            [np.zeros([1, n_dim]), z]
        )
        emb.model.embedding.embeddings.assign(z_pad)
        emb.w = w
        emb.theta = theta_value

        f.close()
    return emb


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
