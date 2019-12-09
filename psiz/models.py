
# -*- coding: utf-8 -*-
# Copyright 2019 The PsiZ Authors. All Rights Reserved.
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

"""

from abc import ABCMeta, abstractmethod
import copy
from random import randint
import sys
import time
import warnings

import h5py
import numpy as np
import numexpr as ne
import numpy.ma as ma
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import mixture
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Layer
import tensorflow.keras.optimizers
from tensorflow.python.keras import backend as K
from tensorflow.keras.constraints import Constraint


from psiz.utils import ProgressBar


class PsychologicalEmbedding(object):
    """Abstract base class for psychological embedding algorithm.

    The embedding procedure jointly infers three components. First, the
    embedding algorithm infers a stimulus representation denoted z.
    Second, the embedding algorithm infers the similarity kernel
    parameters of the concrete class. The set of similarity kernel
    parameters is denoted theta. Third, the embedding algorithm infers
    a set of attention weights if there is more than one group.

    Methods:
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
        fit_duration: The duration (in seconds) of the last called
            fitting procedure.
        posterior_duration: The duration (in seconds) of the last
            called posterior sampling procedure.

    Notes:
        The setter methods as well as the methods fit, trainable, and
            set_log modify the state of the PsychologicalEmbedding
            object.
        The abstract methods _default_theta,
            _get_similarity_parameters_cold, and _tf_similarity must be
            implemented by each concrete class.
        You can use a custom loss by by setting the loss attribute.

    """

    __metaclass__ = ABCMeta

    def __init__(self, n_stimuli, n_dim=2, n_group=1):
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

        Raises:
            ValueError

        """
        if (n_stimuli < 3):
            raise ValueError("There must be at least three stimuli.")
        self.n_stimuli = n_stimuli
        if (n_dim < 1):
            warnings.warn("The provided dimensionality must be an integer \
                greater than 0. Assuming user requested 2 dimensions.")
            n_dim = 2
        self.n_dim = n_dim
        if (n_group < 1):
            warnings.warn("The provided n_group must be an integer greater \
                than 0. Assuming user requested 1 group.")
            n_group = 1
        self.n_group = n_group

        # Initialize model components.
        self._z = self._init_z()
        self.z = self._z["value"]

        self._phi = self._init_phi()
        self.w = self._phi["w"]["value"]

        self._theta = {}

        # Default inference settings.
        self.lr = 0.001
        self.max_n_epoch = 5000
        self.patience_stop = 10  # 5
        self.patience_reduce = 2

        # Default TensorBoard log attributes.
        self.do_log = False
        self.log_dir = '/tmp/psiz/tensorboard_logs/'
        self.log_freq = 10

        # Timer attributes.
        self.fit_duration = 0.0
        self.posterior_duration = 0.0

        # Set loss function.
        self.loss = default_loss

        super().__init__()

    def _init_z(self):
        """Return initialized embedding points.

        Initialize random embedding points using a multivariate
            Gaussian.
        """
        mean = np.zeros((self.n_dim))
        cov = .03 * np.identity(self.n_dim)
        z = {}
        z["value"] = np.random.multivariate_normal(
            mean, cov, (self.n_stimuli)
        )
        z["trainable"] = True
        return z

    @abstractmethod
    def _default_theta(self):
        """Return dictionary of default theta parameters.

        Returns:
            Dictionary of theta parameters.

        """
        pass

    def _init_phi(self):
        """Return initialized phi.

        Initialize group-specific free parameters.
        """
        w = np.ones((self.n_group, self.n_dim), dtype=np.float32)
        if self.n_group is 1:
            is_trainable = np.zeros([1], dtype=bool)
        else:
            is_trainable = np.ones([self.n_group], dtype=bool)
        phi = dict(
            w=dict(value=w, trainable=is_trainable)
        )
        return phi

    @property
    def z(self):
        """Getter method for z."""
        return self._z["value"]

    @z.setter
    def z(self, z):
        """Setter method for z."""
        self._check_z(z)
        self._z["value"] = z

    def _check_z(self, z):
        """Check argument `z`.

        Raises:
            ValueError

        """
        if z.shape[0] != self.n_stimuli:
            raise ValueError(
                "Input 'z' does not have the appropriate shape (number of \
                stimuli)."
            )
        if z.shape[1] != self.n_dim:
            raise ValueError(
                "Input 'z' does not have the appropriate shape \
                (dimensionality)."
            )

    @property
    def w(self):
        """Getter method for phi."""
        return self._phi["w"]["value"]

    @w.setter
    def w(self, w):
        """Setter method for w."""
        self._check_w(w)
        self._phi["w"]["value"] = w

    def _check_w(self, attention):
        """Check argument `w`.

        Raises:
            ValueError

        """
        if attention.shape[0] != self.n_group:
            raise ValueError(
                "Input 'attention' does not have the appropriate shape \
                (number of groups).")
        if attention.shape[1] != self.n_dim:
            raise ValueError(
                "Input 'attention' does not have the appropriate shape \
                (dimensionality).")

    def _set_theta(self, theta):
        """State changing method sets algorithm-specific parameters.

        This method encapsulates the setting of algorithm-specific free
        parameters governing the similarity kernel.

        Arguments:
            theta: A dictionary of algorithm-specific parameter names
                and corresponding values.

        """
        for param_name in theta:
            self._theta[param_name]["value"] = theta[param_name]["value"]

    def _check_theta_param(self, name, val):
        """Check if value is a numerical scalar."""
        if not np.isscalar(val):
            raise ValueError(
                "The parameter `{0}` must be a numerical scalar.".format(name)
            )
        else:
            val = float(val)

        # Check if within bounds.
        bnds = copy.copy(self._theta[name]["bounds"])
        if bnds[0] is None:
            bnds[0] = -np.inf
        if bnds[1] is None:
            bnds[1] = np.inf
        if (val < bnds[0]) or (val > bnds[1]):
            raise ValueError(
                "The parameter `{0}` must be between {1} and {2}.".format(
                    name, bnds[0], bnds[1]
                )
            )
        return val

    def _get_similarity_parameters(self, init_mode):
        """Return a dictionary of TensorFlow variables.

        This method encapsulates the creation of algorithm-specific
        free parameters governing the similarity kernel.

        Arguments:
            init_mode: A string indicating the initialization mode.
                Valid options are 'cold' and 'hot'.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if init_mode is 'hot':
            tf_theta = self._get_similarity_parameters_hot()
        else:
            tf_theta = self._get_similarity_parameters_cold()

            # If a parameter is untrainable, set the parameter value to the
            # value in the class attribute theta.
            for param_name in self._theta:
                if not self._theta[param_name]["trainable"]:
                    tf_theta[param_name] = tf.Variable(
                        initial_value=self._theta[param_name]["value"],
                        trainable=False,
                        dtype=K.floatx(),
                        name=param_name
                    )
        return tf_theta

    def _get_similarity_parameters_hot(self):
        """Return a dictionary.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        for param_name in self._theta:
            if self._theta[param_name]["trainable"]:
                tf_theta[param_name] = tf.Variable(
                    initial_value=self._theta[param_name]["value"],
                    trainable=True,
                    dtype=K.floatx(),
                    name=param_name
                )
            else:
                tf_theta[param_name] = tf.Variable(
                    initial_value=self._theta[param_name]["value"],
                    trainable=False,
                    dtype=K.floatx(),
                    name=param_name
                )
        return tf_theta

    @abstractmethod
    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        pass

    def trainable(self, spec=None):
        """Specify which parameters are trainable.

        During inference, you may want to fix some free parameters or
        allow non-default parameters to be trained. Pass in a
        dictionary specifying how you would like to update the
        trainability of the parameters.

        In addition to a dictionary, you can pass in three different
        strings: `default`, `freeze`, and `thaw`. The `default` option
        restores the defaults, `freeze` makes all parameters
        untrainable, and `thaw` makes all parameters trainable.

        Arguments:
            spec (optional): If no arguments are provided, the current
                settings are returned as a dictionary. Otherwise a
                string (see above) or a dictionary must be passed as
                an argument. The dictionary is organized such that the
                keys refer to the parameter names and the values
                use boolean values to indicate if the parameters are
                trainable.

        """
        if spec is None:
            trainable_spec = {
                'z': self._z["trainable"],
                'w': self._phi["w"]["trainable"]
            }
            trainable_spec_theta = self._theta_trainable()
            trainable_spec = {**trainable_spec, **trainable_spec_theta}
            return trainable_spec
        elif isinstance(spec, str):
            if spec == 'default':
                spec_default = self._trainable_default()
                self._set_trainable(spec_default)
            elif spec == 'freeze':
                self._z["trainable"] = False
                self._phi["w"]["trainable"] = np.zeros(
                    self.n_group, dtype=bool
                )
                for param_name in self._theta:
                    self._theta[param_name]["trainable"] = False
            elif spec == 'thaw':
                self._z["trainable"] = True
                self._phi["w"]["trainable"] = np.ones(self.n_group, dtype=bool)
                for param_name in self._theta:
                    self._theta[param_name]["trainable"] = True
        else:
            # Assume spec is a dictionary.
            self._set_trainable(spec)

    def _set_trainable(self, spec):
        """Set trainable variables using dictionary."""
        for param_name in spec:
            if param_name is 'z':
                self._z["trainable"] = self._check_z_trainable(spec["z"])
            elif param_name is 'w':
                self._phi["w"]["trainable"] = self._check_w_trainable(
                    spec["w"]
                )
            else:
                self._set_theta_parameter_trainable(
                    param_name, spec[param_name]
                )

    def _theta_trainable(self):
        """Return trainable status of theta parameters."""
        trainable_spec = {}
        for param_name in self._theta:
            trainable_spec[param_name] = self._theta[param_name]["trainable"]
        return trainable_spec

    def _check_z_trainable(self, val):
        """Validate the provided trainable settings."""
        if not np.isscalar(val):
            raise ValueError(
                "The parameter `z` requires a boolean value to set it's "
                "`trainable` property."
            )
        return val

    def _check_w_trainable(self, val):
        """Validate the provided trainable settings."""
        if val.shape[0] != self.n_group:
            raise ValueError(
                "The parameter `phi` requires a boolean array that has the "
                "same length as the number of groups in order to set it's "
                "`trainable` property."
            )
        return val

    def _trainable_default(self):
        """Set the free parameters to the default trainable settings."""
        if self.n_group == 1:
            w_trainable = np.zeros([1], dtype=bool)
        else:
            w_trainable = np.ones([self.n_group], dtype=bool)
        trainable_spec = {
            'z': True,
            'w': w_trainable
        }
        theta_default = self._default_theta()
        for param_name in theta_default:
            trainable_spec[param_name] = theta_default[param_name]["trainable"]
        return trainable_spec

    def _set_theta_parameter_trainable(self, param_name, param_value):
        """Handle model specific theta parameters."""
        self._theta[param_name]["trainable"] = param_value

    def _broadcast_for_similarity(
            self, z_q, z_r, group_id=None, theta=None, phi=None):
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
            theta (optional): The parameters governing the similarity
                kernel. If not provided, the theta associated with the
                current object is used.
            phi (optional): The weights allocated to each
                dimension in a weighted minkowski metric. The weights
                should be positive and sum to the dimensionality of the
                weight vector, although this is not enforced.
                shape = (n_trial, n_dim)

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

        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        attention = phi['w']["value"][group_id, :]

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

        return (z_q, z_r, theta, attention)

    def similarity(self, z_q, z_r, group_id=None, theta=None, phi=None):
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
            theta (optional): The parameters governing the similarity
                kernel. If not provided, the theta associated with the
                current object is used.
            phi (optional): The weights allocated to each
                dimension in a weighted minkowski metric. The weights
                should be positive and sum to the dimensionality of the
                weight vector, although this is not enforced.
                shape = (n_trial, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.

        """
        (z_q, z_r, theta, attention) = self._broadcast_for_similarity(
            z_q, z_r, group_id=group_id, theta=theta, phi=phi
        )

        sim = self._similarity(z_q, z_r, theta, attention)
        return sim

    def distance(self, z_q, z_r, group_id=None, theta=None, phi=None):
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
            theta (optional): The parameters governing the similarity
                kernel. If not provided, the theta associated with the
                current object is used.
            phi (optional): The weights allocated to each
                dimension in a weighted minkowski metric. The weights
                should be positive and sum to the dimensionality of the
                weight vector, although this is not enforced.
                shape = (n_trial, n_dim)

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

        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        attention = phi['w']["value"][group_id, :]

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

        d = _mink_distance(z_q, z_r, theta['rho']["value"], attention)
        return d

    @abstractmethod
    def _tf_similarity(self, z_q, z_r, tf_theta, tf_attention):
        """Similarity kernel.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim)
            z_r: A set of embedding points.
                shape = (n_trial, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension in a
                weighted minkowski metric.
                shape = (n_trial, n_dim)
        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        pass

    @staticmethod
    @abstractmethod
    def _similarity(z_q, z_r, theta, attention):
        """Similarity kernel.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim)
            z_r: A set of embedding points.
                shape = (n_trial, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension in a
                weighted minkowski metric.
                shape = (n_trial, n_dim)
        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        pass

    def _get_attention(self, init_mode):
        """Return attention weights of model as TensorFlow variable."""
        attention_list = []
        for group_id in range(self.n_group):
            tf_attention = self._get_group_attention(init_mode, group_id)
            attention_list.append(tf_attention)
        tf_attention = tf.concat(attention_list, axis=0)
        return tf_attention

    def _get_group_attention(self, init_mode, group_id):
        tf_var_name = "attention_{0}".format(group_id)
        if self._phi['w']["trainable"][group_id]:
            if init_mode is 'hot':
                tf_attention = tf.Variable(
                    initial_value=np.expand_dims(
                        self._phi['w']["value"][group_id, :], axis=0
                    ),
                    trainable=True, name=tf_var_name, dtype=K.floatx(),
                    constraint=ProjectAttention(),
                    shape=[1, self.n_dim]
                )
            else:
                scale = tf.constant(self.n_dim, dtype=K.floatx())
                alpha = tf.constant(np.ones((self.n_dim)), dtype=K.floatx())
                tf_attention = tf.Variable(
                    initial_value=RandomAttention(
                        alpha, scale, dtype=K.floatx()
                    )(shape=[1, self.n_dim]),
                    trainable=True, name=tf_var_name, dtype=K.floatx(),
                    constraint=ProjectAttention()
                )
        else:
            tf_attention = tf.Variable(
                initial_value=np.expand_dims(
                    self._phi['w']["value"][group_id, :], axis=0
                ),
                trainable=False, name=tf_var_name, dtype=K.floatx(),
                constraint=ProjectAttention(),
                shape=[1, self.n_dim]
            )
        return tf_attention

    def _get_embedding(self, init_mode):
        """Return embedding of model as TensorFlow variable.

        Arguments:
            init_mode: A string indicating the initialization mode.
                valid options are 'cold' and 'hot'.

        Returns:
            TensorFlow variable representing the embedding points.

        """
        if self._z["trainable"]:
            if init_mode is 'hot':
                z = self._z["value"]
                tf_z = tf.Variable(
                    initial_value=z,
                    trainable=True, name="z", dtype=K.floatx(),
                    constraint=ProjectZ(),
                    shape=[self.n_stimuli, self.n_dim]
                )
            else:
                tf_z = tf.Variable(
                    initial_value=RandomEmbedding(
                        mean=tf.zeros([self.n_dim], dtype=K.floatx()),
                        stdev=tf.ones([self.n_dim], dtype=K.floatx()),
                        minval=tf.constant(-3., dtype=K.floatx()),
                        maxval=tf.constant(0., dtype=K.floatx()),
                        dtype=K.floatx()
                    )(shape=[self.n_stimuli, self.n_dim]), trainable=True,
                    name="z", dtype=K.floatx(),
                    constraint=ProjectZ()
                )
        else:
            tf_z = tf.Variable(
                initial_value=self._z["value"],
                trainable=False, name="z", dtype=K.floatx(),
                constraint=ProjectZ(),
                shape=[self.n_stimuli, self.n_dim]
            )
        return tf_z

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

    def _build_model(self, tf_config, init_mode='cold'):
        """Build TensorFlow model."""
        tf_theta = self._get_similarity_parameters(init_mode)
        tf_attention = self._get_attention(init_mode)
        tf_z = self._get_embedding(init_mode)

        obs_stimulus_set = tf.keras.Input(
            shape=[None], name='inp_stimulus_set', dtype=tf.int32,
        )
        obs_config_idx = tf.keras.Input(
            shape=[], name='inp_config_idx', dtype=tf.int32,
        )
        obs_group_id = tf.keras.Input(
            shape=[], name='inp_group_id', dtype=tf.int32,
        )
        obs_weight = tf.keras.Input(
            shape=[], name='inp_weight', dtype=K.floatx(),
        )

        inputs = [
            obs_stimulus_set,
            obs_config_idx,
            obs_group_id,
            obs_weight
        ]
        c_layer = CoreLayer(
            tf_theta, tf_attention, tf_z, tf_config, self._tf_similarity
        )
        output = c_layer(inputs)
        model = tf.keras.models.Model(
            inputs,
            output
        )
        return model

    def fit(self, obs, n_restart=50, init_mode='cold', verbose=0):
        """Fit the free parameters of the embedding model.

        Arguments:
            obs: A Observations object representing the observed data.
            n_restart (optional): An integer specifying the number of
                restarts to use for the inference procedure. Since the
                embedding procedure can get stuck in local optima,
                multiple restarts help find the global optimum.
            init_mode (optional): A string indicating the
                initialization mode. Valid options are 'cold' and
                'hot'. When fitting using a `cold` initialization, all
                trainable paramters will be randomly initialized on
                their defined support. When fitting using a `hot`
                initialization, trainable parameters will continue from
                their current value.
            verbose (optional): An integer specifying the verbosity of
                printed output. If zero, nothing is printed. Increasing
                integers display an increasing amount of information.

        Returns:
            loss_train_best: The average loss per observation on the
                train set. Loss is defined as the negative
                loglikelihood.
            loss_val_best: The average loss per observation on the
                validation set. Loss is defined as the negative
                loglikelihood.

        """
        start_time_s = time.time()
        #  Infer embedding.
        if (verbose > 0):
            print('[psiz] Inferring embedding...')
        if (verbose > 1):
            print('    Settings:')
            print(
                '    n_stimuli: {0} | n_dim: {1} | n_group: {2}'
                ' | n_obs: {3} | n_restart: {4}'.format(
                    self.n_stimuli, self.n_dim, self.n_group,
                    obs.n_trial, n_restart))
            print('')

        # Grab configuration information. Need too grab here because
        # configuration mapping may change when grabbing train and
        # validation subset.
        (obs_config_list, obs_config_idx) = self._grab_config_info(obs)
        tf_config = self._prepare_config(obs_config_list)

        # Partition observations into train and validation set to
        # control early stopping of embedding algorithm.
        skf = StratifiedKFold(n_splits=10)
        (train_idx, val_idx) = list(
            skf.split(obs.stimulus_set, obs_config_idx)
        )[0]
        obs_train = obs.subset(train_idx)
        config_idx_train = obs_config_idx[train_idx]
        obs_val = obs.subset(val_idx)
        config_idx_val = obs_config_idx[val_idx]

        # Initial evaluation.
        loss_train_best = self.evaluate(obs_train)
        loss_val_best = self.evaluate(obs_val)

        # Prepare observations.
        tf_inputs_train = self._prepare_inputs(obs_train, config_idx_train)
        tf_inputs_val = self._prepare_inputs(obs_val, config_idx_val)

        z_best = self._z["value"]
        attention_best = self._phi['w']["value"]
        theta_best = self._theta
        beat_init = False
        if (verbose > 2):
            print('        Initialization')
            print(
                '        '
                '     --     | loss: {0: .6f} | loss_val: {1: .6f}'.format(
                    loss_train_best, loss_val_best)
            )
            print('')

        if verbose > 0 and verbose < 3:
            progbar = ProgressBar(
                n_restart, prefix='Progress:', suffix='Complete', length=50
            )
            progbar.update(0)

        # Define optimizer.
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)

        # Run multiple restarts of embedding algorithm.
        for i_restart in range(n_restart):
            if (verbose > 2):
                print('        Restart {0}'.format(i_restart))
            if verbose > 0 and verbose < 3:
                progbar.update(i_restart + 1)

            model = self._build_model(tf_config, init_mode=init_mode)
            summary_writer = tf.summary.create_file_writer(
                '{0}/{1}'.format(self.log_dir, i_restart)
            )

            # During computation of gradients, IndexedSlices are created.
            # Despite my best efforts, I cannot prevent this behavior. The
            # following catch environment silences the offending warning.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', category=UserWarning, module=r'.*indexed_slices'
                )
                with summary_writer.as_default():
                    (
                        loss_train, loss_val, epoch, z, attention, tf_theta
                    ) = self._fit_restart(
                        model, optimizer, tf_inputs_train, tf_inputs_val,
                        verbose
                    )

                # Coonvert Tensors to NumPy.
                loss_train = loss_train.numpy()
                loss_val = loss_val.numpy()
                epoch = epoch.numpy()
                z = z.numpy()
                attention = attention.numpy()
                theta = {}
                for param_name in tf_theta:
                    theta[param_name] = {}
                    theta[param_name]["value"] = tf_theta[param_name].numpy()

            if (verbose > 2):
                print(
                    '        final {0:5d} | loss: {1: .6f} | '
                    'loss_val: {2: .6f}'.format(
                        epoch, loss_train, loss_val)
                )
                print('')

            loss_combined = .9 * loss_train + .1 * loss_val
            loss_combined_best = .9 * loss_train_best + .1 * loss_val_best
            if loss_combined < loss_combined_best:
                loss_val_best = loss_val
                loss_train_best = loss_train
                z_best = z
                attention_best = attention
                theta_best = theta
                beat_init = True

        if (verbose > 1):
            if beat_init:
                print(
                    '    Best Restart\n        n_epoch: {0} | '
                    'loss: {1: .6f} | loss_val: {2: .6f}'.format(
                        epoch, loss_train_best, loss_val_best
                    )
                )
            else:
                print('    Did not beat initialization.')

        self._z["value"] = z_best
        self._phi['w']["value"] = attention_best
        self._set_theta(theta_best)
        self.fit_duration = time.time() - start_time_s

        return loss_train_best, loss_val_best

    def _fit_restart(
            self, model, optimizer, tf_inputs_train, tf_inputs_val, verbose):
        """Embed using a TensorFlow implementation."""
        @tf.function
        def train(model, optimizer, tf_inputs_train, tf_inputs_val):
            # Initialize best values.
            epoch_best = tf.constant(0, dtype=tf.int64)
            loss_train_best = tf.constant(np.inf, dtype=K.floatx())
            loss_val_best = tf.constant(np.inf, dtype=K.floatx())
            z_best = tf.constant(self._z["value"], dtype=K.floatx())
            attention_best = tf.constant(
                self._phi['w']["value"], dtype=K.floatx()
            )
            theta_best = {}
            for param_name in self._theta:
                theta_best[param_name] = tf.constant(
                    self._theta[param_name]['value'], dtype=K.floatx()
                )

            last_improvement_stop = tf.constant(0, dtype=tf.int32)
            last_improvement_reduce = tf.constant(0, dtype=tf.int32)
            tf_max_n_epoch = tf.constant(self.max_n_epoch, dtype=tf.int64)
            tf_patience_stop = tf.constant(self.patience_stop, dtype=tf.int32)
            for epoch in tf.range(tf_max_n_epoch):
                # Compute training loss and gradients.
                with tf.GradientTape() as grad_tape:
                    prob_train = model(tf_inputs_train)
                    loss_train = self.loss(
                        prob_train, tf_inputs_train[3],
                        model.layers[4].attention
                    )
                gradients = grad_tape.gradient(
                    loss_train, model.trainable_variables
                )
                # NOTE: There are problems using attention constraints
                # since the d loss / d attention is returned as a
                # tf.IndexedSlices, which in Eager Execution mode
                # cannot be used to update a variable. To solve this
                # problem, uncomment the following line.
                # gradients[4] = tf.convert_to_tensor(gradients[4])

                # Validation loss.
                prob_val = model(tf_inputs_val)
                loss_val = self.loss(
                    prob_val, tf_inputs_val[3], model.layers[4].attention
                )

                # Log progress.
                if self.do_log:
                    if tf.equal(epoch % self.log_freq, 0):
                        tf.summary.scalar('loss_train', loss_train, step=epoch)
                        tf.summary.scalar('loss_val', loss_val, step=epoch)
                        tf_theta = model.layers[4].theta
                        for param_name in tf_theta:
                            tf.summary.scalar(
                                param_name, tf_theta[param_name], step=epoch
                            )
                if verbose > 3:
                    if tf.equal(epoch % self.log_freq, 0):
                        formatted_str = tf.strings.format(
                            '        epoch {} | loss_train: {} | loss_val: {}',
                            (epoch, loss_train, loss_val)
                        )
                        tf.print(formatted_str, output_stream=sys.stderr)

                # Apply gradients (subject to constraints).
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                )

                # Compare current loss to best loss.
                if loss_val < loss_val_best:
                    last_improvement_stop = tf.constant(0, dtype=tf.int32)
                else:
                    last_improvement_stop = (
                        last_improvement_stop + tf.constant(1, dtype=tf.int32)
                    )

                if loss_val < loss_val_best:
                    last_improvement_reduce = tf.constant(0, dtype=tf.int32)
                else:
                    last_improvement_reduce = (
                        last_improvement_reduce +
                        tf.constant(1, dtype=tf.int32)
                    )

                if loss_val < loss_val_best:
                    loss_train_best = loss_train
                    loss_val_best = loss_val
                    epoch_best = epoch + tf.constant(1, dtype=tf.int64)
                    z_best = model.layers[4].z
                    attention_best = model.layers[4].attention
                    theta_best = model.layers[4].theta

                if last_improvement_stop >= tf_patience_stop:
                    break

            return (
                epoch_best, loss_train_best, loss_val_best, z_best,
                attention_best, theta_best
            )

        (
            epoch_best, loss_train_best, loss_val_best, z_best,
            attention_best, theta_best
        ) = train(model, optimizer, tf_inputs_train, tf_inputs_val)

        return (
            loss_train_best, loss_val_best, epoch_best, z_best,
            attention_best, theta_best
        )

    def evaluate(self, obs):
        """Evaluate observations using the current state of the model.

        Arguments:
            obs: A Observations object representing the observed data.

        Returns:
            loss: The average loss per observation. Loss is defined as
                the negative loglikelihood.

        """
        (obs_config_list, obs_config_idx) = self._grab_config_info(obs)
        tf_config = self._prepare_config(obs_config_list)
        tf_inputs = self._prepare_inputs(obs, obs_config_idx)

        model = self._build_model(tf_config, init_mode='hot')

        # Evaluate current model to obtain starting loss.
        prob = model(tf_inputs)
        loss = self.loss(prob, tf_inputs[3], model.layers[4].attention)

        if tf.math.is_nan(loss):
            loss = tf.constant(np.inf, dtype=K.floatx())
        return loss

    @staticmethod
    def _grab_config_info(obs):
        """Grab configuration information."""
        obs_config_list = copy.copy(obs.config_list)
        obs_config_idx = copy.copy(obs.config_idx)
        return obs_config_list, obs_config_idx

    @staticmethod
    def _prepare_config(config_list):
        tf_config = [
            tf.constant(len(config_list.n_outcome.values)),
            tf.constant(config_list.n_reference.values),
            tf.constant(config_list.n_select.values),
            tf.constant(config_list.is_ranked.values)
        ]
        return tf_config

    @staticmethod
    def _prepare_inputs(obs, config_idx):
        tf_obs = [
            tf.constant(
                obs.stimulus_set, dtype=tf.int32, name='obs_stimulus_set'
            ),
            tf.constant(
                config_idx, dtype=tf.int32, name='obs_config_idx'
            ),
            tf.constant(obs.group_id, dtype=tf.int32, name='obs_group_id'),
            tf.constant(obs.weight, dtype=K.floatx(), name='obs_weight')
        ]
        return tf_obs

    def outcome_probability(
            self, docket, group_id=None, z=None, theta=None, phi=None,
            unaltered_only=False):
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
            theta (optional): The theta parameters.
            phi (optional): The phi parameters.
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
            The arguments theta and phi are assigned in a lazy
                manner. If they have a value of None when they are
                needed, the attribute values of the object will be used.

        """
        n_trial_all = docket.n_trial

        if z is None:
            z = self._z["value"]
        else:
            self._check_z(z)

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
        (z_q, z_r) = self._inflate_points(
            docket.stimulus_set, docket.max_n_reference, z
        )
        z_q, z_r, theta, attention = self._broadcast_for_similarity(
            z_q, z_r, group_id=group_id, theta=theta, phi=phi
        )
        sim_qr = self._similarity(z_q, z_r, theta, attention)

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
            prob = np.ones((n_trial, n_outcome, n_sample), dtype=np.float64)
            for i_outcome in range(n_outcome):
                s_qr_perm = sim_qr_config[:, outcome_idx[i_outcome, :], :]
                prob[:, i_outcome, :] = self._ranked_sequence_probability(
                    s_qr_perm, config['n_select']
                )
            prob_all[trial_locs, 0:n_outcome, :] = prob
        prob_all = ma.masked_values(prob_all, -1)

        # Correct for any numerical inaccuracy.
        if not unaltered_only:
            prob_all = ma.divide(
                prob_all, ma.sum(prob_all, axis=1, keepdims=True))

        # Reshape prob_all as necessary.
        if n_sample == 1:
            prob_all = prob_all[:, :, 0]

        return prob_all

    @staticmethod
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

    @staticmethod
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
            z = copy.copy(self._z["value"])
        else:
            z = z_init

        if verbose > 0:
            print('[psiz] Sampling from posterior...')
            progbar = ProgressBar(
                n_total_sample, prefix='Progress:', suffix='Complete',
                length=50
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
        mu = gmm.means_[0]
        sigma = gmm.covariances_[0] * np.identity(n_dim)

        # Center embedding to satisfy assumptions of elliptical slice sampling.
        z = z - mu

        # Define log-likelihood for elliptical slice sampler.
        def log_likelihood(z_part, part_idx, z_full, obs):
            # Assemble full z.
            z_full[part_idx, :] = z_part
            cap = 2.2204e-16
            prob_all = self.outcome_probability(
                obs, group_id=obs.group_id, z=z_full,
                unaltered_only=True
            )
            prob = ma.maximum(cap, prob_all[:, 0])
            ll = ma.sum(ma.log(prob))
            return ll

        # Initialize sampler.
        z_full = copy.copy(z)
        samples = np.empty((n_stimuli, n_dim, n_total_sample))

        # Make first partition.
        n_partition = 2
        part_idx, n_stimuli_part = self._make_partition(
            n_stimuli, n_partition
        )
        # Create a diagonally tiled covariance matrix in order to slice
        # multiple points simultaneously.
        prior = []
        for i_part in range(n_partition):
            prior.append(
                np.linalg.cholesky(
                    self._inflate_sigma(sigma, n_stimuli_part[i_part], n_dim)
                )
            )

        for i_round in range(n_total_sample):
            # Partition stimuli into two groups.
            if np.mod(i_round, 100) == 0:
                if verbose > 0:
                    progbar.update(i_round + 1)

                part_idx, n_stimuli_part = self._make_partition(
                    n_stimuli, n_partition
                )

            for i_part in range(n_partition):
                z_part = z_full[part_idx[i_part], :]
                # Sample.
                (z_part, _) = elliptical_slice(
                    z_part, prior[i_part], log_likelihood,
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

    def tf_posterior_samples(
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
        # Settings.
        n_partition = 2

        start_time_s = time.time()
        n_final_sample = int(n_final_sample)
        n_total_sample = n_burn + (n_final_sample * thin_step)
        n_stimuli = self.n_stimuli
        n_dim = self.n_dim
        if z_init is None:
            z = copy.copy(self._z["value"])
        else:
            z = z_init

        if verbose > 0:
            print('[psiz] Sampling from posterior...')
            progbar = ProgressBar(
                n_total_sample, prefix='Progress:', suffix='Complete',
                length=50
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
        mu = gmm.means_[0]
        sigma = gmm.covariances_[0] * np.identity(n_dim)

        # Center embedding to satisfy assumptions of elliptical slice sampling.
        z = z - mu
        z_orig = tf.constant(z, dtype=K.floatx())
        mu = tf.constant(mu, dtype=K.floatx())
        mu = tf.expand_dims(mu, axis=0)
        mu = tf.expand_dims(mu, axis=0)

        # Prepare obs and model.
        (obs_config_list, obs_config_idx) = self._grab_config_info(obs)
        tf_config = self._prepare_config(obs_config_list)
        tf_inputs = self._prepare_inputs(obs, obs_config_idx)
        model = self._build_model(tf_config, init_mode='hot')
        # flat_log_likelihood overwrites model.z with new tf_z

        # Pre-determine partitions. TODO?
        part_idx, n_stimuli_part = self._make_partition(
            n_stimuli, n_partition
        )
        # NOTE that second row is what we want.
        part_idx = tf.constant(part_idx[1], dtype=tf.int32)

        # Pre-compute prior. TODO?
        # Can I guarantee the number of stimuli for each part doesn't change?
        # Create a diagonally tiled covariance matrix in order to slice
        # multiple points simultaneously.
        prior = []
        for i_part in range(n_partition):
            prior.append(
                tf.constant(
                    np.linalg.cholesky(
                        self._inflate_sigma(
                            sigma, n_stimuli_part[i_part], n_dim
                        )
                    ), dtype=K.floatx()
                )
            )

        n_stimuli_part = tf.constant(n_stimuli_part, dtype=tf.int32)

        # @tf.function
        def flat_log_likelihood(z_part, part_idx, z_full, tf_inputs):
            # Assemble full z. TODO
            z_full = None
            # Assign variable values. TODO
            prob_all = model(tf_inputs)
            cap = tf.constant(2.2204e-16, dtype=K.floatx())
            prob_all = tf.math.log(tf.maximum(prob_all, cap))
            prob_all = tf.multiply(weight, prob_all)
            ll = tf.reduce_sum(prob_all)
            return ll

        # Initialize sampler.
        z_full = tf.constant(z, dtype=K.floatx())
        samples = tf.zeros(
            [n_total_sample, n_stimuli, n_dim], dtype=K.floatx()
        )

        # Perform partition.
        z_part = tf.dynamic_partition(
            z_full,
            part_idx,
            n_partition
        )
        part_indices = tf.dynamic_partition(tf.range(n_stimuli), part_idx, 2)

        z_part_0 = z_part[0]
        z_part_1 = z_part[1]

        z_part_0 = tf.reshape(z_part[0], [-1])
        z_part_1 = tf.reshape(z_part[1], [-1])

        # Sample from prior if there are no observations. TODO

        for i_round in tf.range(n_total_sample):
            # if tf.math.equal(tf.math.floormod(i_round, 100), 0):
            #     # Partition stimuli into two groups. TODO
            #     # Log progress.
            #     if verbose > 0:
            #         progbar.update(i_round + tf.constant(1, dtype=tf.int32))

            (z_part_0, _) = tf_elliptical_slice(
                z_part_0, prior[0], flat_log_likelihood,
                pdf_params=[part_idx[i_part], z_orig, obs]
            )

            (z_part_1, _) = tf_elliptical_slice(
                z_part_1, prior[1], flat_log_likelihood,
                pdf_params=[part_idx[i_part], z_orig, obs]
            )

            # Reshape and stitch.
            z_part_0_r = tf.reshape(z_part_0, (n_stimuli_part[0], n_dim))
            z_part_1_r = tf.reshape(z_part_1, (n_stimuli_part[1], n_dim))
            z_full = tf.dynamic_stitch(part_indices, [z_part_0_r, z_part_1_r])

            i_round_expand = tf.expand_dims(i_round, axis=0)
            i_round_expand = tf.expand_dims(i_round_expand, axis=0)
            samples = tf.tensor_scatter_nd_update(
                samples, i_round_expand, tf.expand_dims(z_full, axis=0)
            )

        # Add back in mean.
        samples = samples + mu

        samples = samples[n_burn::thin_step, :, :]
        sample = sample[0:n_final_sample, :, :]
        # Permute axis. TODO
        samples = dict(z=samples)

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

    @staticmethod
    def _inflate_sigma(sigma, n_stimuli, n_dim):
        """Exploit covariance matrix trick."""
        sigma_inflated = np.zeros((n_stimuli * n_dim, n_stimuli * n_dim))
        for i_stimuli in range(n_stimuli):
            start_idx = i_stimuli * n_dim
            end_idx = ((i_stimuli + 1) * n_dim)
            sigma_inflated[start_idx:end_idx, start_idx:end_idx] = sigma
        return sigma_inflated

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

        grp_z = f.create_group("z")
        grp_z.create_dataset("value", data=self._z["value"])
        grp_z.create_dataset("trainable", data=self._z["trainable"])

        grp_theta = f.create_group("theta")
        for theta_param_name in self._theta:
            grp_theta_param = grp_theta.create_group(theta_param_name)
            grp_theta_param.create_dataset(
                "value", data=self._theta[theta_param_name]["value"])
            grp_theta_param.create_dataset(
                "trainable", data=self._theta[theta_param_name]["trainable"])
            # grp_theta_param.create_dataset(
            #   "bounds", data=self._theta[theta_param_name]["bounds"])

        grp_phi = f.create_group("phi")
        for phi_param_name in self._phi:
            grp_phi_param = grp_phi.create_group(phi_param_name)
            grp_phi_param.create_dataset(
                "value", data=self._phi[phi_param_name]["value"])
            grp_phi_param.create_dataset(
                "trainable", data=self._phi[phi_param_name]["trainable"])

        f.close()

    def subset(self, idx):
        """Return subset of embedding."""
        emb = copy.deepcopy(self)
        emb._z["value"] = emb._z["value"][idx, :]
        emb.n_stimuli = emb._z["value"].shape[0]
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
        z = self._z["value"]
        rho = self._theta["rho"]["value"]
        if np.isscalar(group_id):
            attention_weights = self._phi["w"]["value"][group_id, :]
        else:
            group_id = np.asarray(group_id)
            attention_weights = self._phi["w"]["value"][group_id, :]
            attention_weights = np.mean(attention_weights, axis=0)

        z_group = z * np.expand_dims(attention_weights**(1/rho), axis=0)
        emb._z["value"] = z_group
        emb.n_group = 1
        emb._phi["w"]["value"] = np.ones([1, self.n_dim])
        return emb

    def __deepcopy__(self, memodict={}):
        """Override deepcopy method."""
        # Make shallow copy of whole object.
        cpyobj = type(self)(
            self.n_stimuli, n_dim=self.n_dim, n_group=self.n_group
        )
        # Make deepcopy required attributes
        cpyobj._z = copy.deepcopy(self._z, memodict)
        cpyobj._phi = copy.deepcopy(self._phi, memodict)
        cpyobj._theta = copy.deepcopy(self._theta, memodict)
        # TODO add other necessary attributes.
        return cpyobj


class CoreLayer(Layer):
    """Core layer of model."""

    def __init__(self, tf_theta, tf_attention, tf_z, tf_config, tf_similarity):
        """Initialize.

        Arguments:
            tf_theta:
            tf_attention:
            tf_z:
            tf_config: It is assumed that the indices that will be
                passed in later as inputs will correspond to the
                indices in this data structure.
            tf_similarity:

        """
        super(CoreLayer, self).__init__()
        self.theta = tf_theta
        self.attention = tf_attention
        self.z = tf_z

        self.n_config = tf_config[0]
        self.config_n_reference = tf_config[1]
        self.config_n_select = tf_config[2]
        self.config_is_ranked = tf_config[3]

        self._similarity = tf_similarity

        self.max_n_reference = tf.math.reduce_max(self.config_n_reference)

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

        # Compute the probability of observations for the different
        # trial configurations.
        n_trial = tf.shape(obs_stimulus_set)[0]
        prob_all = tf.zeros([n_trial], dtype=K.floatx())
        for i_config in tf.range(self.n_config):
            n_reference = self.config_n_reference[i_config]
            n_select = self.config_n_select[i_config]
            is_ranked = self.config_is_ranked[i_config]

            # Grab data belonging to current trial configuration.
            locs = tf.equal(obs_config_idx, i_config)
            trial_idx = tf.squeeze(tf.where(locs))
            stimulus_set_config = tf.gather(obs_stimulus_set, trial_idx)
            group_id_config = tf.gather(obs_group_id, trial_idx)

            # Expand attention weights.
            attention_config = tf.gather(self.attention, group_id_config)
            attention_config = tf.expand_dims(attention_config, axis=2)

            # Compute similarity between query and references.
            (z_q, z_r) = self._tf_inflate_points(
                stimulus_set_config, n_reference, self.z
            )
            sim_qr_config = self._similarity(
                z_q, z_r, self.theta, attention_config
            )

            # Compute probability of behavior.
            prob_config = self._tf_ranked_sequence_probability(
                sim_qr_config, n_select
            )

            # Update master results.
            prob_all = tf.tensor_scatter_nd_update(
                prob_all, tf.expand_dims(trial_idx, axis=1), prob_config
            )

        return prob_all

    def _tf_inflate_points(
            self, stimulus_set, n_reference, z):
        """Inflate stimulus set into embedding points.

        Note: This method will not gracefully handle placeholder
        stimulus IDs.

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

    def _tf_ranked_sequence_probability(self, sim_qr, n_select):
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


class Inverse(PsychologicalEmbedding):
    """An inverse-distance model.

    This embedding technique uses the following similarity kernel:
        s(x,y) = 1 / norm(x - y, rho)**tau,
    where x and y are n-dimensional vectors. The similarity kernel has
    two free parameters: rho, tau.

    """

    def __init__(self, n_stimuli, n_dim=2, n_group=1):
        """Initialize.

        Arguments:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded.
            n_dim (optional): An integer indicating the dimensionality
                of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.

        """
        PsychologicalEmbedding.__init__(self, n_stimuli, n_dim, n_group)
        self._theta = self._default_theta()

        # Default inference settings.
        self.lr = 0.001

    def _default_theta(self):
        """Return dictionary of default theta parameters.

        Returns:
            Dictionary of theta parameters.

        """
        cap = 2.2204e-16
        theta = dict(
            rho=dict(value=2., trainable=True, bounds=[1., None]),
            tau=dict(value=1., trainable=True, bounds=[1., None]),
            mu=dict(value=1e-15, trainable=True, bounds=[cap, None])
        )
        return theta

    @property
    def rho(self):
        """Getter method for rho."""
        return self._theta["rho"]["value"]

    @rho.setter
    def rho(self, rho):
        """Setter method for rho."""
        rho = self._check_theta_param('rho', rho)
        self._theta["rho"]["value"] = rho

    @property
    def tau(self):
        """Getter method for tau."""
        return self._theta["tau"]["value"]

    @tau.setter
    def tau(self, tau):
        """Setter method for tau."""
        tau = self._check_theta_param('tau', tau)
        self._theta["tau"]["value"] = tau

    @property
    def mu(self):
        """Getter method for mu."""
        return self._theta["mu"]["value"]

    @mu.setter
    def mu(self, mu):
        """Setter method for mu."""
        mu = self._check_theta_param('mu', mu)
        self._theta["mu"]["value"] = mu

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self._theta['rho']["trainable"]:
            tf_theta['rho'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(1., 3.)(shape=[]),
                trainable=True, name="rho", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['rho']['bounds'][0]
                )
            )
        if self._theta['tau']["trainable"]:
            tf_theta['tau'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(1., 2.)(shape=[]),
                trainable=True, name="tau", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['tau']['bounds'][0]
                )
            )
        if self._theta['mu']["trainable"]:
            tf_theta['mu'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(
                    0.0000000001, .001
                )(shape=[]),
                trainable=True, name="mu", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['mu']['bounds'][0]
                )
            )
        return tf_theta

    def _tf_similarity(self, z_q, z_r, tf_theta, tf_attention):
        """Exponential family similarity kernel.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim)
            z_r: A set of embedding points.
                shape = (n_trial, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_trial, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = tf_theta['rho']
        tau = tf_theta['tau']
        mu = tf_theta['mu']

        # Weighted Minkowski distance.
        d_qr = tf.pow(tf.abs(z_q - z_r), rho)
        d_qr = tf.multiply(d_qr, tf_attention)
        d_qr = tf.pow(tf.reduce_sum(d_qr, axis=1), 1. / rho)

        # Inverse distance similarity kernel.
        sim_qr = 1 / (tf.pow(d_qr, tau) + mu)
        return sim_qr

    @staticmethod
    def _similarity(z_q, z_r, theta, attention):
        """Exponential family similarity kernel.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim)
            z_r: A set of embedding points.
                shape = (n_trial, n_dim)
            theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_trial, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = theta['rho']["value"]
        tau = theta['tau']["value"]
        mu = theta['mu']["value"]

        # Weighted Minkowski distance.
        d_qr = _mink_distance(z_q, z_r, rho, attention)

        # Exponential family similarity kernel.
        sim_qr = 1 / (d_qr**tau + mu)
        return sim_qr


class Exponential(PsychologicalEmbedding):
    """An exponential family stochastic display embedding algorithm.

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

    def __init__(self, n_stimuli, n_dim=2, n_group=1):
        """Initialize.

        Arguments:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded.
            n_dim (optional): An integer indicating the dimensionality
                of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.

        """
        PsychologicalEmbedding.__init__(self, n_stimuli, n_dim, n_group)
        self._theta = self._default_theta()

        # Default inference settings.
        self.lr = 0.001

    def _default_theta(self):
        """Return dictionary of default theta parameters.

        Returns:
            Dictionary of theta parameters.

        """
        theta = dict(
            rho=dict(value=2., trainable=True, bounds=[1., None]),
            tau=dict(value=1., trainable=True, bounds=[1., None]),
            gamma=dict(value=0., trainable=True, bounds=[0., None]),
            beta=dict(value=10., trainable=True, bounds=[1., None])
        )
        return theta

    # TODO
    # @staticmethod
    # def _random_theta():
    #     """Return a dictionary of random theta parameters.

    #     Returns:
    #         Dictionary of theta parameters.

    #     """
    #     theta = dict(
    #         rho=dict(value=2., trainable=True, bounds=[1., None]),
    #         tau=dict(value=1., trainable=True, bounds=[1., None]),
    #         gamma=dict(value=0., trainable=True, bounds=[0., None]),
    #         beta=dict(value=10., trainable=True, bounds=[1., None])
    #     )
    #     return theta

    @property
    def rho(self):
        """Getter method for rho."""
        return self._theta["rho"]["value"]

    @rho.setter
    def rho(self, rho):
        """Setter method for rho."""
        rho = self._check_theta_param('rho', rho)
        self._theta["rho"]["value"] = rho

    @property
    def tau(self):
        """Getter method for tau."""
        return self._theta["tau"]["value"]

    @tau.setter
    def tau(self, tau):
        """Setter method for tau."""
        tau = self._check_theta_param('tau', tau)
        self._theta["tau"]["value"] = tau

    @property
    def gamma(self):
        """Getter method for gamma."""
        return self._theta["gamma"]["value"]

    @gamma.setter
    def gamma(self, gamma):
        """Setter method for gamma."""
        gamma = self._check_theta_param('gamma', gamma)
        self._theta["gamma"]["value"] = gamma

    @property
    def beta(self):
        """Getter method for beta."""
        return self._theta["beta"]["value"]

    @beta.setter
    def beta(self, beta):
        """Setter method for beta."""
        beta = self._check_theta_param('beta', beta)
        self._theta["beta"]["value"] = beta

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self._theta['rho']["trainable"]:
            tf_theta['rho'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(1., 3.)(shape=[]),
                trainable=True, name="rho", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['rho']['bounds'][0]
                )
            )
        if self._theta['tau']["trainable"]:
            tf_theta['tau'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(1., 2.)(shape=[]),
                trainable=True, name="tau", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['tau']['bounds'][0]
                )
            )
        if self._theta['gamma']["trainable"]:
            tf_theta['gamma'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(
                    0., .001
                )(shape=[]),
                trainable=True, name="gamma", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['gamma']['bounds'][0]
                )
            )
        if self._theta['beta']["trainable"]:
            tf_theta['beta'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(1., 30.)(shape=[]),
                trainable=True, name="beta", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['beta']['bounds'][0]
                )
            )
        return tf_theta

    def _tf_similarity(self, z_q, z_r, tf_theta, tf_attention):
        """Exponential family similarity kernel.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim)
            z_r: A set of embedding points.
                shape = (n_trial, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_trial, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = tf_theta['rho']
        tau = tf_theta['tau']
        gamma = tf_theta['gamma']
        beta = tf_theta['beta']

        # Weighted Minkowski distance.
        d_qr = tf.pow(tf.abs(z_q - z_r), rho)
        d_qr = tf.multiply(d_qr, tf_attention)
        d_qr = tf.pow(tf.reduce_sum(d_qr, axis=1), 1. / rho)

        # Exponential family similarity kernel.
        sim_qr = tf.exp(tf.negative(beta) * tf.pow(d_qr, tau)) + gamma
        return sim_qr

    @staticmethod
    def _similarity(z_q, z_r, theta, attention):
        """Exponential family similarity kernel.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim)
            z_r: A set of embedding points.
                shape = (n_trial, n_dim)
            theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_trial, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = theta['rho']["value"]
        tau = theta['tau']["value"]
        gamma = theta['gamma']["value"]
        beta = theta['beta']["value"]

        # Weighted Minkowski distance.
        d_qr = _mink_distance(z_q, z_r, rho, attention)

        # Exponential family similarity kernel.
        sim_qr = np.exp(np.negative(beta) * d_qr**tau) + gamma  # TODO
        # sim_qr = ne.evaluate('exp(-beta * d_qr**tau) + gamma')
        return sim_qr


class HeavyTailed(PsychologicalEmbedding):
    """A heavy-tailed family stochastic display embedding algorithm.

    This embedding technique uses the following similarity kernel:
        s(x,y) = (kappa + (norm(x-y, rho).^tau)).^(-alpha),
    where x and y are n-dimensional vectors. The similarity kernel has
    four free parameters: rho, tau, kappa, and alpha. The
    heavy-tailed family is a generalization of the Student-t family.
    """

    def __init__(self, n_stimuli, n_dim=2, n_group=1):
        """Initialize.

        Arguments:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded.
            n_dim (optional): An integer indicating the dimensionality
                of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.

        """
        PsychologicalEmbedding.__init__(self, n_stimuli, n_dim, n_group)
        self._theta = self._default_theta()

        # Default inference settings.
        self.lr = 0.003

    def _default_theta(self):
        """Return dictionary of default theta parameters.

        Returns:
            Dictionary of theta parameters.

        """
        theta = dict(
            rho=dict(value=2., trainable=True, bounds=[1., None]),
            tau=dict(value=1., trainable=True, bounds=[1., None]),
            kappa=dict(value=2., trainable=True, bounds=[0., None]),
            alpha=dict(value=30., trainable=True, bounds=[0., None])
        )
        return theta

    @property
    def rho(self):
        """Getter method for rho."""
        return self._theta["rho"]["value"]

    @rho.setter
    def rho(self, rho):
        """Setter method for rho."""
        rho = self._check_theta_param('rho', rho)
        self._theta["rho"]["value"] = rho

    @property
    def tau(self):
        """Getter method for tau."""
        return self._theta["tau"]["value"]

    @tau.setter
    def tau(self, tau):
        """Setter method for tau."""
        tau = self._check_theta_param('tau', tau)
        self._theta["tau"]["value"] = tau

    @property
    def kappa(self):
        """Getter method for kappa."""
        return self._theta["kappa"]["value"]

    @kappa.setter
    def kappa(self, kappa):
        """Setter method for kappa."""
        kappa = self._check_theta_param('kappa', kappa)
        self._theta["kappa"]["value"] = kappa

    @property
    def alpha(self):
        """Getter method for alpha."""
        return self._theta["alpha"]["value"]

    @alpha.setter
    def alpha(self, alpha):
        """Setter method for alpha."""
        alpha = self._check_theta_param('alpha', alpha)
        self._theta["alpha"]["value"] = alpha

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self._theta['rho']["trainable"]:
            tf_theta['rho'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(1., 3.)(shape=[]),
                trainable=True, name="rho", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['rho']['bounds'][0]
                )
            )
        if self._theta['tau']["trainable"]:
            tf_theta['tau'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(1., 2.)(shape=[]),
                trainable=True, name="tau", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['tau']['bounds'][0]
                )
            )
        if self._theta['kappa']["trainable"]:
            tf_theta['kappa'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(1., 11.)(shape=[]),
                trainable=True, name="kappa", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['kappa']['bounds'][0]
                )
            )
        if self._theta['alpha']["trainable"]:
            tf_theta['alpha'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(
                    10., 60.
                )(shape=[]), trainable=True, name="alpha",
                dtype=K.floatx(), constraint=GreaterEqualThan(
                    min_value=self._theta['alpha']['bounds'][0]
                )
            )
        return tf_theta

    def _tf_similarity(self, z_q, z_r, tf_theta, tf_attention):
        """Heavy-tailed family similarity kernel.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim)
            z_r: A set of embedding points.
                shape = (n_trial, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_trial, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = tf_theta['rho']
        tau = tf_theta['tau']
        kappa = tf_theta['kappa']
        alpha = tf_theta['alpha']

        # Weighted Minkowski distance.
        d_qr = tf.pow(tf.abs(z_q - z_r), rho)
        d_qr = tf.multiply(d_qr, tf_attention)
        d_qr = tf.pow(tf.reduce_sum(d_qr, axis=1), 1. / rho)

        # Heavy-tailed family similarity kernel.
        sim_qr = tf.pow(kappa + tf.pow(d_qr, tau), (tf.negative(alpha)))
        return sim_qr

    @staticmethod
    def _similarity(z_q, z_r, theta, attention):
        """Heavy-tailed family similarity kernel.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim)
            z_r: A set of embedding points.
                shape = (n_trial, n_dim)
            theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_trial, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = theta['rho']["value"]
        tau = theta['tau']["value"]
        kappa = theta['kappa']["value"]
        alpha = theta['alpha']["value"]

        # Weighted Minkowski distance.
        d_qr = _mink_distance(z_q, z_r, rho, attention)

        # Heavy-tailed family similarity kernel.
        sim_qr = (kappa + d_qr**tau)**(np.negative(alpha))
        return sim_qr


class StudentsT(PsychologicalEmbedding):
    """A Student's t family stochastic display embedding algorithm.

    The embedding technique uses the following similarity kernel:
        s(x,y) = (1 + (((norm(x-y, rho)^tau)/alpha))^(-(alpha + 1)/2),
    where x and y are n-dimensional vectors. The similarity kernel has
    three free parameters: rho, tau, and alpha. The original Student-t
    kernel proposed by van der Maaten [1] uses the parameter settings
    rho=2, tau=2, and alpha=n_dim-1. By default, this embedding
    algorithm will only infer the embedding and not the free parameters
    associated with the similarity kernel. This behavior can be changed
    by setting the inference flags (e.g.,infer_alpha = True).

    References:
    [1] van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic
        triplet embedding. In Machine learning for signal processing
        (MLSP), 2012 IEEE international workshop on (p. 1-6).
        doi:10.1109/MLSP.2012.6349720

    """

    def __init__(self, n_stimuli, n_dim=2, n_group=1):
        """Initialize.

        Arguments:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded.
            n_dim (optional): An integer indicating the dimensionality
                of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.

        """
        PsychologicalEmbedding.__init__(self, n_stimuli, n_dim, n_group)
        self._theta = self._default_theta()

        # Default inference settings.
        self.lr = 0.01

    def _default_theta(self):
        """Return dictionary of default theta parameters.

        Returns:
            Dictionary of theta parameters.

        """
        theta = dict(
            rho=dict(value=2., trainable=False, bounds=[1., None]),
            tau=dict(value=2., trainable=False, bounds=[1., None]),
            alpha=dict(
                value=(self.n_dim - 1.),
                trainable=False,
                bounds=[0.000001, None]
            ),
        )
        return theta

    @property
    def rho(self):
        """Getter method for rho."""
        return self._theta["rho"]["value"]

    @rho.setter
    def rho(self, rho):
        """Setter method for rho."""
        rho = self._check_theta_param('rho', rho)
        self._theta["rho"]["value"] = rho

    @property
    def tau(self):
        """Getter method for tau."""
        return self._theta["tau"]["value"]

    @tau.setter
    def tau(self, tau):
        """Setter method for tau."""
        tau = self._check_theta_param('tau', tau)
        self._theta["tau"]["value"] = tau

    @property
    def alpha(self):
        """Getter method for alpha."""
        return self._theta["alpha"]["value"]

    @alpha.setter
    def alpha(self, alpha):
        """Setter method for alpha."""
        alpha = self._check_theta_param('alpha', alpha)
        self._theta["alpha"]["value"] = alpha

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self._theta['rho']["trainable"]:
            tf_theta['rho'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(1., 3.)(shape=[]),
                trainable=True, name="rho", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['rho']['bounds'][0]
                )
            )
        if self._theta['tau']["trainable"]:
            tf_theta['tau'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(1., 2.)(shape=[]),
                trainable=True, name="tau", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['tau']['bounds'][0]
                )
            )
        if self._theta['alpha']["trainable"]:
            min_alpha = np.max((1, self.n_dim - 5.))
            max_alpha = self.n_dim + 5.
            tf_theta['alpha'] = tf.Variable(
                initial_value=tf.random_uniform_initializer(
                    min_alpha, max_alpha
                )(shape=[]), trainable=True, name="alpha", dtype=K.floatx(),
                constraint=GreaterEqualThan(
                    min_value=self._theta['alpha']['bounds'][0]
                )
            )
        return tf_theta

    def _tf_similarity(self, z_q, z_r, tf_theta, tf_attention):
        """Student-t family similarity kernel.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim)
            z_r: A set of embedding points.
                shape = (n_trial, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_trial, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = tf_theta['rho']
        tau = tf_theta['tau']
        alpha = tf_theta['alpha']

        # Weighted Minkowski distance.
        d_qr = tf.pow(tf.abs(z_q - z_r), rho)
        d_qr = tf.multiply(d_qr, tf_attention)
        d_qr = tf.pow(tf.reduce_sum(d_qr, axis=1), 1. / rho)

        # Student-t family similarity kernel.
        sim_qr = tf.pow(
            1 + (tf.pow(d_qr, tau) / alpha), tf.negative(alpha + 1)/2)
        return sim_qr

    @staticmethod
    def _similarity(z_q, z_r, theta, attention):
        """Student-t family similarity kernel.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim)
            z_r: A set of embedding points.
                shape = (n_trial, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_trial, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = theta['rho']["value"]
        tau = theta['tau']["value"]
        alpha = theta['alpha']["value"]

        # Weighted Minkowski distance.
        d_qr = _mink_distance(z_q, z_r, rho, attention)

        # Student-t family similarity kernel.
        sim_qr = (1 + (d_qr**tau / alpha))**(np.negative(alpha + 1)/2)
        return sim_qr


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


class GreaterEqualThan(Constraint):
    """Constrains the weights to be greater than a specified value."""

    def __init__(self, min_value=0.):
        """Initialize."""
        self.min_value = min_value

    def __call__(self, w):
        """Call."""
        w_adj = w - self.min_value
        w2 = w_adj * tf.cast(tf.math.greater_equal(w_adj, 0.), K.floatx())
        w2 = w2 + self.min_value
        return w2


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


def load_embedding(filepath):
    """Load embedding model saved via the save method.

    The loaded data is instantiated as a concrete class of
    SimilarityTrials.

    Arguments:
        filepath: The location of the hdf5 file to load.

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

    if embedding_type == 'Exponential':
        embedding = Exponential(n_stimuli, n_dim=n_dim, n_group=n_group)
    elif embedding_type == 'HeavyTailed':
        embedding = HeavyTailed(n_stimuli, n_dim=n_dim, n_group=n_group)
    elif embedding_type == 'StudentsT':
        embedding = StudentsT(n_stimuli, n_dim=n_dim, n_group=n_group)
    else:
        raise ValueError(
            'No class found matching the provided `embedding_type`.')

    for name in f['z']:
        embedding._z[name] = f['z'][name][()]

    for p_name in f['theta']:
        for name in f['theta'][p_name]:
            embedding._theta[p_name][name] = f['theta'][p_name][name][()]

    for p_name in f['phi']:
        # Patch for older models using `phi_1` variable name.
        if p_name == 'phi_1':
            p_name_new = 'w'
        else:
            p_name_new = p_name
        for name in f['phi'][p_name]:
            embedding._phi[p_name_new][name] = f['phi'][p_name][name][()]

    f.close()
    return embedding


def elliptical_slice(
        initial_theta, prior, lnpdf, pdf_params=(), angle_range=None):
    """Return samples from elliptical slice sampler.

    Markov chain update for a distribution with a Gaussian "prior"
    factored out.

    Arguments:
        initial_theta: initial vector
        prior: cholesky decomposition of the covariance matrix (like
            what numpy.linalg.cholesky returns)
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

    # Determine number of variables.
    theta_shape = initial_theta.shape
    D = initial_theta.size
    if not prior.shape[0] == D or not prior.shape[1] == D:
        raise IOError("Prior must be given by a DxD chol(Sigma)")
    nu = np.dot(prior, np.random.normal(size=D))
    # Reshape nu to reflect shape of theta.
    nu = np.reshape(nu, theta_shape, order='C')

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


def _mink_distance(z_q, z_r, rho, attention):
    """Weighted minkowski distance function.

    Arguments:
        z_q: A set of embedding points.
            shape = (n_trial, n_dim)
        z_r: A set of embedding points.
            shape = (n_trial, n_dim)
        rho: Scalar value controlling the metric. Must be [1, inf[ to
            be a valid metric.
        attention: The weights allocated to each dimension
            in a weighted minkowski metric.
            shape = (n_trial, n_dim)

    Returns:
        The corresponding similarity between rows of embedding
            points.
            shape = (n_trial,)

    Note:
        The implementation for ne.sum appears to be slower than np.sum.
        Furthermore, since ne.sum must be the last call, it forces at
        least two evaluate methods, reducing the benefit of use ne.

    """
    d_qr = attention * ((np.abs(z_q - z_r))**rho)
    d_qr = np.sum(d_qr, axis=1)**(1. / rho)

    # d_qr = ne.evaluate("attention * ((abs(z_q - z_r))**rho)")  # TODO
    # d_qr = np.sum(d_qr, axis=1)**(1. / rho)

    return d_qr


@tf.function
def default_loss(prob_all, weight, tf_attention):
    """Compute model loss given observation probabilities."""
    n_trial = tf.shape(prob_all)[0]
    n_trial = tf.cast(n_trial, dtype=K.floatx())

    # Convert to (weighted) log probabilities.
    cap = tf.constant(2.2204e-16, dtype=K.floatx())
    prob_all = tf.math.log(tf.maximum(prob_all, cap))
    prob_all = tf.multiply(weight, prob_all)

    # Divide by number of trials to make train and test loss
    # comparable.
    loss = tf.negative(tf.reduce_sum(prob_all))
    loss = tf.divide(loss, n_trial)

    return loss
