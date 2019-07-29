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

Todo:
    - Expose optimizer
    - Expand posterior sampler to handle theta and phi and sampling
        from the prior.
    - Allow different float precision.
    - Document broadcasting in similarity function.
    - MAYBE allow different elements of z to be trainable or not.

"""

import sys
from abc import ABCMeta, abstractmethod
import warnings
import copy
from random import randint

import h5py
import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import StratifiedKFold
from sklearn import mixture
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.initializers import Initializer

from psiz.utils import elliptical_slice

FLOAT_X = tf.float32  # TODO


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
        probability: Return the probability of the possible outcomes
            for each trial.
        log_likelihood: Return the log-likelihood of a set of
            observations.
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
            indicates the bounds of the paramter during inference. The
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
        log_dir: The location of the logs. The defualt location is
            `/tmp/psiz/tensorboard_logs/`.
        
    Notes:
        The setter methods as well as the methods fit, trainable, and
            set_log modify the state of the PsychologicalEmbedding
            object.
        The abstract methods _default_theta,
            _get_similarity_parameters_cold, and _tf_similarity must be
            implemented by each concrete class.

    """

    __metaclass__ = ABCMeta

    def __init__(self, n_stimuli, n_dim=2, n_group=1):
        """Initialize.

        Arguments:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded. This must be equal to or
                greater than three.
            n_dim (optional): An integer indicating the dimensionalty
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

        super().__init__()

    def _init_z(self):
        """Return initialized embedding points.

        Initialize random embedding points using a multivariate
            Gaussian.
        """
        mean = np.ones((self.n_dim))
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
        with tf.compat.v1.variable_scope("similarity_params"):
            tf_theta = {}
            if init_mode is 'hot':
                tf_theta = self._get_similarity_parameters_hot()
            else:
                tf_theta = self._get_similarity_parameters_cold()

            # If a parameter is untrainable, set the parameter value to the
            # value in the class attribute theta.
            for param_name in self._theta:
                if not self._theta[param_name]["trainable"]:
                    tf_theta[param_name] = tf.compat.v1.get_variable(
                        param_name, [1], dtype=FLOAT_X,
                        initializer=tf.constant_initializer(
                            self._theta[param_name]["value"], dtype=FLOAT_X
                        ),
                        trainable=False
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
                tf_theta[param_name] = tf.compat.v1.get_variable(
                    param_name, [1], dtype=FLOAT_X,
                    initializer=tf.constant_initializer(
                        self._theta[param_name]["value"], dtype=FLOAT_X
                    ),
                    trainable=True
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

    def _get_similarity_constraints(self, tf_theta):
        """Return a TensorFlow group of parameter constraints.

        Returns:
            tf_theta_bounds: A TensorFlow operation that imposes
                boundary constraints on the algorithm-specific free
                parameters during inference.

        """
        constraint_list = []
        for param_name in self._theta:
            bounds = self._theta[param_name]["bounds"]
            if bounds[0] is not None:
                # Add lower bound.
                constraint_list.append(
                    tf_theta[param_name].assign(tf.maximum(
                        tf.constant(bounds[0], dtype=FLOAT_X),
                        tf_theta[param_name])
                    )
                )
            if bounds[1] is not None:
                # Add upper bound.
                constraint_list.append(
                    tf_theta[param_name].assign(tf.minimum(
                        tf.constant(bounds[1], dtype=FLOAT_X),
                        tf_theta[param_name])
                    )
                )
        tf_theta_bounds = tf.group(*constraint_list)
        return tf_theta_bounds

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
                self._phi["w"]["trainable"] = np.zeros(self.n_group, dtype=bool)
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

        sim = self._similarity(z_q, z_r, theta, attention)
        return sim

    def distance(self, z_q, z_r, group_id=None, theta=None, phi=None):
        """Return dsitance between two lists of points.

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

        d = self._distance(z_q, z_r, theta, attention)
        return d

    def _distance(self, z_q, z_r, theta, attention):
        """Weighted minkowski distance function.

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

        # Weighted Minkowski distance.
        d_qref = (np.abs(z_q - z_r))**rho
        d_qref = np.multiply(d_qref, attention)
        d_qref = np.sum(d_qref, axis=1)**(1. / rho)

        return d_qref

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

    @abstractmethod
    def _similarity(self, z_q, z_r, theta, attention):
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
        constraint_list = []
        for group_id in range(self.n_group):
            tf_attention = self._get_group_attention(init_mode, group_id)
            attention_list.append(tf_attention)
            constraint_list.append(
                tf_attention.assign(self._project_attention(tf_attention))
            )
        tf_attention_constraint = tf.group(*constraint_list)
        tf_attention = tf.concat(attention_list, axis=0)
        return tf_attention, tf_attention_constraint

    def _get_group_attention(self, init_mode, group_id):
        tf_var_name = "attention_{0}".format(group_id)
        if self._phi['w']["trainable"][group_id]:
            if init_mode is 'hot':
                tf_attention = tf.compat.v1.get_variable(
                    tf_var_name, [1, self.n_dim], dtype=FLOAT_X,
                    initializer=tf.constant_initializer(
                        self._phi['w']["value"][group_id, :], dtype=FLOAT_X
                    )
                )
            else:
                n_dim = tf.constant(self.n_dim, dtype=FLOAT_X)
                alpha = tf.constant(np.ones((self.n_dim)), dtype=FLOAT_X)
                tf_attention = tf.compat.v1.get_variable(
                    tf_var_name, [1, self.n_dim],
                    initializer=RandomAttention(n_dim, alpha, dtype=FLOAT_X)
                )
        else:
            tf_attention = tf.compat.v1.get_variable(
                tf_var_name, [1, self.n_dim], dtype=FLOAT_X,
                initializer=tf.constant_initializer(
                    self._phi['w']["value"][group_id, :], dtype=FLOAT_X),
                trainable=False
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
                tf_z = tf.compat.v1.get_variable(
                    "z", [self.n_stimuli, self.n_dim], dtype=FLOAT_X,
                    initializer=tf.constant_initializer(
                        z, dtype=FLOAT_X)
                )
            else:
                tf_z = tf.compat.v1.get_variable(
                    "z", [self.n_stimuli, self.n_dim], dtype=FLOAT_X,
                    initializer=RandomEmbedding(
                        mean=tf.zeros([self.n_dim], dtype=FLOAT_X),
                        stdev=tf.ones([self.n_dim], dtype=FLOAT_X),
                        minval=tf.constant(-3., dtype=FLOAT_X),
                        maxval=tf.constant(0., dtype=FLOAT_X),
                        dtype=FLOAT_X
                    )
                )
        else:
            tf_z = tf.compat.v1.get_variable(
                "z", [self.n_stimuli, self.n_dim], dtype=FLOAT_X,
                initializer=tf.constant_initializer(
                    self._z["value"], dtype=FLOAT_X
                ),
                trainable=False
            )
        tf_z_constraint = tf_z.assign(self._center_z(tf_z))
        return tf_z, tf_z_constraint

    def set_log(self, do_log, log_dir=None, delete_prev=False):
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

        if delete_prev:
            if tf.gfile.Exists(self.log_dir):
                tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

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
                thier defined support. When fitting using a `hot`
                initiazation, trainable parameters will continue from
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
        #  Infer embedding.
        if (verbose > 0):
            print('Inferring embedding...')
        if (verbose > 1):
            print('    Settings:')
            print(
                '    n_stimuli: {0} | n_dim: {1} | n_group: {2}'
                ' | n_obs: {3} | n_restart: {4}'.format(
                    self.n_stimuli, self.n_dim, self.n_group,
                    obs.n_trial, n_restart))
            print('')

        # Partition observations into train and validation set to
        # control early stopping of embedding algorithm.
        skf = StratifiedKFold(n_splits=10)
        (train_idx, val_idx) = list(
            skf.split(obs.stimulus_set, obs.config_idx))[0]
        obs_train = obs.subset(train_idx)
        obs_val = obs.subset(val_idx)

        # Evaluate current model to obtain starting loss.
        loss_train_best = self.evaluate(obs_train)
        loss_val_best = self.evaluate(obs_val)

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

        # Initialize new model.
        (tf_loss, tf_z, tf_z_constraint, tf_attention,
            tf_attention_constraint, tf_theta, tf_theta_bounds,
            tf_obs) = self._core_model(init_mode)

        # Bind observations.
        tf_obs_train = self._bind_obs(tf_obs, obs_train)
        tf_obs_val = self._bind_obs(tf_obs, obs_val)

        # Define optimizer op.
        tf_learning_rate = tf.compat.v1.placeholder(FLOAT_X, shape=[])
        train_op = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=tf_learning_rate
        ).minimize(tf_loss)

        # Define summary op.
        with tf.name_scope('summaries'):
            # Create a summary to monitor loss.
            tf.compat.v1.summary.scalar('loss_train', tf_loss)
            # Create a summary of the embedding tensor.
            # tf.summary.tensor_summary('z', tf_z)  # Feature not supported.
            # Create a summary to monitor parameters of similarity kernel.
            with tf.name_scope('similarity'):
                for param_name in tf_theta:
                    param_mean = tf.reduce_mean(tf_theta[param_name])
                    tf.compat.v1.summary.scalar(
                        param_name + '_mean', param_mean
                    )
            summary_op = tf.compat.v1.summary.merge_all()

        # Run multiple restarts of embedding algorithm.
        sess = tf.compat.v1.Session()
        for i_restart in range(n_restart):
            if (verbose > 2):
                print('        Restart {0}'.format(i_restart))
            (
                loss_train, loss_val, epoch, z, attention, theta
            ) = self._fit_restart(
                sess, tf_loss, tf_z, tf_z_constraint, tf_attention,
                tf_attention_constraint, tf_theta, tf_theta_bounds,
                tf_learning_rate, train_op, summary_op, tf_obs_train,
                tf_obs_val, i_restart, verbose
            )

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

        sess.close()
        tf.compat.v1.reset_default_graph()

        if (verbose > 1):
            if beat_init:
                print(
                    '        Best Restart\n        n_epoch: {0} | '
                    'loss: {1: .6f} | loss_val: {2: .6f}'.format(
                        epoch, loss_train_best, loss_val_best
                    )
                )
            else:
                print('        Did not beat initialization.')

        self._z["value"] = z_best
        self._phi['w']["value"] = attention_best
        self._set_theta(theta_best)

        return loss_train_best, loss_val_best

    def _fit_restart(
            self, sess, tf_loss, tf_z, tf_z_constraint,
            tf_attention, tf_attention_constraint,
            tf_theta, tf_theta_bounds, tf_learning_rate, train_op, summary_op,
            tf_obs_train, tf_obs_val, i_restart, verbose):
        """Embed using a TensorFlow implementation."""
        sess.run(tf.compat.v1.global_variables_initializer())

        # Write logs for TensorBoard
        if self.do_log:
            summary_writer = tf.summary.FileWriter(
                '%s/%s' % (self.log_dir, i_restart),
                graph=tf.get_default_graph()
            )

        loss_train_best = np.inf
        loss_val_best = np.inf

        lr = copy.copy(self.lr)

        last_improvement_stop = 0
        last_improvement_reduce = 0
        z_best = None

        for epoch in range(self.max_n_epoch):
            _, loss_train, summary = sess.run(
                [train_op, tf_loss, summary_op],
                feed_dict={tf_learning_rate: lr, **tf_obs_train}
            )
            sess.run(tf_theta_bounds)
            sess.run(tf_z_constraint)
            sess.run(tf_attention_constraint)
            loss_val = sess.run(
                tf_loss, feed_dict=tf_obs_val
            )

            # Compare current loss to best loss.
            if loss_val < loss_val_best:
                last_improvement_stop = 0
            else:
                last_improvement_stop = last_improvement_stop + 1

            if loss_val < loss_val_best:
                last_improvement_reduce = 0
            else:
                last_improvement_reduce = last_improvement_reduce + 1

            if loss_val < loss_val_best:
                loss_train_best = loss_train
                loss_val_best = loss_val
                epoch_best = epoch + 1
                (z_best, attention_best) = sess.run(
                    [tf_z, tf_attention])
                theta_best = {}
                for param_name in tf_theta:
                    theta_best[param_name] = {}
                    theta_best[param_name]["value"] = sess.run(
                        tf_theta[param_name]
                    )[0]

            if last_improvement_stop >= self.patience_stop:
                break

            if not epoch % 10:
                # Write logs at every 10th iteration
                if self.do_log:
                    summary_writer.add_summary(summary, epoch)
            if verbose > 3:
                print(
                    "        epoch {0:5d} | ".format(epoch),
                    "loss: {0: .6f} | ".format(loss_train),
                    "loss_val: {0: .6f}".format(loss_val)
                )

        # Handle pathological case where there is no improvement.
        if z_best is None:
            epoch_best = epoch + 1
            z_best = self._z["value"]
            attention_best = self._phi['w']["value"]
            theta_best = self._theta

        return (
            loss_train_best, loss_val_best, epoch_best, z_best, attention_best,
            theta_best)

    def evaluate(self, obs):
        """Evaluate observations using the current state of the model.

        Arguments:
            obs: A Observations object representing the observed data.

        Returns:
            loss: The average loss per observation. Loss is defined as
                the negative loglikelihood.

        """
        (tf_loss, _, _, _, _, _, _, tf_obs) = self._core_model('hot')

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)
        loss = sess.run(tf_loss, feed_dict=self._bind_obs(tf_obs, obs))

        sess.close()
        tf.compat.v1.reset_default_graph()

        if np.isnan(loss):
            loss = np.inf
        return loss

    def _bind_obs(self, tf_obs, obs):
        feed_dict = {
            tf_obs['stimulus_set']: obs.stimulus_set,
            tf_obs['n_reference']: obs.n_reference,
            tf_obs['n_select']: obs.n_select,
            tf_obs['is_ranked']: obs.is_ranked,
            tf_obs['group_id']: obs.group_id,
            tf_obs['weight']: obs.weight,
            tf_obs['config_idx']: obs.config_idx,
            tf_obs['n_config']: len(obs.config_list.n_outcome.values),
            tf_obs['max_n_outcome']: np.max(obs.config_list.n_outcome.values),
            tf_obs['config_n_reference']: obs.config_list.n_reference.values,
            tf_obs['config_n_select']: obs.config_list.n_select.values,
            tf_obs['config_is_ranked']: obs.config_list.is_ranked.values,
            tf_obs['config_n_outcome']: obs.config_list.n_outcome.values,
            tf_obs['config_outcome_tensor']: obs.outcome_tensor()
        }
        return feed_dict

    def _center_z(self, tf_z):
        """Return zero-centered embedding.

        Constraint is used to improve numerical stability.
        """
        tf_mean = tf.reduce_mean(tf_z, axis=0, keepdims=True)
        tf_z_centered = tf_z - tf_mean
        return tf_z_centered

    def _project_attention(self, tf_attention_0):
        """Return projection of attention weights."""
        n_dim = tf.shape(tf_attention_0, out_type=FLOAT_X)[1]
        tf_attention_1 = tf.divide(
            tf.reduce_sum(tf_attention_0, axis=1, keepdims=True), n_dim
        )
        tf_attention_proj = tf.divide(
            tf_attention_0, tf_attention_1
        )
        return tf_attention_proj

    def _core_model(self, init_mode):
        """Embedding model implemented using TensorFlow."""
        with tf.compat.v1.variable_scope("model"):
            # Free parameters.
            tf_theta = self._get_similarity_parameters(init_mode)
            tf_theta_bounds = self._get_similarity_constraints(tf_theta)
            tf_attention, tf_attention_constraint = self._get_attention(
                init_mode
            )
            tf_z, tf_z_constraint = self._get_embedding(init_mode)

            # Observation data.
            tf_stimulus_set = tf.compat.v1.placeholder(
                tf.int32, [None, None], name='stimulus_set'
            )
            tf_n_reference = tf.compat.v1.placeholder(
                tf.int32, [None], name='n_reference')
            tf_n_select = tf.compat.v1.placeholder(
                tf.int32, [None], name='n_select')
            tf_is_ranked = tf.compat.v1.placeholder(
                tf.int32, [None], name='is_ranked')
            tf_group_id = tf.compat.v1.placeholder(
                tf.int32, [None], name='group_id')
            tf_weight = tf.compat.v1.placeholder(
                FLOAT_X, [None], name='weight')
            tf_config_idx = tf.compat.v1.placeholder(
                tf.int32, [None], name='config_idx')
            # Configuration data.
            tf_n_config = tf.compat.v1.placeholder(
                tf.int32, shape=(), name='n_config'
            )
            tf_max_n_outcome = tf.compat.v1.placeholder(
                tf.int32, shape=(), name='max_n_outcome'
            )
            tf_config_n_reference = tf.compat.v1.placeholder(
                tf.int32, [None], name='config_n_reference')
            tf_config_n_select = tf.compat.v1.placeholder(
                tf.int32, [None], name='config_n_select')
            tf_config_is_ranked = tf.compat.v1.placeholder(
                tf.int32, [None], name='config_is_ranked')
            tf_config_n_outcome = tf.compat.v1.placeholder(
                tf.int32, [None], name='config_n_outcome')
            tf_outcome_tensor = tf.compat.v1.placeholder(
                tf.int32, name='config_outcome_tensor')
            tf_obs = {
                'stimulus_set': tf_stimulus_set,
                'n_reference': tf_n_reference,
                'n_select': tf_n_select,
                'is_ranked': tf_is_ranked,
                'group_id': tf_group_id,
                'weight': tf_weight,
                'config_idx': tf_config_idx,
                'n_config': tf_n_config,
                'max_n_outcome': tf_max_n_outcome,
                'config_n_reference': tf_config_n_reference,
                'config_n_select': tf_config_n_select,
                'config_is_ranked': tf_config_is_ranked,
                'config_n_outcome': tf_config_n_outcome,
                'config_outcome_tensor': tf_outcome_tensor
            }

            n_trial = tf.shape(tf_stimulus_set)[0]
            max_n_reference = tf.shape(tf_stimulus_set)[1] - 1

            # Expand attention weights.
            tf_atten_expanded = tf.gather(tf_attention, tf_group_id)

            # Compute similarity between query and references.
            (z_q, z_r) = self._tf_inflate_points(
                tf_stimulus_set, max_n_reference, tf_z)
            sim_qr = self._tf_similarity(
                z_q, z_r, tf_theta, tf.expand_dims(tf_atten_expanded, axis=2)
            )

            # Compute the probability of observations for the different trial
            # configurations.
            cap = tf.constant(2.2204e-16, dtype=FLOAT_X)
            prob_all = tf.zeros((0), dtype=FLOAT_X)
            cond_idx = tf.constant(0, dtype=tf.int32)

            def cond_fn(cond_idx, prob_all):
                return tf.less(cond_idx, tf_n_config)

            def body_fn(cond_idx, prob_all):
                n_reference = tf_n_reference[cond_idx]
                n_select = tf_n_select[cond_idx]
                trial_idx = tf.squeeze(tf.where(tf.logical_and(
                    tf.equal(tf_n_reference, n_reference),
                    tf.equal(tf_n_select, n_select)))
                )
                weight_config = tf.gather(tf_weight, trial_idx)
                sim_qr_config = tf.gather(sim_qr, trial_idx)
                reference_idx = tf.range(
                    0, n_reference, delta=1, dtype=tf.int32)
                sim_qr_config = tf.gather(sim_qr_config, reference_idx, axis=1)
                sim_qr_config.set_shape((None, None))
                prob_config = self._tf_ranked_sequence_probability(
                    sim_qr_config, n_select)
                prob_config = tf.math.log(tf.maximum(prob_config, cap))
                prob_config = tf.multiply(weight_config, prob_config)
                prob_all = tf.concat((prob_all, prob_config), axis=0)
                cond_idx = cond_idx + 1
                return [cond_idx, prob_all]

            r = tf.while_loop(
                cond_fn, body_fn,
                loop_vars=[cond_idx, prob_all],
                shape_invariants=[cond_idx.get_shape(), tf.TensorShape([None])]
            )
            prob_all = r[1]

            # Cost function
            n_trial = tf.cast(n_trial, dtype=FLOAT_X)
            loss = tf.negative(tf.reduce_sum(prob_all))
            # Divide by number of trials to make train and test loss
            # comparable.
            loss = tf.divide(loss, n_trial)

        return (
            loss, tf_z, tf_z_constraint, tf_attention,
            tf_attention_constraint, tf_theta, tf_theta_bounds, tf_obs
        )

    def attention_distance(self, p, q):
        """Distance between attention weights."""
        c = tf.cast(2.0, dtype=FLOAT_X)
        n_dim = tf.cast(tf.shape(p)[0], dtype=FLOAT_X)
        d = tf.divide(
            tf.reduce_sum(tf.abs(p - q)), tf.multiply(c, n_dim)
        )
        return d

    def log_likelihood(self, obs, z=None, theta=None, phi=None):
        """Return the log likelihood of a set of observations.

        Arguments:
            obs: A set of judged similarity trials. The indices
                used must correspond to the rows of z.
            z (optional): The z parameters.
            theta (optional): The theta parameters.
            phi (optional): The phi parameters.

        Returns:
            The total log-likelihood of the observations.

        Notes:
            The arguments z, theta, and phi are assigned in a lazy
                manner. If they have a value of None when they are
                needed, the attribute values of the object will be used.

        """
        cap = 2.2204e-16
        prob_all = self.outcome_probability(
            obs, group_id=obs.group_id, z=z, theta=theta, phi=phi,
            unaltered_only=True)
        prob = ma.maximum(cap, prob_all[:, 0])
        ll = ma.sum(ma.log(prob))
        return ll

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
            phi (optionsl): The phi parameters.
            unaltered_only (optional): Flag the determines whether only
                the unaltered ordering is evaluated.

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
        # ==================================================
        # Create an analogous tensor.
        # shape = (n_config, max_n_outcome, max_n_ref)
        # n_reference_list = docket.config_list['n_reference'].values
        # max_n_reference = np.max(n_reference_list)
        # outcome_tensor = docket.outcome_tensor()
        # ==================================================
        if unaltered_only:
            max_n_outcome = 1

        if z.ndim == 2:
            z = np.expand_dims(z, axis=2)
        n_sample = z.shape[2]

        # Compute similarity between query and references.
        (z_q, z_r) = self._inflate_points(
            docket.stimulus_set, docket.max_n_reference, z)

        sim_qr = self.similarity(
            z_q, z_r, group_id=group_id, theta=theta, phi=phi)

        prob_all = -1 * np.ones((n_trial_all, max_n_outcome, n_sample))
        for i_config in range(n_config):
            config = docket.config_list.iloc[i_config]
            outcome_idx = outcome_idx_list[i_config]
            # outcome_idx = outcome_tensor[
            #     i_config,
            #     0:n_outcome_list[i_config],
            #     0:n_reference_list[i_config]
            # ]
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
                s_qref_perm = sim_qr_config[:, outcome_idx[i_outcome, :], :]
                prob[:, i_outcome, :] = self._ranked_sequence_probabiltiy(
                    s_qref_perm, config['n_select'])
            prob_all[trial_locs, 0:n_outcome, :] = prob
        prob_all = ma.masked_values(prob_all, -1)

        # Correct for numerical inaccuracy.
        if not unaltered_only:
            prob_all = ma.divide(
                prob_all, ma.sum(prob_all, axis=1, keepdims=True))
        if n_sample == 1:
            prob_all = prob_all[:, :, 0]
        return prob_all

    def _inflate_points(self, stimulus_set, n_reference, z):
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

        stimulus_set_temp = copy.copy(stimulus_set) + 1
        z_placeholder = np.zeros((1, n_dim, n_sample), dtype=np.float32)
        z_temp = np.concatenate((z_placeholder, z), axis=0)

        # Inflate query stimuli.
        z_q = z_temp[stimulus_set_temp[:, 0], :, :]
        z_q = np.expand_dims(z_q, axis=2)
        # Inflate reference stimuli.
        z_r = np.empty((n_trial, n_dim, n_reference, n_sample))
        for i_ref in range(n_reference):
            z_r[:, :, i_ref, :] = z_temp[stimulus_set_temp[:, 1+i_ref], :, :]
        return (z_q, z_r)

    def _tf_inflate_points(
            self, stimulus_set, n_reference, z):
        """Inflate stimulus set into embedding points."""
        n_trial = tf.shape(stimulus_set)[0]

        n_dim = tf.shape(z)[1]

        stimulus_set_temp = stimulus_set + 1
        z_placeholder = tf.zeros((1, n_dim), dtype=FLOAT_X)
        z_temp = tf.concat((z_placeholder, z), axis=0)

        # Inflate query stimuli.
        z_q = tf.gather(z_temp, stimulus_set_temp[:, 0])
        z_q = tf.expand_dims(z_q, axis=2)

        # Initialize z_r.
        i_ref = tf.constant(0, dtype=tf.int32)
        z_r = tf.ones([n_trial, n_dim, n_reference], dtype=FLOAT_X)

        def cond_fn(i_ref, z_temp, stimulus_set_temp, z_r):
            return tf.less(i_ref, n_reference)

        def body_fn(i_ref, z_temp, stimulus_set_temp, z_r):
            z_r_new = tf.gather(z_temp, stimulus_set_temp[:, 1+i_ref])
            z_r_new = tf.expand_dims(z_r_new, axis=2)
            z_r = tf.concat(
                [z_r[:, :, :i_ref], z_r_new, z_r[:, :, i_ref+1:]],
                axis=2
            )
            i_ref = i_ref + 1
            return [i_ref, z_temp, stimulus_set_temp, z_r]

        r = tf.while_loop(
            cond_fn, body_fn, [i_ref, z_temp, stimulus_set_temp, z_r],
            [
                i_ref.get_shape(), z_temp.get_shape(),
                stimulus_set_temp.get_shape(),
                tf.TensorShape([None, None, None])
            ]
        )
        z_r = r[3]
        return (z_q, z_r)

    def _ranked_sequence_probabiltiy(self, sim_qr, n_select):
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
            seq_prob = np.multiply(seq_prob, prob)
            # Update denominator in preparation for computing the probability
            # of the previous selection in the sequence.
            if i_selected > 0:
                denom = denom + sim_qr[:, i_selected-1, :]
        return seq_prob

    def _tf_ranked_sequence_probability(self, sim_qr, n_select):
        """Return probability of a ranked selection sequence.

        See: _ranked_sequence_probability

        Arguments:
            sim_qr:
            n_select:

        """
        n_trial = tf.shape(sim_qr)[0]
        # n_trial = tf.cast(n_trial, dtype=tf.int32)

        # Initialize.
        seq_prob = tf.ones([n_trial], dtype=FLOAT_X)
        selected_idx = n_select - 1
        denom = tf.reduce_sum(sim_qr[:, selected_idx:], axis=1)

        def cond_fn(selected_idx, seq_prob, denom):
            return tf.greater_equal(
                selected_idx, tf.constant(0, dtype=tf.int32))

        def body_fn(selected_idx, seq_prob, denom):
            # Compute selection probability.
            prob = tf.divide(sim_qr[:, selected_idx], denom)
            # Update sequence probability.
            seq_prob = np.multiply(seq_prob, prob)
            # Update denominator in preparation for computing the
            # probability of the previous selection in the sequence.
            denom = tf.cond(
                selected_idx > tf.constant(0, dtype=tf.int32),
                lambda: tf.add(denom, sim_qr[:, selected_idx - 1]),
                lambda: denom,
                name='increase_denom'
            )
            return [selected_idx - 1, seq_prob, denom]

        r = tf.while_loop(
            cond_fn, body_fn, [selected_idx, seq_prob, denom]
        )
        return r[1]

    def posterior_samples(
            self, obs, n_final_sample=1000, n_burn=100, thin_step=5,
            z_init=None, verbose=0):
        """Sample from the posterior of the embedding.

        Samples are drawn from the posterior holding theta constant. A
        variant of Eliptical Slice Sampling (Murray & Adams 2010) is
        used to estimate the posterior for the embedding points. Since
        the latent embedding variables are translation and rotation
        invariant, generic sampling will artificailly inflate the
        entropy of the samples. To compensate for this issue, N
        embedding points are selected to serve as anchor points, where
        N is two times the dimensionality of the embedding. Two chains
        are run each using half of the anchor points. The samples from
        the two chains are merged in order to get a posterior estimate
        for all points.

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
        n_final_sample = int(n_final_sample)
        n_total_sample = n_burn + (n_final_sample * thin_step)
        n_stimuli = self.n_stimuli
        n_dim = self.n_dim
        if z_init is None:
            z = copy.copy(self._z["value"])
        else:
            z = z_init

        if verbose > 0:
            print('Sampling from posterior...')
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
            n_components=1, covariance_type='spherical')
        gmm.fit(z)
        mu = gmm.means_[0]
        sigma = gmm.covariances_[0] * np.identity(n_dim)

        # Center embedding to satisfy assumptions of elliptical slice sampling.
        z = z - mu

        n_partition = 2

        # Define log-likelihood for elliptical slice sampler.
        def flat_log_likelihood(z_part, part_idx, z_full, obs):
            # Assemble full z.
            n_stim_part = np.sum(part_idx)
            z_full[part_idx, :] = np.reshape(
                z_part, (n_stim_part, n_dim), order='C')
            # TODO pass in theta and phi.
            return self.log_likelihood(obs, z=z_full)

        # Initalize sampler.
        z_full = copy.copy(z)
        samples = np.empty((n_stimuli, n_dim, n_total_sample))

        # # Sample from prior if there are no observations. TODO
        # if obs.n_trial is 0:
        #     z = np.random.multivariate_normal(
        #         np.zeros((n_dim)), sigma, n_stimuli)
        # else:

        for i_round in range(n_total_sample):
            # Partition stimuli into two groups.
            if np.mod(i_round, 100) == 0:
                part_idx, n_stimuli_part = self._make_partition(
                    n_stimuli, n_partition
                )
                # Create a diagonally tiled covariance matrix in order to slice
                # multiple points simultaneously.
                prior = []
                for i_part in range(n_partition):
                    prior.append(
                        np.linalg.cholesky(
                            self._inflate_sigma(
                                sigma, n_stimuli_part[i_part], n_dim)
                            )
                    )
            for i_part in range(n_partition):
                z_part = z_full[part_idx[i_part], :]
                z_part = z_part.flatten('C')

                (z_part, _) = elliptical_slice(
                    z_part, prior[i_part], flat_log_likelihood,
                    pdf_params=[part_idx[i_part], copy.copy(z), obs])

                z_part = np.reshape(
                    z_part, (n_stimuli_part[i_part], n_dim), order='C')
                # Update z_full.
                z_full[part_idx[i_part], :] = z_part
            samples[:, :, i_round] = z_full

        # Add back in mean.
        mu = np.expand_dims(mu, axis=2)
        samples = samples + mu

        samples_all = samples[:, :, n_burn::thin_step]
        samples_all = samples_all[:, :, 0:n_final_sample]
        samples = dict(z=samples_all)
        return samples

    def _make_partition(self, n_stimuli, n_partition):
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

    def _create_sample_idx(self, n_stimuli, anchor_idx):
        """Create sampling index from anchor index."""
        n_set = 2
        n_anchor_point = anchor_idx.shape[0]
        dmy_idx = np.arange(0, n_stimuli)

        sample_idx = np.empty(
            (n_stimuli - n_anchor_point, n_set), dtype=np.int32)
        good_locs = np.ones((n_stimuli, n_set), dtype=bool)
        for i_set in range(n_set):
            for i_point in range(n_anchor_point):
                good_locs[anchor_idx[i_point, i_set], i_set] = False
            sample_idx[:, i_set] = dmy_idx[good_locs[:, i_set]]
        return sample_idx

    def _inflate_sigma(self, sigma, n_stimuli, n_dim):
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
        in terms of percieved similarity.

        Arguments:
            group_id: Scalar indicating the group_id.

        Returns:
            emb: A group-specific embedding.

        """
        emb = copy.deepcopy(self)
        z = self._z["value"]
        rho = self._theta["rho"]["value"]
        attention_weights = self._phi["w"]["value"][group_id, :]
        z_group = z * np.expand_dims(attention_weights**(1/rho), axis=0)
        emb._z["value"] = z_group
        emb.n_group = 1
        emb.phi["w"]["value"] = np.ones([1, self.n_dim])
        return emb


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
            n_dim (optional): An integer indicating the dimensionalty
                of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.
        """
        PsychologicalEmbedding.__init__(self, n_stimuli, n_dim, n_group)
        self._theta = self._default_theta()
        # self.rho = self._theta["rho"]["value"]
        # self.tau = self._theta["tau"]["value"]
        # self.mu = self._theta["mu"]["value"]

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
            tf_theta['rho'] = tf.compat.v1.get_variable(
                "rho", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(1., 3.)
            )
        if self._theta['tau']["trainable"]:
            tf_theta['tau'] = tf.compat.v1.get_variable(
                "tau", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(1., 2.)
            )
        if self._theta['mu']["trainable"]:
            tf_theta['mu'] = tf.compat.v1.get_variable(
                "mu", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(0.0000000001, .001)
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
        d_qref = tf.pow(tf.abs(z_q - z_r), rho)
        d_qref = tf.multiply(d_qref, tf_attention)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Inverse distance similarity kernel.
        sim_qr = 1 / (tf.pow(d_qref, tau) + mu)
        return sim_qr

    def _similarity(self, z_q, z_r, theta, attention):
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
        d_qref = (np.abs(z_q - z_r))**rho
        d_qref = np.multiply(d_qref, attention)
        d_qref = np.sum(d_qref, axis=1)**(1. / rho)

        # Exponential family similarity kernel.
        sim_qr = 1 / (d_qref**tau + mu)
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
            identication-categorization relationship. Journal of
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
            n_dim (optional): An integer indicating the dimensionalty
                of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.
        """
        PsychologicalEmbedding.__init__(self, n_stimuli, n_dim, n_group)
        self._theta = self._default_theta()
        # self.rho = self._theta["rho"]["value"]
        # self.tau = self._theta["tau"]["value"]
        # self.gamma = self._theta["gamma"]["value"]
        # self.beta = self._theta["beta"]["value"]

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
            tf_theta['rho'] = tf.compat.v1.get_variable(
                "rho", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(1., 3.)
            )
        if self._theta['tau']["trainable"]:
            tf_theta['tau'] = tf.compat.v1.get_variable(
                "tau", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(1., 2.)
            )
        if self._theta['gamma']["trainable"]:
            tf_theta['gamma'] = tf.compat.v1.get_variable(
                "gamma", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(0., .001)
            )
        if self._theta['beta']["trainable"]:
            tf_theta['beta'] = tf.compat.v1.get_variable(
                "beta", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(1., 30.)
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
        d_qref = tf.pow(tf.abs(z_q - z_r), rho)
        d_qref = tf.multiply(d_qref, tf_attention)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Exponential family similarity kernel.
        sim_qr = tf.exp(tf.negative(beta) * tf.pow(d_qref, tau)) + gamma
        return sim_qr

    def _similarity(self, z_q, z_r, theta, attention):
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
        d_qref = (np.abs(z_q - z_r))**rho
        d_qref = np.multiply(d_qref, attention)
        d_qref = np.sum(d_qref, axis=1)**(1. / rho)

        # Exponential family similarity kernel.
        sim_qr = np.exp(np.negative(beta) * d_qref**tau) + gamma
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
            n_dim (optional): An integer indicating the dimensionalty
                of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.
        """
        PsychologicalEmbedding.__init__(self, n_stimuli, n_dim, n_group)
        self._theta = self._default_theta()
        # self.rho = self._theta["rho"]["value"]
        # self.tau = self._theta["tau"]["value"]
        # self.kappa = self._theta["kappa"]["value"]
        # self.alpha = self._theta["alpha"]["value"]

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
            tf_theta['rho'] = tf.compat.v1.get_variable(
                "rho", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(1., 3.)
            )
        if self._theta['tau']["trainable"]:
            tf_theta['tau'] = tf.compat.v1.get_variable(
                "tau", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(1., 2.)
            )
        if self._theta['kappa']["trainable"]:
            tf_theta['kappa'] = tf.compat.v1.get_variable(
                "kappa", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(1., 11.)
            )
        if self._theta['alpha']["trainable"]:
            tf_theta['alpha'] = tf.compat.v1.get_variable(
                "alpha", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(10., 60.)
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
        d_qref = tf.pow(tf.abs(z_q - z_r), rho)
        d_qref = tf.multiply(d_qref, tf_attention)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Heavy-tailed family similarity kernel.
        sim_qr = tf.pow(kappa + tf.pow(d_qref, tau), (tf.negative(alpha)))
        return sim_qr

    def _similarity(self, z_q, z_r, theta, attention):
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
        d_qref = (np.abs(z_q - z_r))**rho
        d_qref = np.multiply(d_qref, attention)
        d_qref = np.sum(d_qref, axis=1)**(1. / rho)

        # Heavy-tailed family similarity kernel.
        sim_qr = (kappa + d_qref**tau)**(np.negative(alpha))
        return sim_qr


class StudentsT(PsychologicalEmbedding):
    """A Student's t family stochastic display embedding algorithm.

    The embedding technique uses the following simialrity kernel:
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
        (mlsp), 2012 IEEE international workshop on (p. 1-6).
        doi:10.1109/MLSP.2012.6349720

    """

    def __init__(self, n_stimuli, n_dim=2, n_group=1):
        """Initialize.

        Arguments:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded.
            n_dim (optional): An integer indicating the dimensionalty
                of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.
        """
        PsychologicalEmbedding.__init__(self, n_stimuli, n_dim, n_group)
        self._theta = self._default_theta()
        # self.rho = self._theta["rho"]["value"]
        # self.tau = self._theta["tau"]["value"]
        # self.alpha = self._theta["alpha"]["value"]

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
            tf_theta['rho'] = tf.compat.v1.get_variable(
                "rho", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(1., 3.)
            )
        if self._theta['tau']["trainable"]:
            tf_theta['tau'] = tf.compat.v1.get_variable(
                "tau", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(1., 2.)
            )
        if self._theta['alpha']["trainable"]:
            min_alpha = np.max((1, self.n_dim - 5.))
            max_alpha = self.n_dim + 5.
            tf_theta['alpha'] = tf.compat.v1.get_variable(
                "alpha", [1], dtype=FLOAT_X,
                initializer=tf.random_uniform_initializer(
                    min_alpha, max_alpha
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
        d_qref = tf.pow(tf.abs(z_q - z_r), rho)
        d_qref = tf.multiply(d_qref, tf_attention)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Student-t family similarity kernel.
        sim_qr = tf.pow(
            1 + (tf.pow(d_qref, tau) / alpha), tf.negative(alpha + 1)/2)
        return sim_qr

    def _similarity(self, z_q, z_r, theta, attention):
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
        d_qref = (np.abs(z_q - z_r))**rho
        d_qref = np.multiply(d_qref, attention)
        d_qref = np.sum(d_qref, axis=1)**(1. / rho)

        # Student-t family similarity kernel.
        sim_qr = (1 + (d_qref**tau / alpha))**(np.negative(alpha + 1)/2)
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
            dtype=tf.float32):
        """Initialize."""
        self.mean = mean
        self.stdev = stdev
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self.dtype = _assert_float_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        """Call."""
        if dtype is None:
            dtype = self.dtype
        scale = tf.pow(
            tf.constant(10., dtype=FLOAT_X),
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
        alpha: A python scalar or a scalar tensor. Alpha parameter(s)
            governing distribution.
        dtype: The data type. Only floating point types are supported.
    """

    def __init__(self, n_dim, concentration=0.0, dtype=tf.float32):
        """Initialize."""
        self.n_dim = n_dim
        self.concentration = concentration
        self.dtype = _assert_float_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        """Call."""
        if dtype is None:
            dtype = self.dtype
        dist = tfp.distributions.Dirichlet(self.concentration)
        return self.n_dim * dist.sample([shape[0]])

    def get_config(self):
        """Return configuration."""
        return {
            "concentration": self.concentration,
            "dtype": self.dtype.name
        }


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
        for name in f['phi'][p_name]:
            embedding._phi[p_name][name] = f['phi'][p_name][name][()]

    f.close()
    return embedding
