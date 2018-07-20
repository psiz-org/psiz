# -*- coding: utf-8 -*-
# Copyright 2018 The PsiZ Authors. All Rights Reserved.
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

Todo:te
    - implement general cost function that includes all scenarios
    - parallelization during fitting (MAYBE)
    - implement warm (currently the same as exact)
    - document how to do warm restarts (warm restarts are sequential
      and can be created by the user using a for loop and fit with
      init_mode='warm')
    - documeent broadcasting in similarity function
    - dcoument verbosity levels
        - modify verbosity so that some basic messages occur every time
    - document meaning of cold, warm, and exact
    - custom exception
        class MyException(Exception):
            pass
    - remove defualts for model initialization?
    - may want to change similarity_matrix to accept attention weight
        vector rather that group_id in order to allow matrices to be
        computed for arbitrary weights.
    - Change from "attention" to more general "tunings" perspective.
        * Encapsulate attention weights in new "tunings" object.
        * Refactor computation of similarity to use tuning object.
        * MAYBE Introduce additional tuning free parameters for each
            stimulus. Control flexibility using rule or L2
            regularization.
"""

import sys
from abc import ABCMeta, abstractmethod
import warnings
import copy
from random import randint

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import StratifiedKFold
from sklearn import mixture
import tensorflow as tf

from psiz.trials import possible_outcomes
from psiz.utils import elliptical_slice


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
        similarity_matrix: Return the similarity matrix characterizing
            the embedding.
        freeze: Freeze the free parameters of an embedding model.
        thaw: Make free parameters trainable.
        probability: Return the probability of the possible outcomes
            for each trial.
        log_likelihood: Return the log-likelihood of a set of
            observations.
        set_log: Adjust the TensorBoard logging behavior.

    Attributes:
        z: A dictionary with the keys 'value', 'trainable'. The key
            'value' contains the actual embedding points. The key
            'trainable' is a boolean flag that determines whether
            the embedding points are inferred during inference.
        theta: Dictionary containing data about the parameter values
            governing the similarity kernel. The dictionary contains
            the variable names as keys at the first level. For each
            variable, there is an additional dictionary containing with
            the keys 'value', 'trainable', and 'bounds'. The key
            'value' indicates the actual value of the parameter. The
            key 'trainable' is a boolean flag indicating whether the
            variable is trainable during inferene. The key 'bounds'
            indicates the bounds of the paramter during inference. The
            bounds are specified using a list of two items where the
            first item indicates the lower bound and the second item
            indicates the upper bound. Use None to indicate no bound.
        attention: The attention weights associated with the embedding
            model. Attention is a dictionary containing the keys
            'value' and 'trainable'. The key 'value' contains the
            actual weights and 'trainable' indicates if the weights are
            trained during inference.

    Notes:
        The methods fit, freeze, thaw, and set_log modify the state of
            the PsychologicalEmbedding object.
        The abstract methods _init_theta,
            _get_similarity_parameters_cold,
            _get_similarity_parameters_warm, and _tf_similarity must be
            implemented by each concrete class.

    """

    __metaclass__ = ABCMeta

    def __init__(self, n_stimuli, n_dim=2, n_group=1):
        """Initialize.

        Args:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded. This must be equal to or
                greater than three.
            n_dim (optional): An integer indicating the dimensionalty
                of the embedding. Must be equal to or greater than one.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group. Must be equal to or greater than one.
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
        self.z = self._init_z()
        self.theta = self._init_theta()
        self.attention = self._init_attention()

        # Default inference settings.
        self.init_scale_list = [.001, .01, .1]
        self.lr = 0.001
        self.max_n_epoch = 5000
        self.patience = 10

        # Default TensorBoard log attributes.
        self.do_log = False
        self.log_dir = '/tmp/tensorflow_logs/embedding/'

        super().__init__()

    def _init_z(self):
        """Return initialized embedding points.

        Initialize random embedding points using a multivariate
            Gaussian.
        """
        mean = np.ones((self.n_dim))
        cov = .1 * np.identity(self.n_dim)
        z = {}
        z['value'] = np.random.multivariate_normal(
            mean, cov, (self.n_stimuli)
        )
        z['trainable'] = True
        return z

    @abstractmethod
    def _init_theta(self):
        """Return dictionary of default theta parameters.

        Returns:
            Dictionary of theta parameters.

        """
        pass

    def _init_attention(self):
        """Return initialize attention.

        Initialize attention weights using uniform distribution.
        """
        attention = {}
        attention['value'] = np.ones(
            (self.n_group, self.n_dim), dtype=np.float32)
        if self.n_group is 1:
            attention['trainable'] = False
        else:
            attention['trainable'] = True
        return attention

    def _check_z(self, z):
        if z.shape[0] != self.n_stimuli:
            raise ValueError(
                "Input 'z' does not have the appropriate shape (number of \
                stimuli).")
        if z.shape[1] != self.n_dim:
            raise ValueError(
                "Input 'z' does not have the appropriate shape \
                (dimensionality).")

    def _check_attention(self, attention):
        if attention.shape[0] != self.n_group:
            raise ValueError(
                "Input 'attention' does not have the appropriate shape \
                (number of groups).")
        if attention.shape[1] != self.n_dim:
            raise ValueError(
                "Input 'attention' does not have the appropriate shape \
                (dimensionality).")
    
    def _set_parameters(self, params):
        """State changing method sets algorithm-specific parameters.

        This method encapsulates the setting of algorithm-specific free
        parameters governing the similarity kernel.

        Args:
            params: A dictionary of algorithm-specific parameter names
                and corresponding values.
        """
        for param_name in params:
            self.theta[param_name]['value'] = params[param_name]

    def _get_similarity_parameters(self, init_mode):
        """Return a dictionary of TensorFlow variables.

        This method encapsulates the creation of algorithm-specific
        free parameters governing the similarity kernel.

        Args:
            init_mode: A string indicating the initialization mode.
                Valid options are 'cold', 'warm', and 'exact'.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        with tf.variable_scope("similarity_params"):
            tf_theta = {}
            if init_mode is 'exact':
                tf_theta = self._get_similarity_parameters_exact()
            elif init_mode is 'warm':
                tf_theta = self._get_similarity_parameters_warm()
            else:
                tf_theta = self._get_similarity_parameters_cold()

            # If a parameter is untrainable, set the parameter value to the
            # value in the class attribute theta.
            for param_name in self.theta:
                if not self.theta[param_name]['trainable']:
                    tf_theta[param_name] = tf.get_variable(
                        param_name, [1], initializer=tf.constant_initializer(
                            self.theta[param_name]['value']
                        ),
                        trainable=False
                    )
        # sim_scope.reuse_variables()
        return tf_theta

    def _get_similarity_parameters_exact(self):
        """Return a dictionary.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        for param_name in self.theta:
            if self.theta[param_name]['trainable']:
                tf_theta[param_name] = tf.get_variable(
                    param_name, [1], initializer=tf.constant_initializer(
                        self.theta[param_name]['value']
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

    @abstractmethod
    def _get_similarity_parameters_warm(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by adding a small amount of noise to
        existing parameter values.

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
        for param_name in self.theta:
            bounds = self.theta[param_name]['bounds']
            if bounds[0] is not None:
                # Add lower bound.
                constraint_list.append(
                    tf_theta[param_name].assign(tf.maximum(
                        bounds[0],
                        tf_theta[param_name])
                    )
                )
            if bounds[1] is not None:
                # Add upper bound.
                constraint_list.append(
                    tf_theta[param_name].assign(tf.minimum(
                        bounds[1],
                        tf_theta[param_name])
                    )
                )
        tf_theta_bounds = tf.group(*constraint_list)
        return tf_theta_bounds

    def freeze(self, freeze_options=None):
        """State changing method specifing which parameters are fixed.

        During inference, you may want to freeze some parameters at
        specific values. To freeze a particular parameter, pass in a
        value for it. If you would like to freeze multiple parameters,
        you can pass in a dictionary containing multiple entries or
        call the freeze method multiple times.

        Args:
            freeze_options (optional): Dictionary of parameter names
                and corresponding values to be frozen (i.e., fixed)
                during inference.
        """
        if freeze_options is not None:
            for param_name in freeze_options:
                if param_name is 'z':
                    z = freeze_options['z'].astype(np.float32)
                    self._check_z(z)
                    self.z['value'] = z
                    self.z['trainable'] = False
                elif param_name is 'attention':
                    attention = freeze_options['attention']
                    self._check_attention(attention)
                    self.attention['value'] = attention
                    self.attention['trainable'] = False
                else:
                    self.theta[param_name]['value'] = \
                        freeze_options[param_name]
                    self.theta[param_name]['trainable'] = False

    def thaw(self, thaw_options=None):
        """State changing method specifying trainable parameters.

        Complement of freeze method. If thaw_options is None, all free
        parameters are set as trainable.

        Args:
            thaw_options (optional): List of parameter names to set as
                trainable during inference. Valid parameter names
                include 'z', 'attention', and the parameters associated
                with the similarity kernel.
        """
        if thaw_options is None:
            self.z['trainable'] = True
            for param_name in self.theta:
                self.theta[param_name]['trainable'] = True
        else:
            for param_name in thaw_options:
                if param_name is 'z':
                    self.z['trainable'] = True
                elif param_name is 'attention':
                    if self.n_group is 1:
                        self.attention['trainable'] = False
                    else:
                        self.attention['trainable'] = True
                else:
                    self.theta[param_name]['trainable'] = True

    def similarity(self, z_q, z_ref, theta=None, attention=None):
        """Return similarity between two lists of points.

        Similarity is determined using the similarity kernel and the
        current similarity parameters.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, n_dim)
            z_ref: A set of embedding points.
                shape = (n_sample, n_dim)
            theta (optional): The parameters governing the similarity
                kernel. If not provided, the theta associated with the
                current object is used.
            attention (optional): The weights allocated to each
                dimension in a weighted minkowski metric. The weights
                should be positive and sum to the dimensionality of the
                weight vector, although this is not enforced.
                shape = (n_sample, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.

        """
        if theta is None:
            theta = self.theta

        if attention is None:
            attention = self.attention['value'][0, :]
            attention = np.expand_dims(attention, axis=0)
        else:
            if len(attention.shape) == 1:
                attention = np.expand_dims(attention, axis=0)

        # Make sure z_q and attention have an appropriate singleton
        # third dimension if z_ref has an array rank of 3.
        if len(z_ref.shape) > 2:
            if len(z_q.shape) == 2:
                z_q = np.expand_dims(z_q, axis=2)
            if len(attention.shape) == 2:
                attention = np.expand_dims(attention, axis=2)

        sim = self._similarity(z_q, z_ref, theta, attention)
        return sim

    @abstractmethod
    def _tf_similarity(self, z_q, z_ref, tf_theta, tf_attention):
        """Similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, n_dim)
            z_ref: A set of embedding points.
                shape = (n_sample, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension in a
                weighted minkowski metric.
                shape = (n_sample, n_dim)
        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_sample,)

        """
        pass

    @abstractmethod
    def _similarity(self, z_q, z_ref, theta, attention):
        """Similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, n_dim)
            z_ref: A set of embedding points.
                shape = (n_sample, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension in a
                weighted minkowski metric.
                shape = (n_sample, n_dim)
        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_sample,)

        """
        pass

    def _get_attention(self, init_mode):
        """Return attention weights of model as TensorFlow variable."""
        if self.attention['trainable']:
            if init_mode is 'exact':
                tf_attention = tf.get_variable(
                    "attention", [self.n_group, self.n_dim],
                    initializer=tf.constant_initializer(
                        self.attention['value']
                    )
                )
            elif init_mode is 'warm':
                tf_attention = tf.get_variable(
                    "attention", [self.n_group, self.n_dim],
                    initializer=tf.constant_initializer(
                        self.attention['value']
                    )
                )
            else:
                alpha = 1. * np.ones((self.n_dim))
                new_attention = (
                    np.random.dirichlet(alpha) * self.n_dim
                )
                tf_attention = tf.get_variable(
                    "attention", [self.n_group, self.n_dim],
                    initializer=tf.constant_initializer(new_attention)
                )
        else:
            tf_attention = tf.get_variable(
                "attention", [self.n_group, self.n_dim],
                initializer=tf.constant_initializer(self.attention['value']),
                trainable=False
            )
        return tf_attention

    def _get_embedding(self, init_mode):
        """Return embedding of model as TensorFlow variable.

        Args:
            init_mode: A string indicating the initialization mode.
                valid options are 'cold', 'warm', and 'exact'.

        Returns:
            TensorFlow variable representing the embedding points.

        """
        # Initialize z with different scales for different restarts
        rand_scale_idx = np.random.randint(0, len(self.init_scale_list))
        scale_value = self.init_scale_list[rand_scale_idx]
        tf_scale_value = tf.constant(scale_value, dtype=tf.float32)

        if self.z['trainable']:
            if init_mode is 'exact':
                tf_z = tf.get_variable(
                    "z", [self.n_stimuli, self.n_dim],
                    initializer=tf.constant_initializer(self.z['value'])
                )
            elif init_mode is 'warm':
                tf_z = tf.get_variable(
                    "z", [self.n_stimuli, self.n_dim],
                    initializer=tf.constant_initializer(self.z['value'])
                )
            else:
                tf_z = tf.get_variable(
                    "z", [self.n_stimuli, self.n_dim],
                    initializer=tf.random_normal_initializer(
                        tf.zeros([self.n_dim]),
                        tf.ones([self.n_dim]) * tf_scale_value
                    )
                )
        else:
            tf_z = tf.get_variable(
                "z", [self.n_stimuli, self.n_dim],
                initializer=tf.constant_initializer(self.z['value']),
                trainable=False
            )
        return tf_z

    def set_log(self, do_log, log_dir=None, delete_prev=False):
        """State changing method that sets TensorBoard logging.

        Args:
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

    def fit(self, obs, n_restart=40, init_mode='cold', verbose=0):
        """Fit the free parameters of the embedding model.

        Args:
            obs: A JudgedTrials object representing the observed data.
            n_restart: An integer specifying the number of restarts to
                use for the inference procedure. Since the embedding
                procedure finds local optima, multiple restarts helps
                find the global optimum.
            init_mode: A string indicating the initialization mode.
                Valid options are 'cold', 'warm', and 'exact'.
            verbose: An integer specifying the verbosity of printed
                output.

        Returns:
            J: The average loss per observation. Loss is defined as the
                negative loglikelihood.

        """
        n_dim = self.n_dim

        #  Infer embedding.
        if (verbose > 0):
            print('Inferring embedding ...')
            print('    Settings:')
            print('    n_observations: ', obs.n_trial)
            print('    n_group: ', len(np.unique(obs.group_id)))
            print('    n_dim: ', n_dim)
            print('    n_restart: ', n_restart)

        # Partition data into train and validation set for early stopping of
        # embedding algorithm.
        skf = StratifiedKFold(n_splits=10)
        (train_idx, test_idx) = list(
            skf.split(obs.stimulus_set, obs.config_idx))[0]

        # Run multiple restarts of embedding algorithm.
        J_all_best = np.inf
        z_best = None
        attention_best = None
        params_best = None

        for i_restart in range(n_restart):
            (J_all, z, attention, params) = self._embed(
                obs, train_idx, test_idx, i_restart, init_mode
            )
            if J_all < J_all_best:
                J_all_best = J_all
                z_best = z
                attention_best = attention
                params_best = params

            if verbose > 1:
                print('Restart ', i_restart)

        self.z['value'] = z_best
        self.attention['value'] = attention_best
        self._set_parameters(params_best)

        return J_all_best

    def evaluate(self, obs):
        """Evaluate observations using the current state of the model.

        Args:
            obs: A JudgedTrials object representing the observed data.

        Returns:
            J: The average loss per observation. Loss is defined as the
                negative loglikelihood.

        """
        (J, _, _, _, _, _, tf_obs) = self._core_model('exact')

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        J_all = sess.run(J, feed_dict=self._bind_obs(tf_obs, obs))

        sess.close()
        tf.reset_default_graph()
        return J_all

    def _embed(self, obs, train_idx, test_idx, i_restart, init_mode):
        """Ebed using a TensorFlow implementation."""
        verbose = 0  # TODO make parameter

        # Partition the observation data.
        obs_train = obs.subset(train_idx)
        obs_val = obs.subset(test_idx)

        (J, tf_z, tf_attention, tf_attention_constraint, tf_theta,
            tf_theta_bounds, tf_obs) = self._core_model(init_mode)

        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(J)

        init = tf.global_variables_initializer()

        with tf.name_scope('summaries'):
            # Create a summary to monitor cost tensor.
            tf.summary.scalar('cost', J)
            # Create a summary of the embedding tensor.
            tf.summary.tensor_summary('z', tf_z)

            # Create a summary to monitor parameteres of similarity kernel.
            with tf.name_scope('similarity'):

                for param_name in tf_theta:
                    param_mean = tf.reduce_mean(tf_theta[param_name])
                    tf.summary.scalar(param_name + '_mean', param_mean)

            # Merge all summaries into a single op.
            merged_summary_op = tf.summary.merge_all()

        sess = tf.Session()
        sess.run(init)

        # op to write logs for TensorBoard
        if self.do_log:
            summary_writer = tf.summary.FileWriter(
                '%s/%s' % (self.log_dir, i_restart),
                graph=tf.get_default_graph()
            )

        J_all_best = np.inf
        J_test_best = np.inf

        last_improvement = 0
        for epoch in range(self.max_n_epoch):
            _, J_train, summary = sess.run(
                [train_op, J, merged_summary_op],
                feed_dict=self._bind_obs(tf_obs, obs_train)
            )

            sess.run(tf_theta_bounds)
            sess.run(tf_attention_constraint)
            J_test = sess.run(J, feed_dict=self._bind_obs(tf_obs, obs_val))
            J_all = sess.run(J, feed_dict=self._bind_obs(tf_obs, obs))

            if J_test < J_test_best:
                J_all_best = J_all
                J_test_best = J_test
                last_improvement = 0
                # TODO handle worst case where there is no improvement from
                # initialization.
                (z_best, attention_best) = sess.run(
                    [tf_z, tf_attention])
                params_best = {}
                for param_name in tf_theta:
                    params_best[param_name] = sess.run(tf_theta[param_name])
            else:
                last_improvement = last_improvement + 1

            if last_improvement > self.patience:
                break

            if not epoch % 10:
                # Write logs at every 10th iteration
                if self.do_log:
                    summary_writer.add_summary(summary, epoch)
            if not epoch % 100:
                if verbose > 2:
                    print("epoch ", epoch, "| J_train: ", J_train,
                          "| J_test: ", J_test, "| J_all: ", J_all)

        sess.close()
        tf.reset_default_graph()

        return (J_all_best, z_best, attention_best, params_best)

    def _bind_obs(self, tf_obs, obs):
        feed_dict = {
            tf_obs['stimulus_set']: obs.stimulus_set,
            tf_obs['n_reference']: obs.n_reference,
            tf_obs['n_selected']: obs.n_selected,
            tf_obs['is_ranked']: obs.is_ranked,
            tf_obs['group_id']: obs.group_id,
            tf_obs['config_idx']: obs.config_idx,
            tf_obs['n_config']: len(obs.config_list.n_outcome.values),
            tf_obs['config_max_n_outcome']: np.max(obs.config_list.n_outcome.values),
            tf_obs['config_n_reference']: obs.config_list.n_reference.values,
            tf_obs['config_n_selected']: obs.config_list.n_selected.values,
            tf_obs['config_is_ranked']: obs.config_list.is_ranked.values,
            tf_obs['config_n_outcome']: obs.config_list.n_outcome.values,
            tf_obs['config_outcome_tensor']: obs.outcome_tensor()
        }
        return feed_dict

    def _project_attention(self, tf_attention_0):
        """Return projection of attention weights."""
        n_dim = tf.shape(tf_attention_0, out_type=tf.float32)[1]
        tf_attention_1 = tf.divide(
            tf.reduce_sum(tf_attention_0, axis=1, keepdims=True), n_dim
        )
        tf_attention_proj = tf.divide(
            tf_attention_0, tf_attention_1
        )

        return tf_attention_proj

    def _cost_2c1(self, tf_z, tf_stimulus_set, tf_theta, tf_attention):
        """Return cost for ordered 2 chooose 1 observations."""
        n_trial = tf.shape(tf_stimulus_set)[0]
        n_trial = tf.cast(n_trial, dtype=tf.float32)

        # Similarity
        Sqa = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 1]),
            tf_theta, tf_attention)
        Sqb = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 2]),
            tf_theta, tf_attention)
        # Probility of behavior
        P = Sqa / (Sqa + Sqb)
        # Cost function
        cap = tf.constant(2.2204e-16)
        J = tf.negative(tf.reduce_sum(tf.log(tf.maximum(P, cap))))
        J = tf.divide(J, n_trial)

        J = tf.cond(
            n_trial > tf.constant(0.), lambda: J, lambda: tf.constant(0.)
            )
        return J

    def _cost_3cN(self, tf_z, tf_stimulus_set, N, tf_theta, tf_attention):
        """Return cost for ordered 6 chooose N observations."""
        n_trial = tf.shape(tf_stimulus_set)[0]
        n_trial = tf.cast(n_trial, dtype=tf.float32)

        # Similarity
        Sqa = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 1]),
            tf_theta, tf_attention)
        Sqb = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 2]),
            tf_theta, tf_attention)
        Sqc = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 3]),
            tf_theta, tf_attention)

        # Probility of behavior
        def f1(): return (
            (Sqa / (Sqa + Sqb + Sqc))
        )

        def f2(): return (
            (Sqa / (Sqa + Sqb + Sqc)) *
            (Sqb / (Sqb + Sqc))
        )

        P = tf.case({
            tf.equal(N, tf.constant(1)): f1,
            tf.equal(N, tf.constant(2)): f2,
            tf.equal(N, tf.constant(3)): f2,
            }, default=f2, exclusive=True)

        # Cost function
        cap = tf.constant(2.2204e-16)
        J = tf.negative(tf.reduce_sum(tf.log(tf.maximum(P, cap))))
        J = tf.divide(J, n_trial)

        J = tf.cond(
            n_trial > tf.constant(0.), lambda: J, lambda: tf.constant(0.))
        return J

    def _cost_4cN(self, tf_z, tf_stimulus_set, N, tf_theta, tf_attention):
        """Return cost for ordered 6 chooose N observations."""
        n_trial = tf.shape(tf_stimulus_set)[0]
        n_trial = tf.cast(n_trial, dtype=tf.float32)

        # Similarity
        Sqa = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 1]),
            tf_theta, tf_attention)
        Sqb = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 2]),
            tf_theta, tf_attention)
        Sqc = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 3]),
            tf_theta, tf_attention)
        Sqd = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 4]),
            tf_theta, tf_attention)

        # Probility of behavior
        def f1(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd))
        )

        def f2(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd)) *
            (Sqb / (Sqb + Sqc + Sqd))
        )

        def f3(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd)) *
            (Sqb / (Sqb + Sqc + Sqd)) *
            (Sqc / (Sqc + Sqd))
        )

        P = tf.case({
            tf.equal(N, tf.constant(1)): f1,
            tf.equal(N, tf.constant(2)): f2,
            tf.equal(N, tf.constant(3)): f3,
            tf.equal(N, tf.constant(4)): f3,
            }, default=f2, exclusive=True)

        # Cost function
        cap = tf.constant(2.2204e-16)
        J = tf.negative(tf.reduce_sum(tf.log(tf.maximum(P, cap))))
        J = tf.divide(J, n_trial)

        J = tf.cond(
            n_trial > tf.constant(0.), lambda: J, lambda: tf.constant(0.))
        return J

    def _cost_5cN(self, tf_z, tf_stimulus_set, N, tf_theta, tf_attention):
        """Return cost for ordered 6 chooose N observations."""
        n_trial = tf.shape(tf_stimulus_set)[0]
        n_trial = tf.cast(n_trial, dtype=tf.float32)

        # Similarity
        Sqa = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 1]),
            tf_theta, tf_attention)
        Sqb = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 2]),
            tf_theta, tf_attention)
        Sqc = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 3]),
            tf_theta, tf_attention)
        Sqd = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 4]),
            tf_theta, tf_attention)
        Sqe = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 5]),
            tf_theta, tf_attention)

        # Probility of behavior
        def f1(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe))
        )

        def f2(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe))
        )

        def f3(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe)) *
            (Sqc / (Sqc + Sqd + Sqe))
        )

        def f4(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe)) *
            (Sqc / (Sqc + Sqd + Sqe)) *
            (Sqd / (Sqd + Sqe))
        )

        P = tf.case({
            tf.equal(N, tf.constant(1)): f1,
            tf.equal(N, tf.constant(2)): f2,
            tf.equal(N, tf.constant(3)): f3,
            tf.equal(N, tf.constant(4)): f4,
            tf.equal(N, tf.constant(5)): f4,
            }, default=f2, exclusive=True)

        # Cost function
        cap = tf.constant(2.2204e-16)
        J = tf.negative(tf.reduce_sum(tf.log(tf.maximum(P, cap))))
        J = tf.divide(J, n_trial)

        J = tf.cond(
            n_trial > tf.constant(0.), lambda: J, lambda: tf.constant(0.))
        return J

    def _cost_6cN(self, tf_z, tf_stimulus_set, N, tf_theta, tf_attention):
        """Return cost for ordered 6 chooose N observations."""
        n_trial = tf.shape(tf_stimulus_set)[0]
        n_trial = tf.cast(n_trial, dtype=tf.float32)

        # Similarity
        Sqa = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 1]),
            tf_theta, tf_attention)
        Sqb = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 2]),
            tf_theta, tf_attention)
        Sqc = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 3]),
            tf_theta, tf_attention)
        Sqd = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 4]),
            tf_theta, tf_attention)
        Sqe = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 5]),
            tf_theta, tf_attention)
        Sqf = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 6]),
            tf_theta, tf_attention)

        # Probility of behavior
        def f1(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf))
        )

        def f2(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf))
        )

        def f3(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf))
        )

        def f4(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf)) *
            (Sqd / (Sqd + Sqe + Sqf))
        )

        def f5(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf)) *
            (Sqd / (Sqd + Sqe + Sqf)) *
            (Sqe / (Sqe + Sqf))
        )

        P = tf.case({
            tf.equal(N, tf.constant(1)): f1,
            tf.equal(N, tf.constant(2)): f2,
            tf.equal(N, tf.constant(3)): f3,
            tf.equal(N, tf.constant(4)): f4,
            tf.equal(N, tf.constant(5)): f5,
            tf.equal(N, tf.constant(6)): f5
            }, default=f2, exclusive=True)

        # Cost function
        cap = tf.constant(2.2204e-16)
        J = tf.negative(tf.reduce_sum(tf.log(tf.maximum(P, cap))))
        J = tf.divide(J, n_trial)

        J = tf.cond(
            n_trial > tf.constant(0.), lambda: J, lambda: tf.constant(0.))
        return J

    def _cost_7cN(self, tf_z, tf_stimulus_set, N, tf_theta, tf_attention):
        """Return cost for ordered 7 chooose N observations."""
        n_trial = tf.shape(tf_stimulus_set)[0]
        n_trial = tf.cast(n_trial, dtype=tf.float32)

        # Similarity
        Sqa = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 1]),
            tf_theta, tf_attention)
        Sqb = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 2]),
            tf_theta, tf_attention)
        Sqc = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 3]),
            tf_theta, tf_attention)
        Sqd = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 4]),
            tf_theta, tf_attention)
        Sqe = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 5]),
            tf_theta, tf_attention)
        Sqf = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 6]),
            tf_theta, tf_attention)
        Sqg = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 7]),
            tf_theta, tf_attention)

        # Probility of behavior
        def f1(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg))
        )

        def f2(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg))
        )

        def f3(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg))
        )

        def f4(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqd / (Sqd + Sqe + Sqf + Sqg))
        )

        def f5(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqd / (Sqd + Sqe + Sqf + Sqg)) *
            (Sqe / (Sqe + Sqf + Sqg))
        )

        def f6(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg)) *
            (Sqd / (Sqd + Sqe + Sqf + Sqg)) *
            (Sqe / (Sqe + Sqf + Sqg)) *
            (Sqf / (Sqf + Sqg))
        )

        P = tf.case({
            tf.equal(N, tf.constant(1)): f1,
            tf.equal(N, tf.constant(2)): f2,
            tf.equal(N, tf.constant(3)): f3,
            tf.equal(N, tf.constant(4)): f4,
            tf.equal(N, tf.constant(5)): f5,
            tf.equal(N, tf.constant(6)): f6,
            tf.equal(N, tf.constant(7)): f6
            }, default=f2, exclusive=True)

        # Cost function
        cap = tf.constant(2.2204e-16)
        J = tf.negative(tf.reduce_sum(tf.log(tf.maximum(P, cap))))
        J = tf.divide(J, n_trial)

        J = tf.cond(
            n_trial > tf.constant(0.), lambda: J, lambda: tf.constant(0.))
        return J

    def _cost_8cN(self, tf_z, tf_stimulus_set, N, tf_theta, tf_attention):
        """Return cost for ordered 8 chooose N observations."""
        n_trial = tf.shape(tf_stimulus_set)[0]
        n_trial = tf.cast(n_trial, dtype=tf.float32)

        # Similarity
        Sqa = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 1]),
            tf_theta, tf_attention)
        Sqb = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 2]),
            tf_theta, tf_attention)
        Sqc = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 3]),
            tf_theta, tf_attention)
        Sqd = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 4]),
            tf_theta, tf_attention)
        Sqe = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 5]),
            tf_theta, tf_attention)
        Sqf = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 6]),
            tf_theta, tf_attention)
        Sqg = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 7]),
            tf_theta, tf_attention)
        Sqh = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 8]),
            tf_theta, tf_attention)

        # Probility of behavior
        def f1(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh))
        )

        def f2(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh))
        )

        def f3(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh))
        )

        def f4(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh))
        )

        def f5(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqe / (Sqe + Sqf + Sqg + Sqh))
        )

        def f6(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqe / (Sqe + Sqf + Sqg + Sqh)) *
            (Sqf / (Sqf + Sqg + Sqh))
        )

        def f7(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqe / (Sqe + Sqf + Sqg + Sqh)) *
            (Sqf / (Sqf + Sqg + Sqh)) *
            (Sqg / (Sqg + Sqh))
        )

        P = tf.case({
            tf.equal(N, tf.constant(1)): f1,
            tf.equal(N, tf.constant(2)): f2,
            tf.equal(N, tf.constant(3)): f3,
            tf.equal(N, tf.constant(4)): f4,
            tf.equal(N, tf.constant(5)): f5,
            tf.equal(N, tf.constant(6)): f6,
            tf.equal(N, tf.constant(7)): f7,
            tf.equal(N, tf.constant(8)): f7
            }, default=f2, exclusive=True)

        # Cost function
        cap = tf.constant(2.2204e-16)
        J = tf.negative(tf.reduce_sum(tf.log(tf.maximum(P, cap))))
        J = tf.divide(J, n_trial)

        J = tf.cond(
            n_trial > tf.constant(0.), lambda: J, lambda: tf.constant(0.))
        return J

    def _prob_2c1(self, tf_z, tf_stimulus_set, tf_theta, tf_attention):
        """Return cost for ordered 2 chooose 1 observations."""
        n_trial = tf.shape(tf_stimulus_set)[0]
        n_trial = tf.cast(n_trial, dtype=tf.float32)

        # Similarity
        Sqa = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 1]),
            tf_theta, tf_attention)
        Sqb = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 2]),
            tf_theta, tf_attention)
        # Probility of behavior
        P = Sqa / (Sqa + Sqb)
        
        return P

    def _prob_8cN(self, tf_z, tf_stimulus_set, N, tf_theta, tf_attention):
        """Return cost for ordered 8 chooose N observations."""
        n_trial = tf.shape(tf_stimulus_set)[0]
        n_trial = tf.cast(n_trial, dtype=tf.float32)

        # Similarity
        Sqa = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 1]),
            tf_theta, tf_attention)
        Sqb = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 2]),
            tf_theta, tf_attention)
        Sqc = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 3]),
            tf_theta, tf_attention)
        Sqd = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 4]),
            tf_theta, tf_attention)
        Sqe = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 5]),
            tf_theta, tf_attention)
        Sqf = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 6]),
            tf_theta, tf_attention)
        Sqg = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 7]),
            tf_theta, tf_attention)
        Sqh = self._tf_similarity(
            tf.gather(tf_z, tf_stimulus_set[:, 0]),
            tf.gather(tf_z, tf_stimulus_set[:, 8]),
            tf_theta, tf_attention)

        # Probility of behavior
        def f1(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh))
        )

        def f2(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh))
        )

        def f3(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh))
        )

        def f4(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh))
        )

        def f5(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqe / (Sqe + Sqf + Sqg + Sqh))
        )

        def f6(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqe / (Sqe + Sqf + Sqg + Sqh)) *
            (Sqf / (Sqf + Sqg + Sqh))
        )

        def f7(): return (
            (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh)) *
            (Sqe / (Sqe + Sqf + Sqg + Sqh)) *
            (Sqf / (Sqf + Sqg + Sqh)) *
            (Sqg / (Sqg + Sqh))
        )

        P = tf.case({
            tf.equal(N, tf.constant(1)): f1,
            tf.equal(N, tf.constant(2)): f2,
            tf.equal(N, tf.constant(3)): f3,
            tf.equal(N, tf.constant(4)): f4,
            tf.equal(N, tf.constant(5)): f5,
            tf.equal(N, tf.constant(6)): f6,
            tf.equal(N, tf.constant(7)): f7,
            tf.equal(N, tf.constant(8)): f7
            }, default=f2, exclusive=True)

        return P

    def _core_model(self, init_mode):
        """Embedding model implemented using TensorFlow."""
        with tf.variable_scope("model"):
            # Similarity function variables
            tf_theta = self._get_similarity_parameters(init_mode)
            tf_theta_bounds = self._get_similarity_constraints(tf_theta)
            tf_attention = self._get_attention(init_mode)
            tf_z = self._get_embedding(init_mode)

            # scope.reuse_variables()
            tf_stimulus_set = tf.placeholder(
                tf.int32, [None, 9], name='stimulus_set'
            )  # TODO remove 9
            tf_n_reference = tf.placeholder(
                tf.int32, [None], name='n_reference')
            tf_n_selected = tf.placeholder(
                tf.int32, [None], name='n_selected')
            tf_is_ranked = tf.placeholder(
                tf.int32, [None], name='is_ranked')
            tf_group_id = tf.placeholder(
                tf.int32, [None], name='group_id')
            tf_config_idx = tf.placeholder(
                tf.int32, [None], name='config_idx')
            # Configuration data.
            tf_n_config = tf.placeholder(
                tf.int32, shape=(), name='n_config'
            )
            tf_config_max_n_outcome = tf.placeholder(
                tf.int32, shape=(), name='max_n_outcome'
            )
            tf_config_n_reference = tf.placeholder(
                tf.int32, name='config_n_reference')
            tf_config_n_selected = tf.placeholder(
                tf.int32, name='config_n_selected')
            tf_config_is_ranked = tf.placeholder(
                tf.int32, name='config_is_ranked')
            tf_config_n_outcome = tf.placeholder(
                tf.int32, name='config_n_outcome')
            tf_outcome_tensor = tf.placeholder(
                tf.int32, name='config_outcome_tensor')
            tf_obs = {
                'stimulus_set': tf_stimulus_set,
                'n_reference': tf_n_reference,
                'n_selected': tf_n_selected,
                'is_ranked': tf_is_ranked,
                'group_id': tf_group_id,
                'config_idx': tf_config_idx,
                'n_config': tf_n_config,
                'config_max_n_outcome': tf_config_max_n_outcome,
                'config_n_reference': tf_config_n_reference,
                'config_n_selected': tf_config_n_selected,
                'config_is_ranked': tf_config_is_ranked,
                'config_n_outcome': tf_config_n_outcome,
                'config_outcome_tensor': tf_outcome_tensor
            }

            # n_trial_all = tf_obs['n_trial']
            # n_config = tf_obs['config_n_reference'].shape[0]
            # n_dim = tf_z.shape[1]

            # # max_n_outcome = tf.reduce_max(tf_obs['config_n_outcome'])
            # max_n_outcome = tf_obs['config_max_n_outcome']
            # # n_trial_all is problematic since the dimension I care
            # # about is None.
            # prob_all = tf.zeros([n_trial_all[0], max_n_outcome[0]])

            n_trial = tf.shape(tf_stimulus_set)[0]
            n_trial = tf.cast(n_trial, dtype=tf.float32)

            # Expand attention weights
            tf_atten_expanded = tf.gather(tf_attention, tf_group_id)

            # max_n_outcome = 1  # TODO
            indices_all = tf.zeros((0), dtype=tf.int64)
            # prob_all = tf.zeros((0, max_n_outcome), dtype=tf.float32)
            prob_all = tf.zeros((0), dtype=tf.float32)

            # ========================================================
            # ========================================================
            # Loop over different display configurations.
            idx_2c1 = tf.squeeze(
                tf.where(tf.equal(tf_n_reference, tf.constant(2)))
            )
            weights_2c1 = tf.gather(tf_atten_expanded, idx_2c1)
            disp_2c1 = tf.gather(tf_stimulus_set, idx_2c1)
            prob_2c1 = self._prob_2c1(
                tf_z, disp_2c1, tf_theta, weights_2c1
            )
            indices_all = tf.concat((indices_all, idx_2c1), axis=0)
            prob_all = tf.concat((prob_all, prob_2c1), axis=0)

            idx_8c2 = tf.squeeze(tf.where(tf.logical_and(
                tf.equal(tf_n_reference, tf.constant(8)),
                tf.equal(tf_n_selected, tf.constant(2))))
            )
            weights_8c2 = tf.gather(tf_atten_expanded, idx_8c2)
            disp_8c2 = tf.gather(tf_stimulus_set, idx_8c2)
            prob_8c2 = self._prob_8cN(
                tf_z, disp_8c2, tf.constant(2), tf_theta, weights_8c2
            )
            indices_all = tf.concat((indices_all, idx_8c2), axis=0)
            prob_all = tf.concat((prob_all, prob_8c2), axis=0)
            # ========================================================
            # ========================================================

            # Outside loop.
            prob_all = tf.gather(prob_all, indices_all)

            # Cost function
            cap = tf.constant(2.2204e-16)
            loss = tf.negative(tf.reduce_sum(tf.log(tf.maximum(prob_all, cap))))
            loss = tf.divide(loss, n_trial)

            # TODO move constraints outside core model?
            tf_attention_constraint = tf_attention.assign(
                self._project_attention(tf_attention))

        return (
            loss, tf_z, tf_attention, tf_attention_constraint, tf_theta,
            tf_theta_bounds, tf_obs
        )

    def log_likelihood(self, obs, z=None, theta=None):
        """Return the log likelihood of a set of observations.

        Args:
            obs: A set of judged similarity trials. The indices
                used must correspond to the rows of z.
            z (optional): A set of embedding points. If no embedding
                points are provided, the points associated with the
                object are used.
            theta (optional): The similarity kernel parameters. If no
                theta parameters are provided, the parameters
                associated with the object are used. TODO

        Returns:
            The total log-likelihood of the observations.

        """
        if z is None:
            z = self.z['value']

        prob_all = self.outcome_probability(
            obs, z, group_id=obs.group_id, unaltered_only=True)
        prob = np.maximum(np.finfo(np.double).tiny, prob_all[:, 0])
        ll = np.sum(np.log(prob))
        return ll

    def outcome_probability(
            self, trials, z=None, theta=None, group_id=None,
            unaltered_only=False):
        """Return probability of each outcome for each trial.

        Args:
            trials: A set of unjudged similarity trials. The indices
                used must correspond to the rows of z.
            z (optional): A set of embedding points. If no embedding
                points are provided, the points associated with the
                object are used.
            theta (optional): The similarity kernel parameters. If no
                theta parameters are provided, the parameters
                associated with the object are used.
            group_id (optional): The group ID for which to compute the
                probabilities.
            unaltered_only (optional): Flag the determines whether only
                the unaltered ordering is evaluated.

        Returns:
            outcome_idx_list: A list with one entry for each display
                configuration. Each entry contains a 2D array where
                each row contains the indices describing one outcome.
            prob_all: The probabilities associated with the different
                outcomes for each unjudged trial. In general, different
                trial configurations will have a different number of
                possible outcomes. Trials with a smaller number of
                possible outcomes are element padded with zeros to
                match the trial with the maximum number of possible
                outcomes.

        Notes:
            The first outcome corresponds to the original order of the
                trial data.

        """
        if z is None:
            z = self.z['value']
        else:
            self._check_z(z)
        # TODO theta

        n_trial_all = trials.n_trial
        n_config = trials.config_list.shape[0]
        n_dim = z.shape[1]

        if group_id is None:
            group_id = np.zeros((trials.n_trial), dtype=np.int)
        attention = self.attention['value'][group_id, :]

        outcome_idx_list = trials.outcome_idx_list
        n_outcome_list = trials.config_list['n_outcome'].values
        max_n_outcome = np.max(n_outcome_list)
        # ==================================================
        # Create an analogous tensor.
        # shape = (n_config, max_n_outcome, max_n_ref)
        n_reference_list = trials.config_list['n_reference'].values
        max_n_reference = np.max(n_reference_list)
        outcome_tensor = trials.outcome_tensor()

        # ==================================================
        if unaltered_only:
            max_n_outcome = 1

        prob_all = np.zeros((n_trial_all, max_n_outcome))
        for i_config in range(n_config):
            config = trials.config_list.iloc[i_config]
            outcome_idx = outcome_idx_list[i_config]
            # outcome_idx = outcome_tensor[
            #     i_config,
            #     0:n_outcome_list[i_config],
            #     0:n_reference_list[i_config]
            # ]
            trial_locs = trials.config_idx == i_config
            n_trial = np.sum(trial_locs)
            n_outcome = n_outcome_list[i_config]
            n_reference = config['n_reference']
            n_selected = config['n_selected']
            n_selected_idx = n_selected - 1
            z_q = z[
                trials.stimulus_set[trial_locs, 0], :
            ]
            z_q = np.expand_dims(z_q, axis=2)
            z_ref = np.empty((n_trial, n_dim, n_reference))
            for i_ref in range(n_reference):
                z_ref[:, :, i_ref] = z[
                    trials.stimulus_set[trial_locs, 1+i_ref], :
                ]

            # Precompute similarity between query and references.
            s_qref = self.similarity(
                z_q, z_ref, attention=attention[trial_locs, :])

            if unaltered_only:
                n_outcome = 1

            # Compute probability of each possible outcome.
            prob = np.ones((n_trial, n_outcome), dtype=np.float64)
            for i_outcome in range(n_outcome):
                s_qref_perm = s_qref[:, outcome_idx[i_outcome, :]]
                prob[:, i_outcome] = self._ranked_sequence_probabiltiy(
                    s_qref_perm, n_selected)
            prob_all[trial_locs, 0:n_outcome] = prob

        # Correct for numerical inaccuracy.
        if not unaltered_only:
            prob_all = np.divide(
                prob_all, np.sum(prob_all, axis=1, keepdims=True))
        return prob_all

    def tf_outcome_probability(self, trials, z_tf, tf_theta):
        """Return probability of outcomes for each trial.

        Args:
            trials: A set of unjudged similarity trials.
            z_tf: TensorFlow tensor representing embedding points.
            tf_theta: Dictionary of Tensorflow tensors representing
                free parameters of similarity kernel.

        Returns:
            outcome_idx_list: A list with one entry for each display
                configuration. Each entry contains a 2D array where
                each row contains the indices describing one outcome.
            prob_all: The probabilities associated with the different
                outcomes for each unjudged trial. In general, different
                trial configurations will have a different number of
                possible outcomes. Trials with a smaller number of
                possible outcomes are element padded with zeros to
                match the trial with the maximum number of possible
                outcomes.

        """
        n_trial_all = trials.n_trial
        dmy_idx = np.arange(n_trial_all)
        n_config = trials.config_list.shape[0]

        attention = self.attention['value'][0, :]  # TODO HACK
        attention = np.expand_dims(attention, axis=0)
        attention = np.expand_dims(attention, axis=2)
        tf_attention = tf.convert_to_tensor(
            attention, dtype=tf.float32
        )

        outcome_idx_list = trials.outcome_idx_list
        n_outcome_list = trials.config_list['n_outcome'].values
        max_n_outcome = np.max(n_outcome_list)

        # prob_all = tf.Variable(tf.zeros((n_trial_all, max_n_outcome)), name='prob_all')
        prob_all = tf.zeros((0, max_n_outcome), dtype=tf.float32)
        indices_all = tf.zeros((0), dtype=tf.int32)
        for i_config in range(n_config):
            config = trials.config_list.iloc[i_config]
            outcome_idx = outcome_idx_list[i_config]
            trial_locs = trials.config_idx == i_config
            n_trial = np.sum(trial_locs)
            n_outcome = n_outcome_list[i_config]
            n_reference = config['n_reference']
            n_selected_idx = config['n_selected'] - 1
            z_q = tf.gather(z_tf, trials.stimulus_set[trial_locs, 0])
            z_q = tf.expand_dims(z_q, axis=2)
            z_ref_list = []
            for i_ref in range(n_reference):
                z_ref_list.append(
                    tf.gather(z_tf, trials.stimulus_set[trial_locs, 1+i_ref])
                )
            z_ref = tf.stack(z_ref_list, axis=2)
            # Precompute similarity between query and references.
            s_qref = self._tf_similarity(
                z_q, z_ref, tf_theta, tf_attention)

            # Compute probability of each possible outcome.
            # prob = tf.Variable(tf.ones((n_outcome, n_trial), dtype=np.float32), name='prob')
            prob = tf.ones((n_outcome, n_trial), dtype=np.float32)
            for i_outcome in range(n_outcome):
                s_qref_perm = tf.gather(
                    s_qref, outcome_idx[i_outcome, :], axis=1)
                # Start with last choice
                denom = tf.reduce_sum(s_qref_perm[:, n_selected_idx:], axis=1)
                # Compute sampling without replacement probability in reverse
                # order for numerical stabiltiy
                for i_selected in range(n_selected_idx, -1, -1):
                    # Grab similarity of selected reference and divide by total
                    # similarity of all available references.
                    updates = tf.divide(s_qref_perm[:, i_selected], denom)
                    split1, split2, split3 = tf.split(
                        prob, [i_outcome, 1, n_outcome-i_outcome-1], 0)
                    split2 = tf.multiply(split2, updates)   # TODO maybe problem with update shape
                    prob = tf.concat((split1, split2, split3), axis=0)
                    # prob = tf.scatter_mul(prob, i_outcome, updates, name='update_prob')
                    # Add similarity for "previous" selection
                    if i_selected > 0:
                        denom = denom + s_qref_perm[:, i_selected-1]
            # Pad absent outcomes before putting in master prob_all.
            prob_zero = tf.zeros((max_n_outcome - n_outcome, n_trial))
            prob = tf.concat((prob, prob_zero), axis=0)
            prob = tf.transpose(prob)
            indices = dmy_idx[trial_locs]
            indices_all = tf.concat((indices_all, indices), axis=0)
            prob_all = tf.concat((prob_all, prob), axis=0)
            # prob_all = tf.scatter_add(prob_all, indices, prob)
        prob_all = tf.gather(prob_all, indices_all)

        # Correct for numerical inaccuracy.
        prob_all = tf.divide(
            prob_all, tf.reduce_sum(prob_all, axis=1, keepdims=True))
        return prob_all

    def _ranked_sequence_probabiltiy(self, s_qref, n_selected):
        """Return probability of a ranked selection sequence.

        Args:
            s_qref: A 2D tensor containing pairwise similarity values.
                Each row contains the similarity between a trial's
                query stimulus and reference stimuli. The tensor is
                arranged such that the first column corresponds to the
                first selection in a sequence, and the last column
                corresponds to the last selection.
            n_selected: Scalar indicating the number of selections made
                by an agent.

        Returns:
            1D Tensor of probabilities.

        Notes:
            For example, given query Q, the probability of selecting
            the references A, B, and C (in that order) would be:

            P(A,B,C) = s_QA/(s_QA + s_QB + s_QC) * s_QB/(s_QB + s_QC)

            where s_QA denotes the similarity between they query and
            reference A.

            The probability is computed by starting with the last
            selection for efficiency and numerical stability. In the
            provided example, this corresponds to first computing the
            probability of selecting B second, given that A was
            selected first.

        """
        n_trial = s_qref.shape[0]
        
        # Initialize.
        prob = np.ones((n_trial), dtype=np.float64)
        n_selected_idx = n_selected - 1
        denom = np.sum(s_qref[:, n_selected_idx:], axis=1)

        for i_selected in range(n_selected_idx, -1, -1):
            # First, divide the similarity of selected reference by the sum of
            # the similarity of all available references (at that point in
            # time). Second, multiply the probability of the selection by the
            # probability of the already computed selections.
            prob = np.multiply(prob, np.divide(s_qref[:, i_selected], denom))
            # Update denominator in preparation for computing the probability
            # of the previous selection in the sequence.
            if i_selected > 0:
                denom = denom + s_qref[:, i_selected-1]
        return prob

    def _tf_sampling_probability(self, s_qref, n_selected):
        """Compute sampling without replacement probability.
        """
        n_selected_idx = n_selected - 1

        denom = tf.reduce_sum(s_qref[:, n_selected_idx:], axis=1)

        return None

    def posterior_samples(self, obs, n_sample=1000, n_burn=1000, thin_step=3):
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

        Args:
            obs: A JudgedTrials object representing the observed data.
            n_sample (optional): The number of samples desired after
                removing the "burn in" samples and applying thinning.
            n_burn (optional): The number of samples to remove from the
                beginning of the sampling sequence.
            thin_step (optional): The interval to use in order to thin
                (i.e., de-correlate) the samples.

        Returns:
            A NumPy array containing the posterior samples. The array
                has shape [n_sample, n_stimuli, n_dim].

        Notes:
            The step_size of the Hamiltonian Monte Carlo procedure is
                determined by the scale of the current embedding.

        References:
            Murray, I., & Adams, R. P. (2010). Slice sampling
            covariance hyperparameters of latent Gaussian models. In
            Advances in Neural Information Processing Systems (pp.
            1732-1740).

        """
        n_sample_set = int(np.ceil(n_sample/2))
        n_stimuli = self.n_stimuli
        n_dim = self.n_dim
        z = copy.copy(self.z['value'])
        n_anchor_point = n_dim

        # Prior
        # p(z_k | Z_negk, theta) ~ N(mu, sigma)
        # Approximate prior of z_k using all embedding points to reduce
        # computational burden.
        gmm = mixture.GaussianMixture(
            n_components=1, covariance_type='spherical')
        gmm.fit(z)
        mu = gmm.means_
        # If spherical and one component, only one element.
        sigma = gmm.covariances_[0] * np.identity(n_dim)

        # Center embedding to satisfy assumptions of elliptical slice sampling.
        z = z - mu

        # Create a diagonally tiled covariance matrix in order to slice
        # multiple points simultaneously.
        prior = np.linalg.cholesky(
            self._inflate_sigma(sigma, n_stimuli - n_anchor_point, n_dim))

        # Select anchor points.
        anchor_idx = self._select_anchor_points(z, n_anchor_point)
        sample_idx = self._create_sample_idx(n_stimuli, anchor_idx)

        # Define log-likelihood for elliptical slice sampler.
        def flat_log_likelihood(z_samp, sample_idx, z_full, obs):
            # Assemble full z.
            n_samp_stimuli = len(sample_idx)
            z_full[sample_idx, :] = np.reshape(
                z_samp, (n_samp_stimuli, n_dim), order='C')
            return self.log_likelihood(obs, z_full)  # TODO theta, group_id?

        combined_samples = [None, None]
        for i_set in range(2):

            # Initalize sampler.
            n_total_sample = n_burn + (n_sample_set * thin_step)
            z_samp = z[sample_idx[:, i_set], :]
            z_samp = z_samp.flatten('C')
            samples = np.empty((n_total_sample, n_stimuli, n_dim))

            # Sample from prior if there are no observations. TODO
            # if obs.n_trial is 0:
            #     z = np.random.multivariate_normal(
            #         np.zeros((n_dim)), sigma, n_stimuli)

            for i_round in range(n_total_sample):
                # print('\r{0}'.format(i_round)) # TODO
                # TODO should pdf params allow for keyword arguments?
                (z_samp, _) = elliptical_slice(
                    z_samp, prior, flat_log_likelihood,
                    pdf_params=[sample_idx[:, i_set], copy.copy(z), obs])
                # Merge z_samp_r and z_anchor
                z_full = copy.copy(z)
                z_samp_r = np.reshape(
                    z_samp, (n_stimuli - n_anchor_point, n_dim), order='C')
                z_full[sample_idx[:, i_set], :] = z_samp_r
                samples[i_round, :, :] = z_full

            # Add back in mean.
            samples = samples + mu
            combined_samples[i_set] = samples[n_burn::thin_step]

        # Replace anchors with samples from other set.
        for i_point in range(n_anchor_point):
            combined_samples[0][:, anchor_idx[i_point, 0], :] = combined_samples[1][:, anchor_idx[i_point, 0], :]
        for i_point in range(n_anchor_point):
            combined_samples[1][:, anchor_idx[i_point, 1], :] = combined_samples[0][:, anchor_idx[i_point, 1], :]

        samples_all = np.vstack((combined_samples[0], combined_samples[1]))
        samples_all = samples_all[0:n_sample]
        return samples_all

    def _select_anchor_points(self, z, n_point):
        """Select anchor points for posterior inference.

        Anchor points are selected so that the sets are
        non-overlapping, all points are far from the center of the
        distribution, all points are far from each other, all points
        within a set are far from one another.

        This code assumes that the incoming points have already been
        centered.

        Args:
            z: The embedding points.
            n_point: The number of points in each set.

        Returns:
            anchor_idx

        """
        n_set = 2
        n_stimuli = z.shape[0]
        n_point_total = n_point * n_set

        # if n_point_total > n_stimuli:
        # TODO issue warning or handle special case

        # First select all points regardless of set assignment.
        # Initialize greedy search.
        candidate_idx = np.random.permutation(n_stimuli)
        best_val = 0

        def obj_func_1(z_candidate):
            # Select those far from mean and far from one another.
            rho = 2.
            n_stim = z_candidate.shape[0]
            loss1 = np.sum(np.abs(z_candidate)**(rho))**(1/rho) / n_stim
            loss2 = np.mean(pdist(z_candidate))
            return (.5 * loss1) + loss2

        n_step = 100
        for _ in range(n_step):
            # Swap a selected point out with a non-selected point.
            selected_idx = randint(0, n_point_total-1)
            nonselected_idx = randint(n_point_total, n_stimuli-1)
            copied_idx = copy.copy(candidate_idx[selected_idx])
            candidate_idx[selected_idx] = candidate_idx[nonselected_idx]
            candidate_idx[nonselected_idx] = copied_idx

            # Evaluate candidate
            z_candidate = z[candidate_idx[0:n_point_total], :]
            candidate_val = obj_func_1(z_candidate)
            if candidate_val > best_val:
                best_val = copy.copy(candidate_val)
                best_idx = copy.copy(candidate_idx)
        best_idx = best_idx[0:n_point_total]

        # Now that points are selected, assign points such that points
        # within a set are far from each other.
        # Initialize greedy search.
        def obj_func_2(z_candidate, n_point):
            # Create two sets such that members within a set are far from
            # one another.
            loss1 = np.mean(pdist(z_candidate[0:n_point]))
            loss2 = np.mean(pdist(z_candidate[n_point:]))
            return loss1 + loss2

        candidate_idx = best_idx
        best_val = 0

        n_step = 2 * n_point**2
        for _ in range(n_step):
            # Swap a selected point out with a non-selected point.
            selected_idx = randint(0, n_point-1)
            nonselected_idx = randint(n_point, (2 * n_point) - 1)
            copied_idx = copy.copy(candidate_idx[selected_idx])
            candidate_idx[selected_idx] = candidate_idx[nonselected_idx]
            candidate_idx[nonselected_idx] = copied_idx

            # Evaluate candidate
            z_candidate = z[candidate_idx, :]
            candidate_val = obj_func_2(z_candidate, n_point)
            if candidate_val > best_val:
                best_val = copy.copy(candidate_val)
                best_idx = copy.copy(candidate_idx)

        anchor_idx = np.zeros((n_point, n_set), dtype=np.int)
        anchor_idx[:, 0] = best_idx[0:n_point]
        anchor_idx[:, 1] = best_idx[n_point:]
        return anchor_idx

    def _create_sample_idx(self, n_stimuli, anchor_idx):
        """Create sampling index from anchor index."""
        n_set = 2
        n_anchor_point = anchor_idx.shape[0]
        dmy_idx = np.arange(0, n_stimuli)

        sample_idx = np.empty(
            (n_stimuli - n_anchor_point, n_set), dtype=np.int)
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

        Args:
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

        # Default inference settings.
        self.lr = 0.003

    def _init_theta(self):
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

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self.theta['rho']['trainable']:
            tf_theta['rho'] = tf.get_variable(
                "rho", [1],
                initializer=tf.random_uniform_initializer(1., 3.)
            )
        if self.theta['tau']['trainable']:
            tf_theta['tau'] = tf.get_variable(
                "tau", [1],
                initializer=tf.random_uniform_initializer(1., 2.)
            )
        if self.theta['gamma']['trainable']:
            tf_theta['gamma'] = tf.get_variable(
                "gamma", [1],
                initializer=tf.random_uniform_initializer(0., .001)
            )
        if self.theta['beta']['trainable']:
            tf_theta['beta'] = tf.get_variable(
                "beta", [1],
                initializer=tf.random_uniform_initializer(1., 30.)
            )
        return tf_theta

    def _get_similarity_parameters_warm(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by adding a small amount of noise to
        existing parameter values.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self.theta['rho']['trainable']:
            tf_theta["rho"] = tf.get_variable(
                "rho", [1], initializer=tf.constant_initializer(
                    self.theta['rho']['value'])
            )
        if self.theta['tau']['trainable']:
            tf_theta["tau"] = tf.get_variable(
                "tau", [1], initializer=tf.constant_initializer(
                    self.theta['tau']['value'])
            )
        if self.theta['gamma']['trainable']:
            tf_theta["gamma"] = tf.get_variable(
                "gamma", [1], initializer=tf.constant_initializer(
                    self.theta['gamma']['value'])
            )
        if self.theta['beta']['trainable']:
            tf_theta["beta"] = tf.get_variable(
                "beta", [1], initializer=tf.constant_initializer(
                    self.theta['beta']['value'])
            )
        return tf_theta

    def _tf_similarity(self, z_q, z_ref, tf_theta, tf_attention):
        """Exponential family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, n_dim)
            z_ref: A set of embedding points.
                shape = (n_sample, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_sample, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_sample,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = tf_theta['rho']
        tau = tf_theta['tau']
        gamma = tf_theta['gamma']
        beta = tf_theta['beta']

        # Weighted Minkowski distance.
        d_qref = tf.pow(tf.abs(z_q - z_ref), rho)
        d_qref = tf.multiply(d_qref, tf_attention)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Exponential family similarity kernel.
        s_qref = tf.exp(tf.negative(beta) * tf.pow(d_qref, tau)) + gamma
        return s_qref

    def _similarity(self, z_q, z_ref, theta, attention):
        """Exponential family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, n_dim)
            z_ref: A set of embedding points.
                shape = (n_sample, n_dim)
            theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_sample, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_sample,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = theta['rho']['value']
        tau = theta['tau']['value']
        gamma = theta['gamma']['value']
        beta = theta['beta']['value']

        # Weighted Minkowski distance.
        d_qref = (np.abs(z_q - z_ref))**rho
        d_qref = np.multiply(d_qref, attention)
        d_qref = np.sum(d_qref, axis=1)**(1. / rho)

        # Exponential family similarity kernel.
        s_qref = np.exp(np.negative(beta) * d_qref**tau) + gamma
        return s_qref


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

        Args:
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

        # Default inference settings.
        self.lr = 0.003

    def _init_theta(self):
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

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self.theta['rho']['trainable']:
            tf_theta['rho'] = tf.get_variable(
                "rho", [1],
                initializer=tf.random_uniform_initializer(1., 3.)
            )
        if self.theta['tau']['trainable']:
            tf_theta['tau'] = tf.get_variable(
                "tau", [1],
                initializer=tf.random_uniform_initializer(1., 2.)
            )
        if self.theta['kappa']['trainable']:
            tf_theta['kappa'] = tf.get_variable(
                "kappa", [1],
                initializer=tf.random_uniform_initializer(1., 11.)
            )
        if self.theta['alpha']['trainable']:
            tf_theta['alpha'] = tf.get_variable(
                "alpha", [1],
                initializer=tf.random_uniform_initializer(10., 60.)
            )
        return tf_theta

    def _get_similarity_parameters_warm(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by adding a small amount of noise to
        existing parameter values.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self.theta['rho']['trainable']:
            tf_theta["rho"] = tf.get_variable(
                "rho", [1],
                initializer=tf.constant_initializer(
                    self.theta['rho']['value']
                )
            )
        if self.theta['tau']['trainable']:
            tf_theta["tau"] = tf.get_variable(
                "tau", [1],
                initializer=tf.constant_initializer(
                    self.theta['tau']['value']
                )
            )
        if self.theta['kappa']['trainable']:
            tf_theta["kappa"] = tf.get_variable(
                "kappa", [1],
                initializer=tf.constant_initializer(
                    self.theta['kappa']['value']
                )
            )
        if self.theta['alpha']['trainable']:
            tf_theta["alpha"] = tf.get_variable(
                "alpha", [1],
                initializer=tf.constant_initializer(
                    self.theta['alpha']['value']
                )
            )
        return tf_theta

    def _tf_similarity(self, z_q, z_ref, tf_theta, tf_attention):
        """Heavy-tailed family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, n_dim)
            z_ref: A set of embedding points.
                shape = (n_sample, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_sample, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_sample,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = tf_theta['rho']
        tau = tf_theta['tau']
        kappa = tf_theta['kappa']
        alpha = tf_theta['alpha']

        # Weighted Minkowski distance.
        d_qref = tf.pow(tf.abs(z_q - z_ref), rho)
        d_qref = tf.multiply(d_qref, tf_attention)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Heavy-tailed family similarity kernel.
        s_qref = tf.pow(kappa + tf.pow(d_qref, tau), (tf.negative(alpha)))
        return s_qref

    def _similarity(self, z_q, z_ref, theta, attention):
        """Heavy-tailed family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, n_dim)
            z_ref: A set of embedding points.
                shape = (n_sample, n_dim)
            theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_sample, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_sample,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = theta['rho']['value']
        tau = theta['tau']['value']
        kappa = theta['kappa']['value']
        alpha = theta['alpha']['value']

        # Weighted Minkowski distance.
        d_qref = (np.abs(z_q - z_ref))**rho
        d_qref = np.multiply(d_qref, tf_attention)
        d_qref = np.sum(d_qref, axis=1)**(1. / rho)

        # Heavy-tailed family similarity kernel.
        s_qref = (kappa + d_qref**tau)**(np.negative(alpha))
        return s_qref


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

        Args:
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

        # Default inference settings.
        self.lr = 0.01

    def _init_theta(self):
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

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self.theta['rho']['trainable']:
            tf_theta['rho'] = tf.get_variable(
                "rho", [1],
                initializer=tf.random_uniform_initializer(1., 3.)
            )
        if self.theta['tau']['trainable']:
            tf_theta['tau'] = tf.get_variable(
                "tau", [1],
                initializer=tf.random_uniform_initializer(1., 2.)
            )
        if self.theta['alpha']['trainable']:
            min_alpha = np.max((1, self.n_dim - 5.))
            max_alpha = self.n_dim + 5.
            tf_theta['alpha'] = tf.get_variable(
                "alpha", [1],
                initializer=tf.random_uniform_initializer(
                    min_alpha, max_alpha
                )
            )
        return tf_theta

    def _get_similarity_parameters_warm(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by adding a small amount of noise to
        existing parameter values.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self.theta['rho']['trainable']:
            tf_theta["rho"] = tf.get_variable(
                "rho", [1],
                initializer=tf.constant_initializer(
                    self.theta['rho']['value']
                )
            )
        if self.theta['tau']['trainable']:
            tf_theta["tau"] = tf.get_variable(
                "tau", [1],
                initializer=tf.constant_initializer(
                    self.theta['tau']['value']
                )
            )
        if self.theta['alpha']['trainable']:
            tf_theta["alpha"] = tf.get_variable(
                "alpha", [1],
                initializer=tf.constant_initializer(
                    self.theta['alpha']['value']
                )
            )
        return tf_theta

    def _tf_similarity(self, z_q, z_ref, tf_theta, tf_attention):
        """Student-t family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, n_dim)
            z_ref: A set of embedding points.
                shape = (n_sample, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_sample, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_sample,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = tf_theta['rho']
        tau = tf_theta['tau']
        alpha = tf_theta['alpha']

        # Weighted Minkowski distance.
        d_qref = tf.pow(tf.abs(z_q - z_ref), rho)
        d_qref = tf.multiply(d_qref, tf_attention)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Student-t family similarity kernel.
        s_qref = tf.pow(
            1 + (tf.pow(d_qref, tau) / alpha), tf.negative(alpha + 1)/2)
        return s_qref

    def _similarity(self, z_q, z_ref, theta, attention):
        """Student-t family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, n_dim)
            z_ref: A set of embedding points.
                shape = (n_sample, n_dim)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            tf_attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_sample, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_sample,)

        """
        # Algorithm-specific parameters governing the similarity kernel.
        rho = theta['rho']['value']
        tau = theta['tau']['value']
        alpha = theta['alpha']['value']

        # Weighted Minkowski distance.
        d_qref = (np.abs(z_q - z_ref))**rho
        d_qref = np.multiply(d_qref, attention)
        d_qref = np.sum(d_qref, axis=1)**(1. / rho)

        # Student-t family similarity kernel.
        s_qref = (1 + (d_qref**tau / alpha))**(np.negative(alpha + 1)/2)
        return s_qref
