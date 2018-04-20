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

Todo:
    - attention weights functionality
    - parallelization
    - implement warm (currently the same as exact)
    - document how to do warm restarts (warm restarts are sequential
      and can be created by the user using a for loop and fit with
      init_mode='warm')
    - dcoument verbosity levels
    - document meaning of cold, warm, and exact
    - document correct implementation of bounds. i.e.,
        [lowerbound, upperbound] use None if doesn't exist
    - One dictionary for z, theta, and attention weights
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class PsychologicalEmbedding(object):
    """Abstract base class for psychological embedding algorithm.

    The embedding procedure jointly infers two components. First, the
    embedding algorithm infers a stimulus representation denoted z.
    Second, the embedding algorithm infers the similarity kernel
    parameters of the concrete class. The set of similarity kernel
    parameters is denoted theta.

    Methods:
        fit: Fit the embedding model using the provided observations.
        evaluate: Evaluate the embedding model using the provided
            observations.
        freeze: Freeze the free parameters of an embedding model.
        thaw: Make free parameters trainable.
        set_log: Adjust the TensorBoard logging behavior.

    Attributes:
        z: The embedding points.
        infer_z: Flag that determines whether the embedding points are
            inferred.
        theta: Dictionary containing the parameter values
            governing the similarity kernel.
        sim_trainable: Dictionary of flags controlling which parameters
            of the similarity kernel are trainable.
        sim_bounds: Dictionary containing a list of the lower and upper
            bounds, in that order, for each parameter in the similarity
            kernel.
        attention_weights: The attention weights associated with the
            embedding model.
        infer_attention_weights: Flag the determines whether attention
            weights are inferred.

    Notes:
        The dictionaries sim_parms, sim_trainable, and sim_bounds
            should use the same set of keys.
        The methods fit, freeze, thaw, and set_log modify the state of
            the PsychologicalEmbedding object.
        The attributes theta, sim_trainable, and sim_bounds must
            be initialized by each concrete class.
        The abstract methods _get_similarity_parameters_cold,
            _get_similarity_parameters_warm, and _similarity
            must be implemented by each concrete class.

    """

    __metaclass__ = ABCMeta

    def __init__(self, n_stimuli, dimensionality=2, n_group=1):
        """Initialize.

        Args:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded.
            dimensionality (optional): An integer indicating the
                dimensionalty of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.
        """
        self.n_stimuli = n_stimuli
        self.n_group = n_group

        # Check arguments are valid.
        if (dimensionality < 1):
            raise ValueError('The provided dimensionality must be an integer \
                greater than 0.')
        if (n_group < 1):
            raise ValueError('The provided n_group must be an integer \
            greater than 0.')

        # Initialize dimension dependent attributes.
        self.dimensionality = dimensionality
        # Initialize random embedding points using multivariate Gaussian.
        mean = np.ones((dimensionality))
        cov = np.identity(dimensionality)
        self.z = np.random.multivariate_normal(mean, cov, (self.n_stimuli))
        # Initialize attentional weights using uniform distribution.
        self.attention_weights = np.ones(
            (self.n_group, dimensionality), dtype=np.float64)

        self.infer_z = True
        if n_group is 1:
            self.infer_attention_weights = False
        else:
            self.infer_attention_weights = True

        # Abstract attributes.
        self.theta = {}
        self.sim_trainable = {}
        self.sim_bounds = {}

        # Embedding scaling factors to draw from.
        self.init_scale_list = [.001, .01, .1]

        # Initialize default TensorBoard log attributes.
        self.do_log = False
        self.log_dir = '/tmp/tensorflow_logs/embedding/'

        # Default inference settings.
        # self.lr = 0.00001
        self.lr = 0.001
        self.max_n_epoch = 5000
        self.patience = 10

        super().__init__()

    def _set_parameters(self, params):
        """State changing method sets algorithm-specific parameters.

        This method encapsulates the setting of algorithm-specific free
        parameters governing the similarity kernel.

        Args:
            params: A dictionary of algorithm-specific parameter names
                and corresponding values.
        """
        for param_name in params:
            self.theta[param_name] = params[param_name]

    def _get_similarity_parameters(self, init_mode):
        """Return a dictionary and TensorFlow operation.

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
                if not self.sim_trainable[param_name]:
                    tf_theta[param_name] = tf.get_variable(
                        param_name, [1], initializer=tf.constant_initializer(
                            self.theta[param_name]
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
            if self.sim_trainable[param_name]:
                tf_theta[param_name] = tf.get_variable(
                    param_name, [1], initializer=tf.constant_initializer(
                        self.theta[param_name]
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

    # TODO
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
            sim_constraints: A TensorFlow operation that imposes
                boundary constraints on the algorithm-specific free
                parameters during inference.

        """
        constraint_list = []
        for param_name in self.sim_bounds:
            bounds = self.sim_bounds[param_name]
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
        sim_constraints = tf.group(*constraint_list)
        return sim_constraints

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
                    self.z = freeze_options['z']
                    self.infer_z = False
                else:
                    self.theta[param_name] = freeze_options[param_name]
                    self.sim_trainable[param_name] = False

    def thaw(self, thaw_options=None):
        """State changing method specifying trainable parameters.

        Complement of freeze method. If thaw_options is None, all free
        parameters are set as trainable.

        Args:
            thaw_options (optional): List of parameter names to set as
                trainable during inference.
        """
        # Unfreeze model parameters based on incoming list.
        if thaw_options is None:
            self.infer_z = True
            for param_name in self.sim_trainable:
                self.sim_trainable[param_name] = True
        else:
            for param_name in thaw_options:
                if param_name is 'z':
                    self.infer_z = True
                else:
                    self.sim_trainable[param_name] = True

    def similarity(self, z_q, z_ref, attention_weights=None, group_id=None):
        """Return similarity between two lists of points.

        Similarity is determined using the similarity kernel and the
        current similarity parameters.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, dimensionality)
            z_ref: A set of embedding points.
                shape = (n_sample, dimensionality)
            attention_weights: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_sample, dimensionality)
        """
        # TODO handle group_id is not None, attention_weights is not None
        tf_z_q = tf.constant(z_q, dtype=tf.float32)
        tf_z_ref = tf.constant(z_ref, dtype=tf.float32)
        tf_attention_weights = tf.convert_to_tensor(
            attention_weights, dtype=tf.float32
        )

        tf_theta = {}
        for param_name in self.theta:
            tf_theta[param_name] = \
                tf.constant(self.theta[param_name], dtype=tf.float32)

        sim_op = self._similarity(
            tf_z_q, tf_z_ref, tf_theta, tf_attention_weights
        )
        sess = tf.Session()
        sim = sess.run(sim_op)
        sess.close()
        tf.reset_default_graph()
        return sim

    @abstractmethod
    def _similarity(self, z_q, z_ref, tf_theta, attention_weights):
        """Similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, dimensionality)
            z_ref: A set of embedding points.
                shape = (n_sample, dimensionality)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            attention_weights: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_sample, dimensionality)
        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_sample,)

        """
        pass

    def _get_attention_weights(self, init_mode):
        """Return attention weights of model as TensorFlow variable."""
        if self.infer_attention_weights:
            if init_mode is 'exact':
                attention_weights = tf.get_variable(
                    "attention_weights", [self.n_group, self.dimensionality],
                    initializer=tf.constant_initializer(self.attention_weights)
                )
            elif init_mode is 'warm':
                attention_weights = tf.get_variable(
                    "attention_weights", [self.n_group, self.dimensionality],
                    initializer=tf.constant_initializer(self.attention_weights)
                )
            else:
                alpha = 1. * np.ones((self.dimensionality))
                new_attention_weights = (
                    np.random.dirichlet(alpha) * self.dimensionality
                )
                attention_weights = tf.get_variable(
                    "attention_weights", [self.n_group, self.dimensionality],
                    initializer=tf.constant_initializer(new_attention_weights)
                )
        else:
            attention_weights = tf.get_variable(
                "attention_weights", [self.n_group, self.dimensionality],
                initializer=tf.constant_initializer(self.attention_weights),
                trainable=False
            )
        return attention_weights

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

        if self.infer_z:
            if init_mode is 'exact':
                z = tf.get_variable(
                    "z", [self.n_stimuli, self.dimensionality],
                    initializer=tf.constant_initializer(self.z)
                )
            elif init_mode is 'warm':
                z = tf.get_variable(
                    "z", [self.n_stimuli, self.dimensionality],
                    initializer=tf.constant_initializer(self.z)
                )
            else:
                z = tf.get_variable(
                    "z", [self.n_stimuli, self.dimensionality],
                    initializer=tf.random_normal_initializer(
                        tf.zeros([self.dimensionality]),
                        tf.ones([self.dimensionality]) * tf_scale_value
                    )
                )
        else:
            z = tf.get_variable(
                "z", [self.n_stimuli, self.dimensionality],
                initializer=tf.constant_initializer(self.z),
                trainable=False
            )
        return z

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
        dimensionality = self.dimensionality

        #  Infer embedding.
        if (verbose > 0):
            print('Inferring embedding ...')
            print('    Settings:')
            print('    n_observations: ', obs.n_trial)
            print('    n_group: ', len(np.unique(obs.group_id)))
            print('    dimensionality: ', dimensionality)
            print('    n_restart: ', n_restart)

        # Partition data into train and validation set for early stopping of
        # embedding algorithm.
        skf = StratifiedKFold(n_splits=10)
        (train_idx, test_idx) = list(
            skf.split(obs.stimulus_set, obs.configuration_id))[0]

        # Run multiple restarts of embedding algorithm.
        J_all_best = np.inf
        z_best = None
        attention_weights_best = None
        params_best = None

        for i_restart in range(n_restart):
            (J_all, z, attention_weights, params) = self._embed(
                obs, train_idx, test_idx, i_restart, init_mode
            )
            if J_all < J_all_best:
                J_all_best = J_all
                z_best = z
                attention_weights_best = attention_weights
                params_best = params

            if verbose > 1:
                print('Restart ', i_restart)

        self.z = z_best
        self.attention_weights = attention_weights_best
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
        (J, _, _, _, _, tf_stimulus_set, tf_n_reference, tf_n_selected,
            tf_is_ranked, tf_group_id) = self._core_model('exact')

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        J_all = sess.run(J, feed_dict={
                tf_stimulus_set: obs.stimulus_set,
                tf_n_reference: obs.n_reference,
                tf_n_selected: obs.n_selected,
                tf_is_ranked: obs.is_ranked,
                tf_group_id: obs.group_id})

        sess.close()
        tf.reset_default_graph()
        return J_all

    def _embed(self, obs, train_idx, test_idx, i_restart, init_mode):
        """Ebed using a TensorFlow implementation."""
        verbose = 0  # TODO make parameter

        # Partition the observation data.
        obs_train = obs.subset(train_idx)
        obs_val = obs.subset(test_idx)

        (J, z, attention_weights, tf_theta, constraint, tf_stimulus_set,
            tf_n_reference, tf_n_selected, tf_is_ranked,
            tf_group_id) = self._core_model(init_mode)

        # train_op = tf.train.GradientDescentOptimizer(
        #   learning_rate=self.lr
        # ).minimize(J)
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(J)

        init = tf.global_variables_initializer()

        with tf.name_scope('summaries'):
            # Create a summary to monitor cost tensor.
            tf.summary.scalar('cost', J)
            # Create a summary of the embedding tensor.
            tf.summary.tensor_summary('z', z)
            # Create a summary of the attention weights.
            # tf.summary.tensor_summary('attention_weights', attention_weights)
            # tf.summary.scalar('attention_00', attention_weights[0,0])

            # Create a summary to monitor parameteres of similarity kernel.
            with tf.name_scope('similarity'):

                for param_name in tf_theta:
                    param_mean = tf.reduce_mean(tf_theta[param_name])
                    tf.summary.scalar(param_name + '_mean', param_mean)
                    # tf.summary.histogram(param_name + '_hist',
                    #   tf_theta[param_name])

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
                [train_op, J, merged_summary_op], feed_dict={
                    tf_stimulus_set: obs_train.stimulus_set,
                    tf_n_reference: obs_train.n_reference,
                    tf_n_selected: obs_train.n_selected,
                    tf_is_ranked: obs_train.is_ranked,
                    tf_group_id: obs_train.group_id
                    }
                )

            sess.run(constraint)
            J_test = sess.run(J, feed_dict={
                tf_stimulus_set: obs_val.stimulus_set,
                tf_n_reference: obs_val.n_reference,
                tf_n_selected: obs_val.n_selected,
                tf_is_ranked: obs_val.is_ranked,
                tf_group_id: obs_val.group_id})

            J_all = sess.run(J, feed_dict={
                tf_stimulus_set: obs.stimulus_set,
                tf_n_reference: obs.n_reference,
                tf_n_selected: obs.n_selected,
                tf_is_ranked: obs.is_ranked,
                tf_group_id: obs.group_id})

            if J_test < J_test_best:
                J_all_best = J_all
                J_test_best = J_test
                last_improvement = 0
                # TODO handle worst case where there is no improvement from
                # initialization.
                (z_best, attention_weights_best) = sess.run(
                    [z, attention_weights])
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

        return (J_all_best, z_best, attention_weights_best, params_best)

    def _project_attention_weights(self, attention_weights_0):
        """Return projection of attention weights."""
        n_dim = tf.shape(attention_weights_0, out_type=tf.float64)[1]
        attention_weights_1 = tf.divide(
            tf.reduce_sum(attention_weights_0, axis=1, keepdims=True), n_dim
            )
        attention_weights_proj = tf.divide(
            attention_weights_0, attention_weights_1
            )

        return attention_weights_proj

    def _cost_2c1(self, z, triplets, theta, attention_weights):
        """Return cost for ordered 2 chooose 1 observations."""
        n_disp = tf.shape(triplets)[0]
        n_disp = tf.cast(n_disp, dtype=tf.float32)

        # Similarity
        Sqa = self._similarity(
            tf.gather(z, triplets[:, 0]), tf.gather(z, triplets[:, 1]),
            theta, attention_weights)
        Sqb = self._similarity(
            tf.gather(z, triplets[:, 0]), tf.gather(z, triplets[:, 2]),
            theta, attention_weights)
        # Probility of behavior
        P = Sqa / (Sqa + Sqb)
        # Cost function
        cap = tf.constant(2.2204e-16)
        J = tf.negative(tf.reduce_sum(tf.log(tf.maximum(P, cap))))
        J = tf.divide(J, n_disp)

        J = tf.cond(
            n_disp > tf.constant(0.), lambda: J, lambda: tf.constant(0.)
            )
        return J

    def _cost_8cN(self, z, nines, N, theta, attention_weights):
        """Return cost for ordered 8 chooose N observations."""
        n_disp = tf.shape(nines)[0]
        n_disp = tf.cast(n_disp, dtype=tf.float32)

        # Similarity
        Sqa = self._similarity(
            tf.gather(z, nines[:, 0]), tf.gather(z, nines[:, 1]), theta,
            attention_weights)
        Sqb = self._similarity(
            tf.gather(z, nines[:, 0]), tf.gather(z, nines[:, 2]), theta,
            attention_weights)
        Sqc = self._similarity(
            tf.gather(z, nines[:, 0]), tf.gather(z, nines[:, 3]), theta,
            attention_weights)
        Sqd = self._similarity(
            tf.gather(z, nines[:, 0]), tf.gather(z, nines[:, 4]), theta,
            attention_weights)
        Sqe = self._similarity(
            tf.gather(z, nines[:, 0]), tf.gather(z, nines[:, 5]), theta,
            attention_weights)
        Sqf = self._similarity(
            tf.gather(z, nines[:, 0]), tf.gather(z, nines[:, 6]), theta,
            attention_weights)
        Sqg = self._similarity(
            tf.gather(z, nines[:, 0]), tf.gather(z, nines[:, 7]), theta,
            attention_weights)
        Sqh = self._similarity(
            tf.gather(z, nines[:, 0]), tf.gather(z, nines[:, 8]), theta,
            attention_weights)

        # Probility of behavior
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
        J = tf.divide(J, n_disp)

        J = tf.cond(
            n_disp > tf.constant(0.), lambda: J, lambda: tf.constant(0.))
        return J

    def _core_model(self, init_mode):
        """Embedding model implemented using TensorFlow."""
        with tf.variable_scope("model"):
            # Similarity function variables
            tf_theta = self._get_similarity_parameters(init_mode)
            sim_constraints = self._get_similarity_constraints(tf_theta)
            attention_weights = self._get_attention_weights(init_mode)
            z = self._get_embedding(init_mode)

            # scope.reuse_variables()

            tf_stimulus_set = tf.placeholder(
                tf.int32, [None, 9], name='stimulus_set'
            )
            tf_n_reference = tf.placeholder(tf.int32, name='n_reference')
            tf_n_selected = tf.placeholder(tf.int32, name='n_selected')
            tf_is_ranked = tf.placeholder(tf.int32, name='is_ranked')
            tf_group_id = tf.placeholder(tf.int32, name='group_id')

            # Get indices of different display configurations
            idx_8c2 = tf.squeeze(tf.where(tf.logical_and(
                tf.equal(tf_n_reference, tf.constant(8)),
                tf.equal(tf_n_selected, tf.constant(2))))
            )
            idx_2c1 = tf.squeeze(
                tf.where(tf.equal(tf_n_reference, tf.constant(2)))
            )

            # Get appropriate observations.
            disp_8c2 = tf.gather(tf_stimulus_set, idx_8c2)

            disp_2c1 = tf.gather(tf_stimulus_set, idx_2c1)
            disp_2c1 = disp_2c1[:, 0:3]

            # Expand attention weights
            group_idx_2c1 = tf.gather(tf_group_id, idx_2c1)
            group_idx_2c1 = tf.reshape(
                group_idx_2c1, [tf.shape(group_idx_2c1)[0], 1]
            )
            weights_2c1 = tf.gather_nd(attention_weights, group_idx_2c1)
            group_idx_8c2 = tf.gather(tf_group_id, idx_8c2)
            group_idx_8c2 = tf.reshape(
                group_idx_8c2, [tf.shape(group_idx_8c2)[0], 1]
            )
            weights_8c2 = tf.gather_nd(attention_weights, group_idx_8c2)

            # Cost function
            J = (
                self._cost_2c1(z, disp_2c1, tf_theta, weights_2c1) +
                self._cost_8cN(
                    z, disp_8c2, tf.constant(2), tf_theta, weights_8c2
                )
            )

            # TODO constraint_weights
            # constraint_weights = attention_weights.assign(
            #   self._project_attention_weights(attention_weights))

        return (
            J, z, attention_weights, tf_theta, sim_constraints,
            tf_stimulus_set, tf_n_reference, tf_n_selected, tf_is_ranked,
            tf_group_id)


class Exponential(PsychologicalEmbedding):
    """An exponential family stochastic display embedding algorithm.

    This embedding technique uses the following similarity kernel:
        s(x,y) = exp(-beta .* norm(x - y, rho).^tau) + gamma,
    where x and y are n-dimensional vectors. The similarity function
    has four free parameters: rho, tau, gamma, and beta. The
    exponential family is obtained by integrating across various
    psychological theores [1,2,3,4].

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

    def __init__(self, n_stimuli, dimensionality=2, n_group=1):
        """Initialize.

        Args:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded.
            dimensionality (optional): An integer indicating the
                dimensionalty of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.
        """
        PsychologicalEmbedding.__init__(
            self, n_stimuli, dimensionality, n_group
            )

        # Default parameter settings.
        self.theta = dict(rho=2., tau=1., gamma=0., beta=10.)
        self.sim_trainable = dict(rho=True, tau=True, gamma=True, beta=True)
        self.sim_bounds = dict(
            rho=[1., None], tau=[1., None], gamma=[0., None], beta=[1., None])

        # self.theta = dict(
        #     rho=dict(value=2., trainable=True, bounds=[1., None]),
        #     tau=dict(value=1., trainable=True, bounds=[1., None]),
        #     gamma=dict(value=0., trainable=True, bounds=[0., None]),
        #     beta=dict(value=10., trainable=True, bounds=[1., None]),
        # )

        # Default inference settings.
        self.lr = 0.003
        # self.max_n_epoch = 2000
        # self.patience = 10

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self.sim_trainable['rho']:
            tf_theta['rho'] = tf.get_variable(
                "rho", [1],
                initializer=tf.random_uniform_initializer(1., 3.)
            )
        if self.sim_trainable['tau']:
            tf_theta['tau'] = tf.get_variable(
                "tau", [1],
                initializer=tf.random_uniform_initializer(1., 2.)
            )
        if self.sim_trainable['gamma']:    
            tf_theta['gamma'] = tf.get_variable(
                "gamma", [1],
                initializer=tf.random_uniform_initializer(0., .001)
            )
        if self.sim_trainable['beta']:
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
        if self.sim_trainable['rho']:
            tf_theta["rho"] = tf.get_variable(
                "rho", [1], initializer=tf.constant_initializer(
                    self.theta['rho'])
            )
        if self.sim_trainable['tau']:
            tf_theta["tau"] = tf.get_variable(
                "tau", [1], initializer=tf.constant_initializer(
                    self.theta['tau'])
            )
        if self.sim_trainable['gamma']:
            tf_theta["gamma"] = tf.get_variable(
                "gamma", [1], initializer=tf.constant_initializer(
                    self.theta['gamma'])
            )
        if self.sim_trainable['beta']:
            tf_theta["beta"] = tf.get_variable(
                "beta", [1], initializer=tf.constant_initializer(
                    self.theta['beta'])
            )
        return tf_theta

    def _similarity(self, z_q, z_ref, tf_theta, attention_weights):
        """Exponential family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, dimensionality)
            z_ref: A set of embedding points.
                shape = (n_sample, dimensionality)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            attention_weights: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_sample, dimensionality)

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
        d_qref = tf.multiply(d_qref, attention_weights)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Exponential family similarity kernel.
        s_qref = tf.exp(tf.negative(beta) * tf.pow(d_qref, tau)) + gamma
        return s_qref


class HeavyTailed(PsychologicalEmbedding):
    """A heavy-tailed family stochastic display embedding algorithm.

    This embedding technique uses the following similarity kernel:
        s(x,y) = (kappa + (norm(x-y, rho).^tau)).^(-alpha),
    where x and y are n-dimensional vectors. The similarity function
    has four free parameters: rho, tau, kappa, and alpha. The
    heavy-tailed family is a generalization of the Student-t family.
    """

    def __init__(self, n_stimuli, dimensionality=2, n_group=1):
        """Initialize.

        Args:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded.
            dimensionality (optional): An integer indicating the
                dimensionalty of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.
        """
        PsychologicalEmbedding.__init__(
            self, n_stimuli, dimensionality, n_group)

        # Default parameter settings.
        self.theta = dict(rho=2., tau=1., kappa=2., alpha=30.)
        self.sim_trainable = dict(rho=True, tau=True, kappa=True, alpha=True)
        self.sim_bounds = dict(
            rho=[1., None], tau=[1., None], kappa=[0., None], alpha=[0., None])
        # self.theta = dict(
        #     rho=dict(value=2., trainable=True, bounds=[1., None]),
        #     tau=dict(value=1., trainable=True, bounds=[1., None]),
        #     kappa=dict(value=2., trainable=True, bounds=[0., None]),
        #     alpha=dict(value=30., trainable=True, bounds=[0., None]),
        # )

        # Default inference settings.
        self.lr = 0.003
        # self.max_n_epoch = 2000
        # self.patience = 10

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            tf_theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self.sim_trainable['rho']:
            tf_theta['rho'] = tf.get_variable(
                "rho", [1],
                initializer=tf.random_uniform_initializer(1., 3.)
            )
        if self.sim_trainable['tau']:
            tf_theta['tau'] = tf.get_variable(
                "tau", [1],
                initializer=tf.random_uniform_initializer(1., 2.)
            )
        if self.sim_trainable['kappa']:
            tf_theta['kappa'] = tf.get_variable(
                "kappa", [1],
                initializer=tf.random_uniform_initializer(1., 11.)
            )
        if self.sim_trainable['alpha']:
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
        if self.sim_trainable['rho']:
            tf_theta["rho"] = tf.get_variable(
                "rho", [1],
                initializer=tf.constant_initializer(self.theta['rho'])
            )
        if self.sim_trainable['tau']:
            tf_theta["tau"] = tf.get_variable(
                "tau", [1],
                initializer=tf.constant_initializer(self.theta['tau'])
            )
        if self.sim_trainable['kappa']:
            tf_theta["kappa"] = tf.get_variable(
                "kappa", [1],
                initializer=tf.constant_initializer(self.theta['kappa'])
            )
        if self.sim_trainable['alpha']:
            tf_theta["alpha"] = tf.get_variable(
                "alpha", [1],
                initializer=tf.constant_initializer(self.theta['alpha'])
            )
        return tf_theta

    def _similarity(self, z_q, z_ref, tf_theta, attention_weights):
        """Heavy-tailed family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, dimensionality)
            z_ref: A set of embedding points.
                shape = (n_sample, dimensionality)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            attention_weights: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_sample, dimensionality)

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
        d_qref = tf.multiply(d_qref, attention_weights)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Heavy-tailed family similarity kernel.
        s_qref = tf.pow(kappa + tf.pow(d_qref, tau), (tf.negative(alpha)))
        return s_qref


class StudentsT(PsychologicalEmbedding):
    """A Student's t family stochastic display embedding algorithm.

    The embedding technique uses the following simialrity kernel:
        s(x,y) = (1 + (((norm(x-y, rho)^tau)/alpha))^(-(alpha + 1)/2),
    where x and y are n-dimensional vectors. The similarity kernel has
    three free parameters: rho, tau, and alpha. The original Student-t
    kernel proposed by van der Maaten [1] uses the parameter settings
    rho=2, tau=2, and alpha=dimensionality-1. By default, this
    embedding algorithm will only infer the embedding and not the
    free parameters associated with the similarity kernel. This
    behavior can bechanged by setting the inference flags (e.g.,
    infer_alpha = True).

    References:
    [1] van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic
        triplet embedding. In Machine learning for signal processing
        (mlsp), 2012 IEEE international workshop on (p. 1-6).
        doi:10.1109/MLSP.2012.6349720

    """

    def __init__(self, n_stimuli, dimensionality=2, n_group=1):
        """Initialize.

        Args:
            n_stimuli: An integer indicating the total number of unique
                stimuli that will be embedded.
            dimensionality (optional): An integer indicating the
                dimensionalty of the embedding.
            n_group (optional): An integer indicating the number of
                different population groups in the embedding. A
                separate set of attention weights will be inferred for
                each group.
        """
        PsychologicalEmbedding.__init__(
            self, n_stimuli, dimensionality, n_group)

        # Default parameter settings.
        self.theta = dict(rho=2., tau=2., alpha=dimensionality - 1.)
        self.sim_trainable = dict(rho=False, tau=False, alpha=False)
        self.sim_bounds = dict(
            rho=[1., None], tau=[1., None], alpha=[0.000001, None])
        # self.theta = dict(
        #     rho=dict(value=2., trainable=False, bounds=[1., None]),
        #     tau=dict(value=1., trainable=False, bounds=[1., None]),
        #     alpha=dict(
        #         value=(dimensionality - 1.),
        #         trainable=False,
        #         bounds=[0.000001, None]
        #     ),
        # )

        # Default inference settings.
        # self.lr = 0.003
        self.lr = 0.01
        # self.max_n_epoch = 2000
        # self.patience = 10
        # self.init_scale_list = [.001, .01, .1]

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            theta: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_theta = {}
        if self.sim_trainable['rho']:
            tf_theta['rho'] = tf.get_variable(
                "rho", [1],
                initializer=tf.random_uniform_initializer(1., 3.)
            )
        if self.sim_trainable['tau']:
            tf_theta['tau'] = tf.get_variable(
                "tau", [1],
                initializer=tf.random_uniform_initializer(1., 2.)
            )
        if self.sim_trainable['alpha']:
            min_alpha = np.max((1, self.dimensionality - 5.))
            max_alpha = self.dimensionality + 5.
            tf_theta['alpha'] = tf.get_variable(
                "alpha", [1],
                initializer=tf.random_uniform_initializer(min_alpha, max_alpha)
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
        tf_theta["rho"] = tf.get_variable(
            "rho", [1],
            initializer=tf.constant_initializer(self.theta['rho'])
        )
        tf_theta["tau"] = tf.get_variable(
            "tau", [1],
            initializer=tf.constant_initializer(self.theta['tau'])
        )
        tf_theta["alpha"] = tf.get_variable(
            "alpha", [1],
            initializer=tf.constant_initializer(self.theta['alpha'])
        )
        return tf_theta

    def _similarity(self, z_q, z_ref, tf_theta, attention_weights):
        """Student-t family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, dimensionality)
            z_ref: A set of embedding points.
                shape = (n_sample, dimensionality)
            tf_theta: A dictionary of algorithm-specific parameters
                governing the similarity kernel.
            attention_weights: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_sample, dimensionality)

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
        d_qref = tf.multiply(d_qref, attention_weights)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Student-t family similarity kernel.
        s_qref = tf.pow(
            1 + (tf.pow(d_qref, tau) / alpha), tf.negative(alpha + 1)/2)
        return s_qref
