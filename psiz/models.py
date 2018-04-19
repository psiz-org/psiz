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
    - add similarity method for public API
    - rework resuse
    - change default behavior of fit to warm restart and add reinit method
    - add cold and warm option?
    - parallelization and/or warm restarts
    - docs should be clear regarding verbosity levels

    - use exact values (frozen, evaluate)
    - warm values (warm fit)
    - cold start (new inits for each restart)

    - document correct implementation of bounds. i.e., 
        [lowerbound, upperbound] use None if doesn't exist
    - do cold and warm need trainable, or is always trure?

"""

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class PsychologicalEmbedding(object):
    """Abstract base class for psychological embedding algorithm.

    The embedding procedure jointly infers two components. First, the
    embedding algorithm infers a stimulus representation denoted Z.
    Second, the embedding algoirthm infers the similarity kernel
    parameters of the concrete class.

    Methods:
        fit: Fit the embedding model using the provided observations.
        evaluate: Evaluate the embedding model using the provided
            observations.
        freeze: Freeze the free parameters of an embedding model.
        thaw: Make free parameters trainable.
        reuse: Reuse the free parameters of an embedding model when
            fitting.
        set_log: Adjust the TensorBoard logging behavior.

    Attributes:
        Z: The embedding points.
        infer_Z: Flag that determines whether the embedding points are
            inferred.
        sim_params: Dictionary containing the parameter values
            governing the similarity kernel.
        sim_trainable: Dictionary of flags controlling which parameters
            of the similarity kernel are trainable.
        attention_weights: The attention weights associated with the
            embedding model.
        infer_attention_weights: Flag the determines whether attention
            weights are inferred.

    Notes:
        The methods fit, freeze, thaw, and reuse modify the state of 
            the object.
        The attributes sim_params, sim_trainable, and sim_bounds must 
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
        self.Z = np.random.multivariate_normal(mean, cov, (self.n_stimuli))
        # Initialize attentional weights using uniform distribution.
        self.attention_weights = np.ones(
            (self.n_group, dimensionality), dtype=np.float64)

        self.infer_Z = True
        if n_group is 1:
            self.infer_attention_weights = False
        else:
            self.infer_attention_weights = True

        # Abstract attributes.
        self.sim_params = {}
        self.sim_trainable = {}
        self.sim_bounds = {}

        # Embedding scaling factors to draw from.
        self.init_scale_list = [.001, .01, .1]

        # Initialize default reuse attributes.
        self.do_reuse = False
        self.init_scale = 0

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
            self.sim_params[param_name] = params[param_name]
   
    def _get_similarity_parameters(self, init='cold'):
        """Return a dictionary and TensorFlow operation.

        This method encapsulates the creation of algorithm-specific
        free parameters governing the similarity kernel.

        Args:
            init: A string indicating the initialization mode. The
                options are 'cold', 'warm', and 'exact'. The default
                mode is 'cold'.

        Returns:
            sim_params: A dictionary of algorithm-specific TensorFlow
                variables.
            sim_constraints: A TensorFlow operation that imposes
                boundary constraints on the algorithm-specific free
                parameters during inference.

        """
        with tf.variable_scope("similarity_params"):
            tf_sim_params = {}
            if init is 'exact':
                tf_sim_params = self._get_similarity_parameters_exact()
            elif init is 'warm':
                tf_sim_params = self._get_similarity_parameters_warm()
            else:
                tf_sim_params = self._get_similarity_parameters_cold()

            # Override previous initialization if a parameter is untrainable.
            # Sets the parameter value to the current class attribute.
            for param_name in self.sim_params:
                if not self.sim_trainable[param_name]:
                    tf_sim_params[param_name] = tf.get_variable(
                        param_name, [1], initializer=tf.constant_initializer(
                            self.sim_params[param_name]),
                        trainable=False)

        # TensorFlow operation to enforce free parameter constraints.
        sim_constraints = self._get_similarity_constraints(tf_sim_params)

        # TODO remove
        #     # ===== OLD =====
        #     if self.do_reuse:
        #         rho = tf.get_variable(
        #             "rho", [1], initializer=tf.constant_initializer(
        #                 self.sim_params['rho']),
        #             trainable=True)
        #         tau = tf.get_variable(
        #             "tau", [1], initializer=tf.constant_initializer(
        #                 self.sim_params['tau']),
        #             trainable=True)
        #         gamma = tf.get_variable(
        #             "gamma", [1], initializer=tf.constant_initializer(
        #                 self.sim_params['gamma']),
        #             trainable=True)
        #         beta = tf.get_variable(
        #             "beta", [1], initializer=tf.constant_initializer(
        #                 self.sim_params['beta']),
        #             trainable=True)
        #     else:
        #         if self.sim_trainable['rho']:
        #             rho = tf.get_variable(
        #                 "rho", [1],
        #                 initializer=tf.random_uniform_initializer(1., 3.)
        #                 )
        #         else:
        #             rho = tf.get_variable(
        #                 "rho", [1], initializer=tf.constant_initializer(
        #                     self.sim_params['rho']),
        #                 trainable=False)
        #         if self.sim_trainable['tau']:
        #             tau = tf.get_variable(
        #                 "tau", [1],
        #                 initializer=tf.random_uniform_initializer(1., 2.)
        #                 )
        #         else:
        #             tau = tf.get_variable(
        #                 "tau", [1], initializer=tf.constant_initializer(
        #                     self.sim_params['tau']),
        #                 trainable=False)
        #         if self.sim_trainable['gamma']:
        #             gamma = tf.get_variable(
        #                 "gamma", [1],
        #                 initializer=tf.random_uniform_initializer(0., .001))
        #         else:
        #             gamma = tf.get_variable(
        #                 "gamma", [1], initializer=tf.constant_initializer(
        #                     self.sim_params['gamma']),
        #                 trainable=False)
        #         if self.sim_trainable['beta']:
        #             beta = tf.get_variable(
        #                 "beta", [1],
        #                 initializer=tf.random_uniform_initializer(1., 30.))
        #         else:
        #             beta = tf.get_variable(
        #                 "beta", [1], initializer=tf.constant_initializer(
        #                     self.sim_params['beta']),
        #                 trainable=False)
        # sim_params = {'rho': rho, 'tau': tau, 'gamma': gamma, 'beta': beta}
        return (tf_sim_params, sim_constraints)

    def _get_similarity_parameters_exact(self):
        """Return a dictionary.

        Returns:
            sim_params: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_sim_params = {}
        for param_name in self.sim_params:
            tf_sim_params[param_name] = tf.get_variable(
                param_name, [1], initializer=tf.constant_initializer(
                    self.sim_params[param_name]),
                trainable=True)
        return tf_sim_params

    # TODO
    @abstractmethod
    def _get_similarity_parameters_warm(self):
        """Return a dictionary and TensorFlow operation.

        This method encapsulates the creation of algorithm-specific
        free parameters governing the similarity kernel.

        Returns:
            sim_params: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        pass

    # TODO
    @abstractmethod
    def _get_similarity_parameters_cold(self):
        """Return a dictionary and TensorFlow operation.

        This method encapsulates the creation of algorithm-specific
        free parameters governing the similarity kernel.

        Returns:
            sim_params: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        pass

    def _get_similarity_constraints(self, tf_sim_params):
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
                    tf_sim_params[param_name].assign(tf.maximum(
                        bounds[0],
                        tf_sim_params[param_name])
                    )
                )
            if bounds[1] is not None:
                # Add upper bound.
                constraint_list.append(
                    tf_sim_params[param_name].assign(tf.minimum(
                        bounds[1],
                        tf_sim_params[param_name])
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
                if param_name is 'Z':
                    self.Z = freeze_options['Z']
                    self.infer_Z = False
                else:
                    self.sim_params[param_name] = freeze_options[param_name]
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
            self.infer_Z = True
            for param_name in self.sim_trainable:
                self.sim_trainable[param_name] = True
        else:
            for param_name in thaw_options:
                if param_name is 'Z':
                    self.infer_Z = True
                else:
                    self.sim_trainable[param_name] = True

    @abstractmethod
    def _similarity(self, z_q, z_ref, sim_params, attention_weights):
        """Similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, dimensionality)
            z_ref: A set of embedding points.
                shape = (n_sample, dimensionality)
            sim_params: A dictionary of algorithm-specific parameters
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

    def _get_attention_weights(self):
        """Return attention weights of model as TensorFlow variable."""
        # Attention variable
        if self.do_reuse:
            attention_weights = tf.get_variable(
                "attention_weights", [self.n_group, self.dimensionality],
                initializer=tf.constant_initializer(self.attention_weights),
                trainable=True
                )
        else:
            if self.infer_attention_weights:
                alpha = 1. * np.ones((self.dimensionality))
                new_attention_weights = (np.random.dirichlet(alpha) *
                                         self.dimensionality)
                attention_weights = tf.get_variable(
                    "attention_weights", [self.n_group, self.dimensionality],
                    initializer=tf.constant_initializer(new_attention_weights)
                    )
            else:
                attention_weights = tf.get_variable(
                    "attention_weights", [self.n_group, self.dimensionality],
                    initializer=tf.constant_initializer(
                        self.attention_weights),
                    trainable=False)
        return attention_weights

    def _get_embedding(self):
        """Return embedding of model as TensorFlow variable."""
        # Embedding variable
        # Iniitalize Z with different scales for different restarts
        rand_scale_idx = np.random.randint(0, len(self.init_scale_list))
        scale_value = self.init_scale_list[rand_scale_idx]
        tf_scale_value = tf.constant(scale_value, dtype=tf.float32)

        if self.do_reuse:
            Z = tf.get_variable(
                "Z", [self.n_stimuli, self.dimensionality],
                initializer=tf.constant_initializer(self.Z), trainable=True
                )
        else:
            if self.infer_Z:
                Z = tf.get_variable(
                    "Z", [self.n_stimuli, self.dimensionality],
                    initializer=tf.random_normal_initializer(
                        tf.zeros([self.dimensionality]),
                        tf.ones([self.dimensionality]) * tf_scale_value)
                    )
            else:
                Z = tf.get_variable(
                    "Z", [self.n_stimuli, self.dimensionality],
                    initializer=tf.constant_initializer(self.Z),
                    trainable=False)
        return Z

    def reuse(self, do_reuse, init_scale=0):
        """State changing method that sets reuse of embedding.

        Args:
            do_reuse: Boolean that indicates whether the current
                embedding should be used for initialization during
                inference.
            init_scale: A scalar value indicating to went extent the
                previous embedding points should be reused. For
                example, a value of 0.05 would add uniform noise to all
                the points in the embedding such that each embedding
                point was randomly jittered up to 5% on each dimension
                relative to the overall size of the embedding. The
                value can be between [0,1].
        """
        self.do_reuse = do_reuse
        self.init_scale = init_scale

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

    def fit(self, obs, n_restart=40, verbose=0):
        """Fit the free parameters of the embedding model.

        Args:
            obs: A JudgedTrials object representing the observed data.
            n_restart: An integer specifying the number of restarts to
                use for the inference procedure. Since the embedding
                procedure finds local optima, multiple restarts helps
                find the global optimum.
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
        Z_best = None
        attention_weights_best = None
        params_best = None

        for i_restart in range(n_restart):
            (J_all, Z, attention_weights, params) = self._embed(
                obs, train_idx, test_idx, i_restart
                )
            if J_all < J_all_best:
                J_all_best = J_all
                Z_best = Z
                attention_weights_best = attention_weights
                params_best = params

            if verbose > 1:
                print('Restart ', i_restart)

        self.Z = Z_best
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
        # Is this really necessary?
        old_do_reuse = self.do_reuse
        old_init_scale = self.init_scale

        self.do_reuse = True
        self.init_scale = 0.
        # TODO does calling core model in this context grab the appropriate
        # model parameters or does it re-initialize them?

        (J, _, _, _, _, tf_stimulus_set, tf_n_reference, tf_n_selected,
            tf_is_ranked, tf_group_id) = self._core_model()

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

        self.do_reuse = old_do_reuse
        self.init_scale = old_init_scale

        return J_all

    def _embed(self, obs, train_idx, test_idx, i_restart):
        """Ebed using a TensorFlow implementation."""
        verbose = 0  # TODO make parameter

        # Partition the observation data.
        obs_train = obs.subset(train_idx)
        obs_val = obs.subset(test_idx)

        (J, Z, attention_weights, sim_params, constraint, tf_stimulus_set,
            tf_n_reference, tf_n_selected, tf_is_ranked,
            tf_group_id) = self._core_model()

        # train_op = tf.train.GradientDescentOptimizer(
        #   learning_rate=self.lr
        # ).minimize(J)
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(J)

        init = tf.global_variables_initializer()

        with tf.name_scope('summaries'):
            # Create a summary to monitor cost tensor.
            tf.summary.scalar('cost', J)
            # Create a summary of the embedding tensor.
            tf.summary.tensor_summary('Z', Z)
            # Create a summary of the attention weights.
            # tf.summary.tensor_summary('attention_weights', attention_weights)
            # tf.summary.scalar('attention_00', attention_weights[0,0])

            # Create a summary to monitor parameteres of similarity kernel.
            with tf.name_scope('similarity'):

                for param_name in sim_params:
                    param_mean = tf.reduce_mean(sim_params[param_name])
                    tf.summary.scalar(param_name + '_mean', param_mean)
                    # tf.summary.histogram(param_name + '_hist',
                    #   sim_params[param_name])

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
                (Z_best, attention_weights_best) = sess.run(
                    [Z, attention_weights])
                params_best = {}
                for param_name in sim_params:
                    params_best[param_name] = sess.run(sim_params[param_name])
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

        return (J_all_best, Z_best, attention_weights_best, params_best)

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

    def _cost_2c1(self, Z, triplets, sim_params, attention_weights):
        """Return cost for ordered 2 chooose 1 observations."""
        n_disp = tf.shape(triplets)[0]
        n_disp = tf.cast(n_disp, dtype=tf.float32)

        # Similarity
        Sqa = self._similarity(
            tf.gather(Z, triplets[:, 0]), tf.gather(Z, triplets[:, 1]),
            sim_params, attention_weights)
        Sqb = self._similarity(
            tf.gather(Z, triplets[:, 0]), tf.gather(Z, triplets[:, 2]),
            sim_params, attention_weights)
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

    def _cost_8cN(self, Z, nines, N, sim_params, attention_weights):
        """Return cost for ordered 8 chooose N observations."""
        n_disp = tf.shape(nines)[0]
        n_disp = tf.cast(n_disp, dtype=tf.float32)

        # Similarity
        Sqa = self._similarity(
            tf.gather(Z, nines[:, 0]), tf.gather(Z, nines[:, 1]), sim_params,
            attention_weights)
        Sqb = self._similarity(
            tf.gather(Z, nines[:, 0]), tf.gather(Z, nines[:, 2]), sim_params,
            attention_weights)
        Sqc = self._similarity(
            tf.gather(Z, nines[:, 0]), tf.gather(Z, nines[:, 3]), sim_params,
            attention_weights)
        Sqd = self._similarity(
            tf.gather(Z, nines[:, 0]), tf.gather(Z, nines[:, 4]), sim_params,
            attention_weights)
        Sqe = self._similarity(
            tf.gather(Z, nines[:, 0]), tf.gather(Z, nines[:, 5]), sim_params,
            attention_weights)
        Sqf = self._similarity(
            tf.gather(Z, nines[:, 0]), tf.gather(Z, nines[:, 6]), sim_params,
            attention_weights)
        Sqg = self._similarity(
            tf.gather(Z, nines[:, 0]), tf.gather(Z, nines[:, 7]), sim_params,
            attention_weights)
        Sqh = self._similarity(
            tf.gather(Z, nines[:, 0]), tf.gather(Z, nines[:, 8]), sim_params,
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

    def _core_model(self, init='cold'):
        """Embedding model implemented using TensorFlow."""
        with tf.variable_scope("model"):
            # Similarity function variables
            (sim_params, sim_constraints) = \
                self._get_similarity_parameters(init)
            attention_weights = self._get_attention_weights()
            Z = self._get_embedding()

            # scope.reuse_variables() TODO

            tf_stimulus_set = tf.placeholder(
                tf.int32, [None, 9], name='stimulus_set')
            tf_n_reference = tf.placeholder(tf.int32, name='n_reference')
            tf_n_selected = tf.placeholder(tf.int32, name='n_selected')
            tf_is_ranked = tf.placeholder(tf.int32, name='is_ranked')
            tf_group_id = tf.placeholder(tf.int32, name='group_id')

            # Get indices of different display configurations
            idx_8c2 = tf.squeeze(tf.where(tf.logical_and(
                tf.equal(tf_n_reference, tf.constant(8)),
                tf.equal(tf_n_selected, tf.constant(2)))))
            idx_2c1 = tf.squeeze(
                tf.where(tf.equal(tf_n_reference, tf.constant(2))))

            # Get appropriate observations.
            disp_8c2 = tf.gather(tf_stimulus_set, idx_8c2)

            disp_2c1 = tf.gather(tf_stimulus_set, idx_2c1)
            disp_2c1 = disp_2c1[:, 0:3]

            # Expand attention weights
            group_idx_2c1 = tf.gather(tf_group_id, idx_2c1)
            group_idx_2c1 = tf.reshape(
                group_idx_2c1, [tf.shape(group_idx_2c1)[0], 1])
            weights_2c1 = tf.gather_nd(attention_weights, group_idx_2c1)
            group_idx_8c2 = tf.gather(tf_group_id, idx_8c2)
            group_idx_8c2 = tf.reshape(
                group_idx_8c2, [tf.shape(group_idx_8c2)[0], 1])
            weights_8c2 = tf.gather_nd(attention_weights, group_idx_8c2)

            # Cost function
            J = (
                self._cost_2c1(Z, disp_2c1, sim_params, weights_2c1) +
                self._cost_8cN(
                    Z, disp_8c2, tf.constant(2), sim_params, weights_8c2
                    )
                )

            # TODO constraint_weights
            # constraint_weights = attention_weights.assign(
            #   self._project_attention_weights(attention_weights))

        return (
            J, Z, attention_weights, sim_params, sim_constraints,
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
        self.sim_params = dict(rho=2., tau=1., gamma=0., beta=10.)

        # Default inference settings.
        self.sim_trainable = dict(rho=True, tau=True, gamma=True, beta=True)
        self.sim_bounds = dict(
            rho=[1., None], tau=[1., None], gamma=[0., None], beta=[1., None])
        self.lr = 0.003
        # self.max_n_epoch = 2000
        # self.patience = 10

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            sim_params: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_sim_params = {}
        tf_sim_params['rho'] = tf.get_variable(
            "rho", [1],
            initializer=tf.random_uniform_initializer(1., 3.))
        tf_sim_params['tau'] = tf.get_variable(
            "tau", [1],
            initializer=tf.random_uniform_initializer(1., 2.))
        tf_sim_params['gamma'] = tf.get_variable(
            "gamma", [1],
            initializer=tf.random_uniform_initializer(0., .001))
        tf_sim_params['beta'] = tf.get_variable(
            "beta", [1],
            initializer=tf.random_uniform_initializer(1., 30.))
        return tf_sim_params

    def _get_similarity_parameters_warm(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by adding a small amount of noise to
        existing parameter values.
        
        Returns:
            sim_params: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_sim_params = {}
        tf_sim_params["rho"] = tf.get_variable(
            "rho", [1], initializer=tf.constant_initializer(
                self.sim_params['rho']),
            trainable=self.sim_trainable['rho'])
        tf_sim_params["tau"] = tf.get_variable(
            "tau", [1], initializer=tf.constant_initializer(
                self.sim_params['tau']),
            trainable=self.sim_trainable['tau'])
        tf_sim_params["gamma"] = tf.get_variable(
            "gamma", [1], initializer=tf.constant_initializer(
                self.sim_params['gamma']),
            trainable=self.sim_trainable['gamma'])
        tf_sim_params["beta"] = tf.get_variable(
            "beta", [1], initializer=tf.constant_initializer(
                self.sim_params['beta']),
            trainable=self.sim_trainable['beta'])
        return tf_sim_params

    def _similarity(self, z_q, z_ref, sim_params, attention_weights):
        """Exponential family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, dimensionality)
            z_ref: A set of embedding points.
                shape = (n_sample, dimensionality)
            sim_params: A dictionary of algorithm-specific parameters
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
        rho = sim_params['rho']
        tau = sim_params['tau']
        gamma = sim_params['gamma']
        beta = sim_params['beta']

        # Weighted Minkowski distance.
        d_qref = tf.pow(tf.abs(z_q - z_ref), rho)
        d_qref = tf.multiply(d_qref, attention_weights)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Exponential family similarity kernel.
        s_qref = tf.exp(tf.negative(beta) * tf.pow(d_qref, tau) + gamma)
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
        self.sim_params = dict(rho=2., tau=1., kappa=2., alpha=30.)

        # Default inference settings.
        self.sim_trainable = dict(rho=True, tau=True, kappa=True, alpha=True)
        self.sim_bounds = dict(
            rho=[1., None], tau=[1., None], kappa=[0., None], alpha=[0., None])
        self.lr = 0.003
        # self.max_n_epoch = 2000
        # self.patience = 10

    def _get_similarity_parameters_cold(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by sampling from a relatively large
        set.

        Returns:
            sim_params: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_sim_params = {}
        tf_sim_params['rho'] = tf.get_variable(
            "rho", [1],
            initializer=tf.random_uniform_initializer(1., 3.))
        tf_sim_params['tau'] = tf.get_variable(
            "tau", [1],
            initializer=tf.random_uniform_initializer(1., 2.))
        tf_sim_params['kappa'] = tf.get_variable(
            "kappa", [1],
            initializer=tf.random_uniform_initializer(1., 11.))
        tf_sim_params['alpha'] = tf.get_variable(
            "alpha", [1],
            initializer=tf.random_uniform_initializer(10., 60.))
        return tf_sim_params

    def _get_similarity_parameters_warm(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by adding a small amount of noise to
        existing parameter values.
        
        Returns:
            sim_params: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_sim_params = {}
        tf_sim_params["rho"] = tf.get_variable(
            "rho", [1], initializer=tf.constant_initializer(
                self.sim_params['rho']),
            trainable=self.sim_trainable['rho'])
        tf_sim_params["tau"] = tf.get_variable(
            "tau", [1], initializer=tf.constant_initializer(
                self.sim_params['tau']),
            trainable=self.sim_trainable['tau'])
        tf_sim_params["kappa"] = tf.get_variable(
            "kappa", [1], initializer=tf.constant_initializer(
                self.sim_params['kappa']),
            trainable=self.sim_trainable['kappa'])
        tf_sim_params["alpha"] = tf.get_variable(
            "alpha", [1], initializer=tf.constant_initializer(
                self.sim_params['alpha']),
            trainable=self.sim_trainable['alpha'])
        return tf_sim_params

    def _similarity(self, z_q, z_ref, sim_params, attention_weights):
        """Heavy-tailed family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, dimensionality)
            z_ref: A set of embedding points.
                shape = (n_sample, dimensionality)
            sim_params: A dictionary of algorithm-specific parameters
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
        rho = sim_params['rho']
        tau = sim_params['tau']
        kappa = sim_params['kappa']
        alpha = sim_params['alpha']

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
        self.sim_params = dict(rho=2., tau=2., alpha=dimensionality - 1.)

        # Default inference settings.
        self.sim_trainable = dict(rho=False, tau=False, alpha=False)
        self.sim_bounds = dict(
            rho=[1., None], tau=[1., None], alpha=[0.000001, None])
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
            sim_params: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_sim_params = {}
        tf_sim_params['rho'] = tf.get_variable(
            "rho", [1],
            initializer=tf.random_uniform_initializer(1., 3.))
        tf_sim_params['tau'] = tf.get_variable(
            "tau", [1],
            initializer=tf.random_uniform_initializer(1., 2.))
        min_alpha = np.max((1, self.dimensionality - 5.))
        max_alpha = self.dimensionality + 5.
        tf_sim_params['alpha'] = tf.get_variable(
            "alpha", [1],
            initializer=tf.random_uniform_initializer(min_alpha, max_alpha)
        )
        return tf_sim_params

    def _get_similarity_parameters_warm(self):
        """Return a dictionary of TensorFlow parameters.

        Parameters are initialized by adding a small amount of noise to
        existing parameter values.
        
        Returns:
            sim_params: A dictionary of algorithm-specific TensorFlow
                variables.

        """
        tf_sim_params = {}
        tf_sim_params["rho"] = tf.get_variable(
            "rho", [1], initializer=tf.constant_initializer(
                self.sim_params['rho']),
            trainable=self.sim_trainable['rho'])
        tf_sim_params["tau"] = tf.get_variable(
            "tau", [1], initializer=tf.constant_initializer(
                self.sim_params['tau']),
            trainable=self.sim_trainable['tau'])
        tf_sim_params["alpha"] = tf.get_variable(
            "alpha", [1], initializer=tf.constant_initializer(
                self.sim_params['alpha']),
            trainable=self.sim_trainable['alpha'])
        return tf_sim_params

    def _similarity(self, z_q, z_ref, sim_params, attention_weights):
        """Student-t family similarity kernel.

        Args:
            z_q: A set of embedding points.
                shape = (n_sample, dimensionality)
            z_ref: A set of embedding points.
                shape = (n_sample, dimensionality)
            sim_params: A dictionary of algorithm-specific parameters
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
        rho = sim_params['rho']
        tau = sim_params['tau']
        alpha = sim_params['alpha']

        # Weighted Minkowski distance.
        d_qref = tf.pow(tf.abs(z_q - z_ref), rho)
        d_qref = tf.multiply(d_qref, attention_weights)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=1), 1. / rho)

        # Student-t family similarity kernel.
        s_qref = tf.pow(
            1 + (tf.pow(d_qref, tau) / alpha), tf.negative(alpha + 1)/2)
        return s_qref
