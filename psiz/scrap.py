# mean = np.zeros((self.n_dim))
        # cov = .03 * np.identity(self.n_dim)
        # z = {}
        # z["value"] = np.random.multivariate_normal(
        #     mean, cov, (self.n_stimuli)
        # )
        # if self._is_nonneg:
        #     z["value"] = np.abs(z["value"])
        # z["trainable"] = True
        
    # def _get_embedding(self, init_mode, is_nonneg):
    #     """Return embedding of model as TensorFlow variable.

    #     Arguments:
    #         init_mode: A string indicating the initialization mode.
    #             valid options are 'cold' and 'hot'.

    #     Returns:
    #         TensorFlow variable representing the embedding points.

    #     """
    #     # Handle constraint. TODO
    #     if is_nonneg:
    #         z_constraint = tf.keras.constraints.NonNeg()
    #     else:
    #         z_constraint = ProjectZ()

    #     if self._z["trainable"]:
    #         if init_mode is 'hot':
    #             z = self._z["value"]
    #             tf_z = tf.Variable(
    #                 initial_value=z,
    #                 trainable=True, name="z", dtype=K.floatx(),
    #                 constraint=z_constraint,
    #                 shape=[self.n_stimuli, self.n_dim]
    #             )
    #         else:
    #             # TODO do I need to do anything special to handle nonnegativitiy here?
    #             tf_z = tf.Variable(
    #                 initial_value=RandomEmbedding(
    #                     mean=tf.zeros([self.n_dim], dtype=K.floatx()),
    #                     stdev=tf.ones([self.n_dim], dtype=K.floatx()),
    #                     minval=tf.constant(-3., dtype=K.floatx()),
    #                     maxval=tf.constant(0., dtype=K.floatx()),
    #                     dtype=K.floatx()
    #                 )(shape=[self.n_stimuli, self.n_dim]), trainable=True,
    #                 name="z", dtype=K.floatx(),
    #                 constraint=z_constraint
    #             )
    #     else:
    #         tf_z = tf.Variable(
    #             initial_value=self._z["value"],
    #             trainable=False, name="z", dtype=K.floatx(),
    #             constraint=z_constraint,
    #             shape=[self.n_stimuli, self.n_dim]
    #         )
    #     return tf_z

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
    
    def _get_attention(self, init_mode):
        """Return attention weights of model as TensorFlow variable."""
        attention_list = []
        for group_id in range(self.n_group):
            tf_attention = self._get_group_attention(init_mode, group_id)
            attention_list.append(tf_attention)
        tf_attention = tf.concat(attention_list, axis=0)
        return tf_attention
    
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
                self._z.trainable = False
                self._phi["w"]["trainable"] = np.zeros(
                    self.n_group, dtype=bool
                )
                for param_name in self._theta:
                    self._theta[param_name]["trainable"] = False
            elif spec == 'thaw':
                self._z.trainable = True
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
            obs: A RankObservations object representing the observed data.
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
        mu = gmm.means_[0]
        sigma = gmm.covariances_[0] * np.identity(n_dim)

        # Center embedding to satisfy assumptions of elliptical slice sampling.
        z = z - mu
        z_orig = tf.constant(z, dtype=K.floatx())
        mu = tf.constant(mu, dtype=K.floatx())
        mu = tf.expand_dims(mu, axis=0)
        mu = tf.expand_dims(mu, axis=0)

        # Prepare obs and model.
        (obs_config_list, obs_config_idx) = self._grab_config_info(obs)  # TODO
        tf_inputs = self._prepare_inputs(obs, obs_config_idx)
        model = self._build_model(obs)  # TODO
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

# class EarlyStopping(object):
#     """Early Stopping."""

#     def __init__(
#             self, monitor='val_loss', min_delta=0, patience=0, verbose=0,
#             mode='auto', baseline=None, restore_best_weights=False):
#         """Initialize."""
#         self.monitor = monitor
#         self.patience = patience
#         self.verbose = verbose
#         self.baseline = baseline
#         self.min_delta = abs(min_delta)
#         self.wait = 0
#         self.stopped_epoch = 0
#         self.restore_best_weights = restore_best_weights
#         self.best_weights = None
#         self.model = None  # TODO

#         if mode not in ['auto', 'min', 'max']:
#             print(
#                 'WARNING: EarlyStopping mode %s is unknown, '
#                 'fallback to auto mode.', mode
#             )
#             mode = 'auto'

#         if mode == 'min':
#             self.monitor_op = np.less
#         elif mode == 'max':
#             self.monitor_op = np.greater
#         else:
#             if 'acc' in self.monitor:
#                 self.monitor_op = np.greater
#             else:
#                 self.monitor_op = np.less

#         if self.monitor_op == np.greater:
#             self.min_delta *= 1
#         else:
#             self.min_delta *= -1

#     def on_train_begin(self, logs=None):
#         """On train begin."""
#         # Allow instances to be re-used
#         self.wait = 0
#         self.stopped_epoch = 0
#         if self.baseline is not None:
#             self.best = self.baseline
#         else:
#             self.best = np.Inf if self.monitor_op == np.less else -np.Inf

#     def on_epoch_end(self, epoch, logs=None):
#         """On epoch end."""
#         current = self.get_monitor_value(logs)
#         if current is None:
#             return
#         if self.monitor_op(current - self.min_delta, self.best):
#             self.best = current
#             self.wait = 0
#             if self.restore_best_weights:
#                 self.best_weights = self.model.get_weights()
#         else:
#             self.wait += 1
#             if self.wait >= self.patience:
#                 self.stopped_epoch = epoch
#                 self.model.stop_training = True
#                 if self.restore_best_weights:
#                     if self.verbose > 0:
#                         print('Restoring model weights from the end of the best epoch.')
#                     self.model.set_weights(self.best_weights)

#     def get_monitor_value(self, logs):
#         """Get monitor value."""
#         logs = logs or {}
#         monitor_value = logs.get(self.monitor)
#         if monitor_value is None:
#             logging.warning(
#                 'Early stopping conditioned on metric `%s` '
#                 'which is not available. Available metrics are: %s',
#                 self.monitor, ','.join(list(logs.keys()))
#             )
#         return monitor_value