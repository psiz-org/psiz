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

    