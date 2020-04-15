def _build_model(self, tf_config):
        """Build TensorFlow model."""
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
            self.vars['theta'], self.vars['phi'], self.vars['z'],
            tf_config, self._tf_similarity
        )
        output = c_layer(inputs)
        model = tf.keras.models.Model(inputs, output)

        return model

class CoreLayer(Layer):
    """Core layer of model."""

    def __init__(self, tf_theta, tf_phi, tf_z, tf_config, tf_similarity):
        """Initialize.

        Arguments:
            tf_theta:
            tf_phi:
            tf_z:
            tf_config: It is assumed that the indices that will be
                passed in later as inputs will correspond to the
                indices in this data structure.
            tf_similarity:

        """
        super(CoreLayer, self).__init__()
        self.z = tf_z
        self.theta = tf_theta

        w_list = []
        for group, v in tf_phi['w'].items():
            w_list.append(v)
        self.attention = tf.concat(w_list, axis=0)

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
            prob_config = _tf_ranked_sequence_probability(
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


        # TODO write new code to resolve config indices and create locally
        # controlled IDs?

        # (obs_config_list, obs_config_idx) = self._grab_config_info(obs)
        # locs_train = np.hstack((
        #     np.ones(obs_train.n_trial, dtype=bool),
        #     np.zeros(obs_val.n_trial, dtype=bool)
        # ))
        # config_idx_train = obs_config_idx[locs_train]
        # config_idx_val = obs_config_idx[np.logical_not(locs_train)]

        # # Prepare observations.
        # tf_inputs_train = self._prepare_inputs(obs_train, config_idx_train)
        # tf_inputs_val = self._prepare_inputs(obs_val, config_idx_val)

        # model = self._build_model(obs)


        # TODO
        # tf_obs = [
        #     tf.constant(
        #         obs.stimulus_set, dtype=tf.int32, name='obs_stimulus_set'
        #     ),
        #     tf.constant(
        #         config_idx, dtype=tf.int32, name='obs_config_idx'
        #     ),
        #     tf.constant(obs.group_id, dtype=tf.int32, name='obs_group_id'),
        #     tf.constant(obs.weight, dtype=K.floatx(), name='obs_weight'),
        #     tf.constant(is_present, dtype=tf.bool, name='obs_is_present'),
        #     tf.constant(is_select, dtype=tf.bool, name='obs_is_select'),
        # ]
        # return tf_obs
        # ===================================================================
        # obs_config_list = copy.copy(obs.config_list)
        # obs_config_idx = copy.copy(obs.config_idx)
    
    def _build_model(self, obs):
        """Build TensorFlow model."""
        tf_config = [
            tf.constant(len(config_list.n_outcome.values)),
            tf.constant(config_list.n_reference.values),
            tf.constant(config_list.n_select.values),
            tf.constant(config_list.is_ranked.values)
        ]

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

        # TODO don't pass in config ID, pass in boolean arrays
        # is_reference
        # is_select
        # is_ranked
        obs_is_reference = tf.keras.Input(
            shape=[None], name='inp_is_reference', dtype=tf.bool
        )
        obs_is_select = tf.keras.Input(
            shape=[None], name='inp_is_select', dtype=tf.bool
        )

        inputs = [
            obs_stimulus_set,
            obs_config_idx,
            obs_group_id,
            obs_weight,  # TODO
            obs_is_reference,
            obs_is_select
        ]
        c_layer = CoreLayer(
            self.vars['theta'], self.vars['phi'], self.vars['z'],
            tf_config, self._tf_similarity
        )
        output = c_layer(inputs)
        model = tf.keras.models.Model(inputs, output)

        return model

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
    def _prepare_inputs_old(obs, config_idx):
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

    @staticmethod
    def _grab_config_info(obs):
        """Grab configuration information."""
        obs_config_list = copy.copy(obs.config_list)
        obs_config_idx = copy.copy(obs.config_idx)
        return obs_config_list, obs_config_idx

    @staticmethod
    def _prepare_inputs(obs, config_idx):
        is_present = obs.is_present()
        is_select = obs.is_select()

        tf_obs = [
            tf.constant(
                obs.stimulus_set, dtype=tf.int32, name='obs_stimulus_set'
            ),
            tf.constant(
                config_idx, dtype=tf.int32, name='obs_config_idx'
            ),
            tf.constant(obs.group_id, dtype=tf.int32, name='obs_group_id'),
            tf.constant(obs.weight, dtype=K.floatx(), name='obs_weight'),
            tf.constant(is_present, dtype=tf.bool, name='obs_is_present'),
            tf.constant(is_select, dtype=tf.bool, name='obs_is_select'),
        ]
        return tf_obs

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

    def _init_phi(self):
        """Initialize group-specific phi variables."""
        if 'phi' not in self.vars:
            self.vars['phi'] = {}

        if 'w' not in self.vars['phi']:
            self.vars['phi']['w'] = {}

        # Create group-specific `w_i`.
        for group_id in range(self.n_group):
            var_name = '{0}'.format(group_id)
            if var_name not in self.vars['phi']['w'] or self.vars['phi']['w'][var_name].trainable:
                self.vars['phi']['w'][var_name] = self._init_w_group(group_id)

    def _init_w_group(self, group_id):
        if self.n_group == 1:
            is_trainable = False
        else:
            is_trainable = True

        var_name = "w_{0}".format(group_id)
        scale = tf.constant(self.n_dim, dtype=K.floatx())
        alpha = tf.constant(np.ones((self.n_dim)), dtype=K.floatx())
        tf_attention = tf.Variable(
            initial_value=RandomAttention(
                alpha, scale, dtype=K.floatx()
            )(shape=[1, self.n_dim]),
            trainable=is_trainable, name=var_name, dtype=K.floatx(),
            constraint=ProjectAttention()
        )
        return tf_attention

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

    def _do_init_theta_param(self, param_name):
        """Return conditional for initialization."""
        cond = (
            param_name not in self.vars['theta'] or
            self.vars['theta'][param_name].trainable
        )
        return cond

class Inverse(PsychologicalEmbedding):
    """An inverse-distance model."""

    def __init__(self, n_stimuli, n_dim=2, n_group=1, z_min=None, z_max=None):
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
        PsychologicalEmbedding.__init__(
            self, n_stimuli, n_dim, n_group, z_min, z_max
        )
        self.kernel_layer = InverseKernel()

    # @property
    # def rho(self):
    #     """Getter method for rho."""
    #     return self.kernel_layer.distance_layer.rho.numpy()

    # @rho.setter
    # def rho(self, rho):
    #     """Setter method for rho."""
    #     self.kernel_layer.distance_layer.rho.assign(rho)

    # @property
    # def tau(self):
    #     """Getter method for tau."""
    #     return self.kernel_layer.tau.numpy()

    # @tau.setter
    # def tau(self, tau):
    #     """Setter method for tau."""
    #     self.kernel_layer.tau.assign(tau)

    # @property
    # def mu(self):
    #     """Getter method for mu."""
    #     return self.kernel_layer.mu.numpy()

    # @mu.setter
    # def mu(self, mu):
    #     """Setter method for mu."""
    #     self.kernel_layer.mu.assign(mu)


class Exponential(PsychologicalEmbedding):
    """An exponential family stochastic display embedding algorithm."""

    def __init__(self, n_stimuli, n_dim=2, n_group=1, z_min=None, z_max=None):
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
        PsychologicalEmbedding.__init__(
            self, n_stimuli, n_dim, n_group, z_min, z_max
        )
        self.kernel_layer = ExponentialKernel()  # TODO how to pass in trainable settings?

    # @property
    # def rho(self):
    #     """Getter method for rho."""
    #     return self.kernel_layer.distance_layer.rho.numpy()

    # @rho.setter
    # def rho(self, rho):
    #     """Setter method for rho."""
    #     self.kernel_layer.distance_layer.rho.assign(rho)

    # @property
    # def tau(self):
    #     """Getter method for tau."""
    #     return self.kernel_layer.tau.numpy()

    # @tau.setter
    # def tau(self, tau):
    #     """Setter method for tau."""
    #     self.kernel_layer.tau.assign(tau)

    # @property
    # def gamma(self):
    #     """Getter method for gamma."""
    #     return self.kernel_layer.gamma.numpy()

    # @gamma.setter
    # def gamma(self, gamma):
    #     """Setter method for gamma."""
    #     self.kernel_layer.gamma.assign(gamma)

    # @property
    # def beta(self):
    #     """Getter method for beta."""
    #     return self.kernel_layer.beta.numpy()

    # @beta.setter
    # def beta(self, beta):
    #     """Setter method for beta."""
    #     self.kernel_layer.beta.assign(beta)

    # @property
    # def theta(self):
    #     """Getter method for theta."""
    #     d = {
    #         'rho': self.rho,
    #         'tau': self.tau,
    #         'beta': self.beta,
    #         'gamma': self.gamma
    #     }
    #     return d

    # @theta.setter
    # def theta(self, theta):
    #     """Setter method for w."""
    #     for k, v in theta.items():
    #         var = getattr(self.kernel_layer, k)
    #         var.assign(v)


class HeavyTailed(PsychologicalEmbedding):
    """A heavy-tailed family stochastic display embedding algorithm."""

    def __init__(self, n_stimuli, n_dim=2, n_group=1, z_min=None, z_max=None):
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
        PsychologicalEmbedding.__init__(
            self, n_stimuli, n_dim, n_group, z_min, z_max
        )
        self.kernel_layer = HeavyTailedKernel()

    # @property
    # def rho(self):
    #     """Getter method for rho."""
    #     return self.kernel_layer.distance_layer.rho.numpy()

    # @rho.setter
    # def rho(self, rho):
    #     """Setter method for rho."""
    #     self.kernel_layer.distance_layer.rho.assign(rho)

    # @property
    # def tau(self):
    #     """Getter method for tau."""
    #     return self.kernel_layer.tau.numpy()

    # @tau.setter
    # def tau(self, tau):
    #     """Setter method for tau."""
    #     self.kernel_layer.tau.assign(tau)

    # @property
    # def kappa(self):
    #     """Getter method for kappa."""
    #     return self.kernel_layer.kappa.numpy()

    # @kappa.setter
    # def kappa(self, kappa):
    #     """Setter method for kappa."""
    #     self.kernel_layer.kappa.assign(kappa)

    # @property
    # def alpha(self):
    #     """Getter method for alpha."""
    #     return self.kernel_layer.alpha.numpy()

    # @alpha.setter
    # def alpha(self, alpha):
    #     """Setter method for alpha."""
    #     self.kernel_layer.alpha.assign(alpha)


class StudentsT(PsychologicalEmbedding):
    """A Student's t family stochastic display embedding algorithm."""

    def __init__(self, n_stimuli, n_dim=2, n_group=1, z_min=None, z_max=None):
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
        PsychologicalEmbedding.__init__(
            self, n_stimuli, n_dim, n_group, z_min, z_max
        )
        self.kernel_layer = StudentsTKernel()

    # @property
    # def rho(self):
    #     """Getter method for rho."""
    #     return self.kernel_layer.rho.numpy()

    # @rho.setter
    # def rho(self, rho):
    #     """Setter method for rho."""
    #     self.kernel_layer.rho.assign(rho)

    # @property
    # def tau(self):
    #     """Getter method for tau."""
    #     return self.kernel_layer.tau.numpy()

    # @tau.setter
    # def tau(self, tau):
    #     """Setter method for tau."""
    #     self.kernel_layer.tau.assign(tau)

    # @property
    # def alpha(self):
    #     """Getter method for alpha."""
    #     return self.kernel_layer.alpha.numpy()

    # @alpha.setter
    # def alpha(self, alpha):
    #     """Setter method for alpha."""
    #     self.kernel_layer.alpha.assign(alpha)

@tf.function(experimental_relax_shapes=True)
def custom_loss(prob, weight, tf_attention):
    """Compute model loss given observation probabilities."""
    n_trial = tf.shape(prob)[0]
    n_trial = tf.cast(n_trial, dtype=K.floatx())
    n_group = tf.shape(tf_attention)[0]

    # Convert to (weighted) log probabilities.
    cap = tf.constant(2.2204e-16, dtype=K.floatx())
    logprob = tf.math.log(tf.maximum(prob, cap))
    logprob = tf.multiply(weight, logprob)

    # Divide by number of trials to make train and test loss
    # comparable.
    loss = tf.negative(tf.reduce_sum(logprob))
    loss = tf.divide(loss, n_trial)

    # Penalty on attention weights (independently).
    attention_penalty = tf.constant(0, dtype=K.floatx())
    for i_group in tf.range(n_group):
        attention_penalty = (
            attention_penalty +
            entropy_loss(tf_attention[i_group, :])
        )
    attention_penalty = (
        attention_penalty / tf.cast(n_group, dtype=K.floatx())
    )
    loss = loss + (attention_penalty / tf.constant(10.0, dtype=K.floatx()))

    return loss


def attention_sparsity_loss(w):
    """Sparsity encouragement.

    The traditional regularizer to encourage sparsity is L1.
    Unfortunately, L1 regularization does not work for the attention
    weights since they are all constrained to sum to the same value
    (i.e., the number of dimensions). Instead, we achieve sparsity
    pressure by using a complement version of L2 loss. It tries to make
    each attention weight as close to zero as possible, putting
    pressure on the model to only use the dimensions it really needs.

    Arguments:
        w: Attention weights assumed to be nonnegative.

    """
    n_dim = tf.cast(tf.shape(w)[0], dtype=K.floatx())
    loss = tf.negative(
        tf.math.reduce_mean(tf.math.pow(n_dim - w, 2))
    )
    return loss


def entropy_loss(w):
    """Loss term based on entropy that encourages sparsity.

    Arguments:
        w: Attention weights assumed to be nonnegative.

    """
    n_dim = tf.cast(tf.shape(w)[0], dtype=K.floatx())
    w_1 = w / n_dim + tf.keras.backend.epsilon()
    loss = tf.negative(
        tf.math.reduce_sum(w_1 * tf.math.log(w_1))
    )
    return loss


        (z_q, z_r) = self._tf_inflate_points_old(
            stimulus_set, max_n_reference, z_pad
        )
        z_stimulus_set_0 = tf.concat([z_q, z_r], axis=2)
        # z_stimulus_set_1 = self._tf_inflate_points_1(
        #     stimulus_set, input_length, z_pad
        # )
        # np.testing.assert_array_equal(z_stimulus_set_0, z_stimulus_set_1)


def _tf_inflate_points_1(self, stimulus_set, input_length, z):
        """Inflate stimulus set into embedding points.

        Note: This method will not gracefully handle the masking
        placeholder stimulus ID (i.e., -1). The stimulus IDs and
        coordinates must already have been adjusted for the masking
        placeholder.

        """
        n_trial = tf.shape(stimulus_set)[0]
        n_dim = tf.shape(z)[1]

        # Pre-allocate for embedding points.
        # NOTE: Dimensions are permuted to facilitate scatter update.
        z_set = tf.zeros([input_length, n_trial, n_dim], dtype=K.floatx())

        for i_input in tf.range(input_length):
            # Grab indices.
            z_set_update = tf.gather(z, stimulus_set[:, i_input])
            z_set_update = tf.expand_dims(z_set_update, axis=0)

            # Expand dimensions for scatter update.
            i_input_expand = tf.expand_dims(i_input, axis=0)
            i_input_expand = tf.expand_dims(i_input_expand, axis=0)

            z_set = tf.tensor_scatter_nd_update(
                z_set, i_input_expand, z_set_update
            )

        z_set = tf.transpose(z_set, perm=[1, 2, 0])
        return z_set

    def _tf_inflate_points_old(
            self, stimulus_set, n_reference, z):
        """Inflate stimulus set into embedding points.

        Note: This method will not gracefully handle the masking
        placeholder stimulus ID (i.e., -1). The stimulus IDs and
        coordinates must already have been adjusted for the masking
        placeholder.

        """
        n_trial = tf.shape(stimulus_set)[0]
        n_dim = tf.shape(z)[1]

        # Inflate query stimuli.
        z_q = tf.gather(z, stimulus_set[:, 0])
        z_q = tf.expand_dims(z_q, axis=2)

        # Initialize z_r.
        # z_r = tf.zeros([n_trial, n_dim, n_reference], dtype=K.floatx())
        z_r = tf.zeros([n_reference, n_trial, n_dim], dtype=K.floatx())

        for i_ref in tf.range(n_reference):
            z_r_new = tf.gather(
                z, stimulus_set[:, i_ref + tf.constant(1, dtype=tf.int32)]
            )

            i_ref_expand = tf.expand_dims(i_ref, axis=0)
            i_ref_expand = tf.expand_dims(i_ref_expand, axis=0)
            z_r_new_2 = tf.expand_dims(z_r_new, axis=0)
            z_r = tf.tensor_scatter_nd_update(
                z_r, i_ref_expand, z_r_new_2
            )

            # z_r_new = tf.expand_dims(z_r_new, axis=2)
            # pre_pad = tf.zeros([n_trial, n_dim, i_ref], dtype=K.floatx())
            # post_pad = tf.zeros([
            #     n_trial, n_dim,
            #     n_reference - i_ref - tf.constant(1, dtype=tf.int32)
            # ], dtype=K.floatx())
            # z_r_new = tf.concat([pre_pad, z_r_new, post_pad], axis=2)
            # z_r = z_r + z_r_new

        z_r = tf.transpose(z_r, perm=[1, 2, 0])
        return (z_q, z_r)

    