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


@tf.function(experimental_relax_shapes=True)
def custom_loss(prob, weight, tf_attention):
    """Compute model loss given observation probabilities."""
    n_group = tf.shape(tf_attention)[0]

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

    # Alternative call and _tf_ranked_sequence that can't handle different
    # n_select.
    @tf.function
    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A dictionary of inputs:
                stimulus_set: dtype=tf.int32, consisting of the
                    integers on the interval [0, n_stimuli[
                config_idx: dtype=tf.int32, consisting of the
                    integers on the interval [0, n_config[
                group_id: dtype=tf.int32, consisting of the
                    integers on the interval [0, n_group[
                is_present: dtype=tf.bool

        """
        # Grab inputs.
        obs_stimulus_set = inputs['stimulus_set']
        obs_config_idx = inputs['config_idx']
        obs_group_id = inputs['group_id']
        is_present = inputs['is_present']
        is_select = inputs['is_select']

        # Expand attention weights.
        attention = self.attention(obs_group_id)

        # Inflate cooridnates.
        z_stimulus_set = self.embedding(obs_stimulus_set)
        max_n_reference = tf.shape(z_stimulus_set)[2] - 1
        z_q, z_r = tf.split(z_stimulus_set, [1, max_n_reference], 2)

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

        # TODO
        likelihood_alt = _tf_ranked_sequence_probability_alt(
            sim_qr, is_select
        )
        return likelihood


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


# OLD intermediate save
def save(filepath):
    f = h5py.File(filepath, "w")
    f.create_dataset("embedding_type", data=type(self).__name__)
    f.create_dataset("n_stimuli", data=self.n_stimuli)
    f.create_dataset("n_dim", data=self.n_dim)
    f.create_dataset("n_group", data=self.n_group)

    # Save model architecture.
    grp_arch = f.create_group('architecture')
    # Create group for embedding layer.
    grp_coord = grp_arch.create_group('embedding')
    _add_layer_to_save_architecture(grp_coord, self.embedding)
    # Create group for attention layer.
    grp_attention = grp_arch.create_group('attention')
    _add_layer_to_save_architecture(grp_attention, self.attention)
    # Create group for kernel layer.
    grp_kernel = grp_arch.create_group('kernel')
    _add_layer_to_save_architecture(grp_kernel, self.kernel)

    # Save weights.
    weights = self.get_weights()
    grp_weights = f.create_group("weights")
    for layer_name, layer_dict in weights.items():
        grp_layer = grp_weights.create_group(layer_name)
        for k, v in layer_dict.items():
            grp_layer.create_dataset(k, data=v)

    f.close()


def _add_layer_to_save_architecture(grp_layer, layer):
    """Add layer information to layer group.

    Arguments:
        grp_layer: An HDF5 group.
        layer: A TensorFlow layer with a `get_config` method.

    """
    grp_layer.create_dataset(
        'class_name', data=type(layer).__name__
    )
    grp_config = grp_layer.create_group('config')
    layer_config = layer.get_config()
    for k, v in layer_config.items():
        if v is not None:
            grp_config.create_dataset(k, data=v)


# Old load
    if embedding_type == 'Rank':
        grp_architecture = f['architecture']
        # Instantiate embedding layer.
        embedding = _load_layer(
            grp_architecture['embedding'], custom_objects
        )
        # Instantiate attention layer.
        attention = _load_layer(
            grp_architecture['attention'], custom_objects
        )
        # Instantiate kernel layer.
        kernel = _load_layer(
            grp_architecture['kernel'], custom_objects
        )

        emb = Rank(
            n_stimuli, n_dim=n_dim, n_group=n_group,
            embedding=embedding, attention=attention,
            kernel=kernel
        )

        # Set weights.
        grp_weights = f['weights']
        # Assemble dictionary of weights.
        weights = {}
        for layer_name, grp_layer in grp_weights.items():
            layer_weights = {}
            for var_name in grp_layer:
                layer_weights[var_name] = grp_weights[layer_name][var_name][()]
            weights[layer_name] = layer_weights
        emb.set_weights(weights)

def _load_layer(grp_layer, custom_objects):
    """Load a configured layer.

    Arguments:
        grp_layer: An HDF5 group.
        custom_objects: A list of custom classes.

    Returns:
        layer: An instantiated and configured TensorFlow layer.

    """
    layer_class_name = grp_layer['class_name'][()]
    layer_config = {}
    for k in grp_layer['config']:
        layer_config[k] = grp_layer['config'][k][()]

    if layer_class_name in custom_objects:
        layer_class = custom_objects[layer_class_name]
    else:
        layer_class = getattr(psiz.keras.layers, layer_class_name)
    return layer_class.from_config(layer_config)


# SavedModel format code
# Save.
self.model.save(
    os.fspath(filepath), overwrite=overwrite,
    include_optimizer=include_optimizer, save_format=save_format,
    signatures=signatures, options=options
)

# Load.
model = tf.keras.models.load_model(
    os.fspath(filepath), custom_objects=custom_objects, compile=compile
)
emb = Proxy(model=model)


def _load_layer(config, custom_objects={}):
    """Load a configured layer.

    Arguments:
        config: A configuration dictionary.
        custom_objects: A dictionary of custom classes.

    Returns:
        layer: An instantiated and configured TensorFlow layer.

    """
    layer_class_name = config.get('class_name')
    layer_config = config.get('config')

    if layer_class_name in custom_objects:
        layer_class = custom_objects[layer_class_name]
    else:
        layer_class = getattr(psiz.keras.layers, layer_class_name)

    return layer_class.from_config(layer_config)
