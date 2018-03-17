'''Embedding models

n_group must be defined on instantiation. this determines how many separate
sets of attention weights will be used and inferred.

fit, freeze, reuse are the only methods that modify the state of the class

Author: B D Roads
'''
from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import psiz.utils as ut

class Observations(object):
    '''A wrapper object used by the class PsychologicalEmbedding for passing
    around observation data.
    '''
    def __init__(self, displays, n_reference, n_selected, is_ranked, group_id):
        '''
        Initialize
        '''
        self.displays = displays
        self.n_reference = n_reference
        self.n_selected = n_selected
        self.is_ranked = is_ranked
        self.group_id = group_id

class PsychologicalEmbedding(object):
    '''Abstract base class for psychological embedding algorithm. 
    
    The embedding procedure jointly infers two components. First, the embedding
    algorithm infers a stimulus representation denoted Z. Second, the embedding
    algoirthm infers the similarity kernel parameters of the concrete class.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, n_stimuli, dimensionality=2, n_group=1):
        '''
        Initialize

        Parameters:
          n_stimuli: An integer indicating the total number of unique stimuli
            that will be embedded.
          dimensionality: An integer indicating the dimensionalty of the 
            embedding. The dimensionality can be inferred using the function
            "suggest_dimensionality".
          n_group: (default: 1) An integer indicating the number of different
            population groups in the embedding. A separate set of attention 
            weights will be inferred for each group.
        '''
        self.n_stimuli = n_stimuli
        self.n_group = n_group

        # Check if the dimensionality is valid.
        if (dimensionality < 1):
            # User supplied a bad dimensionality value.
            raise ValueError('The provided dimensionality must be an integer greater than 0.')
        
        # Initialize dimension dependent attributes.
        self.dimensionality = dimensionality
        # Initialize random embedding points using multivariate Gaussian.
        mean = np.ones((dimensionality))
        cov = np.identity(dimensionality)
        self.Z = np.random.multivariate_normal(mean, cov, (self.n_stimuli))
        # Initialize attentional weights using uniform distribution.
        self.attention_weights = np.ones((self.n_group, dimensionality), dtype=np.float64)
        
        self.infer_Z = True
        if n_group is 1:
            self.infer_attention_weights = False
        else:
            self.infer_attention_weights = True

        # Initialize default reuse attributes.
        self.do_reuse = False
        self.init_scale = 0

        # Initialize default TensorBoard log attributes.
        self.do_log = False
        self.log_dir = '/tmp/tensorflow_logs/embedding/'

        # Default inference settings.
        self.lr = 0.00001
        self.max_n_epoch = 2000
        self.patience = 10

        super().__init__()

    @abstractmethod
    def _get_similarity_parameters(self):
        '''Returns a dictionary and TensorFlow operation.
        
        This method encapsulates the creation of algorithm-specific free
        parameters governing the similarity kernel.

        Returns:
          sim_params: A dictionary of algorithm-specific TensorFlow variables.
          sim_constraints: A TensorFlow operation that imposes boundary 
            constraints on the algorithm-specific free parameters during
            inference.
        '''
        pass
    
    @abstractmethod
    def _set_parameters(self, params):
        '''A state changing method that sets algorithm-specific parameters.
        
        This method encapsulates the setting of algorithm-specific free 
        parameters governing the similarity kernel.

        Parameters:
          params: An ordered list of algorithm-specific parameters.

        TODO add "*args" to parameter list?
        TODO use dictionary instead of ordered list?
        '''
        pass

    @abstractmethod
    def freeze(self):
        '''A state changing method specifing which parameters are fixed.

        During inference, you may want to freeze some parameters at specific
        values. To freeze a particular parameter, pass in a value for it. If you
        would like to freeze multiple parameters, they must be passed in at the 
        same time. If no value is provided for a parameter, it will not be fixed.
        
        TODO add "*args" to parameter list?
        '''
        pass
    
    @abstractmethod
    def similarity(self, z_q, z_ref, sim_params, attention_weights):
        '''Similarity kernel.

        Parameters:
          z_q: A set of embedding points.
            shape = (n_sample, dimensionality)
          z_ref: A set of embedding points.
            shape = (n_sample, dimensionality)
          sim_params: A dictionary of algorithm-specific parameters governing
            the similarity kernel.
          attention_weights: The weights allocated to each dimension in a 
            weighted minkowski metric.
            shape = (n_sample, dimensionality)
        Returns:
          similarity: The corresponding similairty between rows of embedding
            points.
            shape = (n_sample,)
        '''
        pass

    def _get_attention_weights(self):
        '''TODO
        '''
        # Attention variable
        if self.do_reuse:
            attention_weights = tf.get_variable("attention_weights", [self.n_group, self.dimensionality], initializer=tf.constant_initializer(self.attention_weights), trainable=True)
        else:
            if self.infer_attention_weights:
                alpha = 1. * np.ones((self.dimensionality))
                new_attention_weights = np.random.dirichlet(alpha) * self.dimensionality
                attention_weights = tf.get_variable("attention_weights", [self.n_group, self.dimensionality], initializer=tf.constant_initializer(new_attention_weights))
            else:
                attention_weights = tf.get_variable("attention_weights", [self.n_group, self.dimensionality], initializer=tf.constant_initializer(self.attention_weights), trainable=False)
        return attention_weights

    def _get_embedding(self):
        '''TODO
        '''
        # Embedding variable
        # Iniitalize Z with different scales for different restarts
        init_scale_list = [.001, .01, .1]
        rand_scale_idx = np.random.randint(0,3)
        scale_value = init_scale_list[rand_scale_idx]
        tf_scale_value = tf.constant(scale_value, dtype=tf.float32)
        
        if self.do_reuse:
            Z = tf.get_variable("Z", [self.n_stimuli, self.dimensionality], initializer=tf.constant_initializer(self.Z), trainable=True)
        else:
            if self.infer_Z:
                Z = tf.get_variable("Z", [self.n_stimuli, self.dimensionality], initializer=tf.random_normal_initializer(tf.zeros([self.dimensionality]), tf.ones([self.dimensionality]) * tf_scale_value))
            else:
                Z = tf.get_variable("Z", [self.n_stimuli, self.dimensionality], initializer=tf.constant_initializer(self.Z), trainable=False)
        return Z
    
    def reuse(self, do_reuse, init_scale=0):
        '''State changing method that sets reuse of embedding.
        
        Parameters:
          do_reuse: Boolean that indicates whether the current embedding should
          be used for initialization during inference.
          init_scale: A scalar value indicating to went extent the previous
            embedding points should be reused. For example, a value of 0.05
            would add uniform noise to all the points in the embedding such
            that each embedding point was randomly jittered up to 5% on each
            dimension relative to the overall size of the embedding. The value
            can be between [0,1].
        '''
        self.do_reuse = do_reuse
        self.init_scale = init_scale

    def set_log(self, do_log, log_dir=None, delete_prev=False):
        '''State changing method that sets TensorBoard logging.

        Parameters:
          do_log: Boolean that indicates whether logs should be recorded.
          log_dir: A string indicating the file path for the logs.
          delete_prev: Boolean indicating whether the directory should
            be cleared of previous files first.
        '''
        if do_log:
            self.do_log = True
            if log_dir is not None:
                self.log_dir = log_dir
            
        if delete_prev:
            if tf.gfile.Exists(self.log_dir):
                tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
        
    def fit(self, displays, n_selected=None, is_ranked=None, group_id=None, 
        n_restart=40, verbose=0):
        '''Fits the free parameters of the embedding model.

        Parameters:
          displays: An integer matrix representing the displays (rows) that 
            have been judged based on similarity. The shape implies the 
            number of references in shown in each display. The first column 
            is the query, then the selected references in order of selection,
            and then any remaining unselected references.
            shape = [n_display, max(n_reference) + 1]
          n_selected: An integer array indicating the number of references 
            selected in each display.
            shape = [n_display, 1]
          is_ranked:  Boolean array indicating which displays had selected
            references that were ordered.
            shape = [n_display, 1]
          group_id: An integer array indicating the group membership of each 
            display. It is assumed that group is composed of integers from 
            [0,N] where N is the total number of groups. Separate attention 
            weights are inferred for each group.
            shape = [n_display, 1]
          n_restart: An integer specifying the number of restarts to use for 
            the inference procedure. Since the embedding procedure sometimes
            gets stuck in local optima, multiple restarts helps find the global
            optimum.
          verbose: An integer specifying the verbosity of printed output.

        Returns:
          J: The average loss (-loglikelihood) per observation (i.e., display).
        '''

        n_display = displays.shape[0]
        # Handle default settings.
        if n_selected is None:
            n_selected = np.ones((n_display))
        if is_ranked is None:
            is_ranked = np.full((n_display), True)
        if group_id is None:
            group_id = np.zeros((n_display))

        # Infer n_reference for each display.
        n_reference = ut.infer_n_reference(displays)

        # Package up the observation data.
        obs = Observations(displays, n_reference, n_selected, is_ranked, group_id)

        dimensionality = self.dimensionality

        #  Infer embedding.
        if (verbose > 0):
            print('Inferring embedding ...')
            print('\tSettings:')
            print('\tn_observations: ', displays.shape[0])
            print('\tn_group: ', len(np.unique(group_id)))
            print('\tdimensionality: ', dimensionality)
            print('\tn_restart: ', n_restart)
        
        # Partition data into train and validation set for early stopping of 
        # embedding algorithm.
        display_type_id = ut.generate_display_type_id(n_reference, n_selected, is_ranked, group_id)
        skf = StratifiedKFold(n_splits=10)
        (train_idx, test_idx) = list(skf.split(displays, display_type_id))[0]
        
        # Run multiple restarts of embedding algorithm.
        loaded_func = lambda i_restart: self._embed(obs, train_idx, test_idx, i_restart)
        (J_all, Z, attention_weights, params) = self._embed_restart(loaded_func, n_restart, verbose)

        self.Z = Z
        self.attention_weights = attention_weights
        self._set_parameters(params)

        return J_all / n_display
    
    def evaluate(self, displays, n_selected=None, is_ranked=None, group_id=None):
        '''Evaluate observations using the current state of the embedding object.

        Parameters:
          displays: An integer matrix representing the displays (rows) that 
            have been judged based on similarity. The shape implies the 
            number of references in shown in each display. The first column 
            is the query, then the selected references in order of selection,
            and then any remaining unselected references.
            shape = [n_display, max(n_reference) + 1]
          n_selected: An integer array indicating the number of references 
            selected in each display.
            shape = [n_display, 1]
          is_ranked:  Boolean array indicating which displays had selected
            references that were ordered.
            shape = [n_display, 1]
          group_id: An integer array indicating the group membership of each 
            display. It is assumed that group is composed of integers from 
            [0,N] where N is the total number of groups. Separate attention 
            weights are inferred for each group.
            shape = [n_display, 1]

        Returns:
          J: The average loss (-loglikelihood) per observation (i.e., display) 
            given the current model.
        '''
        n_display = displays.shape[0]
        # Handle default settings
        if n_selected is None:
            n_selected = np.ones((n_display))
        if is_ranked is None:
            is_ranked = np.full((n_display), True)
        if group_id is None:
            group_id = np.zeros((n_display))
        
        # Infer n_reference for each display
        n_reference = ut.infer_n_reference(displays)

        # Package up
        obs = Observations(displays, n_reference, n_selected, is_ranked, group_id)

        J = self._concrete_evaluate(obs)
        return J / n_display

    def _embed(self, obs, train_idx, test_idx, i_restart):
        '''TODO
        '''
        verbose = 0 # TODO make parameter

        # Partition the observation data.
        displays_train = obs.displays[train_idx,:]
        n_reference_train = obs.n_reference[train_idx]
        n_selected_train = obs.n_selected[train_idx]
        is_ranked_train = obs.is_ranked[train_idx]
        group_id_train = obs.group_id[train_idx]

        displays_val = obs.displays[test_idx,:]
        n_reference_val = obs.n_reference[test_idx]
        n_selected_val = obs.n_selected[test_idx]
        is_ranked_val = obs.is_ranked[test_idx]
        group_id_val = obs.group_id[test_idx]

        (J, Z, attention_weights, sim_params, constraint, tf_displays, tf_n_reference, tf_n_selected, tf_is_ranked, tf_group_id) = self._core_model()
        
        train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(J)

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
                    # tf.summary.histogram(param_name + '_hist', sim_params[param_name])
            
            # Merge all summaries into a single op.
            merged_summary_op = tf.summary.merge_all()

        sess = tf.Session()
        sess.run(init)

        # op to write logs for TensorBoard
        if self.do_log:
            summary_writer = tf.summary.FileWriter('%s/%s' % (self.log_dir, i_restart), graph=tf.get_default_graph())

        J_all_best = np.inf
        J_test_best = np.inf
        
        last_improvement = 0
        for epoch in range(self.max_n_epoch):
            _, J_train, summary = sess.run([train_op, J, merged_summary_op], 
            feed_dict={tf_displays: displays_train, 
            tf_n_reference: n_reference_train, 
            tf_n_selected: n_selected_train, 
            tf_is_ranked: is_ranked_train, 
            tf_group_id: group_id_train})
            
            sess.run(constraint)
            J_test = sess.run(J, feed_dict={
                tf_displays: displays_val, 
                tf_n_reference: n_reference_val,
                tf_n_selected: n_selected_val, 
                tf_is_ranked: is_ranked_val, 
                tf_group_id: group_id_val})

            J_all =  sess.run(J, feed_dict={
                tf_displays: obs.displays, 
                tf_n_reference: obs.n_reference,
                tf_n_selected: obs.n_selected, 
                tf_is_ranked: obs.is_ranked, 
                tf_group_id: obs.group_id})

            if J_test < J_test_best:
                J_all_best = J_all
                J_test_best = J_test
                last_improvement = 0
                (Z_best, attention_weights_best) = sess.run([Z, attention_weights])
                params_best = {}
                for param_name in sim_params:
                    params_best[param_name] = sess.run(sim_params[param_name])
            else:
                last_improvement = last_improvement + 1

            if last_improvement > self.patience:
                break
            
            if not epoch%10:
                # Write logs at every 10th iteration
                if self.do_log:
                    summary_writer.add_summary(summary, epoch)
            if not epoch%100:
                if verbose > 2:
                    print("epoch ", epoch, "| J_train: ", J_train, 
                    "| J_test: ", J_test, "| J_all: ", J_all)


        sess.close()
        tf.reset_default_graph()

        return (J_all_best, Z_best, attention_weights_best, params_best)

    def _embed_restart(self, loaded_func, n_restart, verbose):
        '''Multiple restart wrapper. 
        
        The results of the best performing restart are returned.

        Parameters:
          n_restart: The number of restarts to perform.
        Returns:
          loss:
          Z:
          attention_weights:
          params:
        '''
        J_all_best = np.inf
        Z_best = None
        attention_weights_best = None
        params_best = None

        for i_restart in range(n_restart):
            (J_all, Z, attention_weights, params) = loaded_func(i_restart)
            if J_all < J_all_best:
                J_all_best = J_all
                Z_best = Z
                attention_weights_best = attention_weights
                params_best = params

            if verbose > 1:
                print('Restart ', i_restart)

        return (J_all_best, Z_best, attention_weights_best, params_best)

    def _project_attention_weights(self, attention_weights_0):
        '''Projection attention weights for gradient descent.
        '''
        n_dim = tf.shape(attention_weights_0, out_type=tf.float64)[1]
        attention_weights_1 = tf.divide(tf.reduce_sum(attention_weights_0, axis=1, keepdims=True), n_dim)
        attention_weights_proj = tf.divide(attention_weights_0, attention_weights_1)
    
        return attention_weights_proj
    
    def _cost_2c1(self, Z, triplets, sim_params, attention_weights):
        '''Cost associated with an ordered 2 chooose 1 display.
        '''
        # Similarity
        Sqa = self.similarity(tf.gather(Z, triplets[:,0]), 
        tf.gather(Z, triplets[:,1]), sim_params, attention_weights)
        Sqb = self.similarity(tf.gather(Z, triplets[:,0]), 
        tf.gather(Z, triplets[:,2]), sim_params, attention_weights)
        # Probility of behavior
        P = Sqa / (Sqa + Sqb)
        # Cost function
        cap = tf.constant(2.2204e-16)
        J = tf.negative(tf.reduce_sum(tf.log(tf.maximum(P, cap))))
        return J
    
    def _cost_8cN(self, Z, nines, N, sim_params, attention_weights):
        '''Cost associated with an ordered 8 chooose N display.
        '''
        # Similarity
        Sqa = self.similarity(tf.gather(Z, nines[:,0]), 
        tf.gather(Z, nines[:,1]), sim_params, attention_weights)
        Sqb = self.similarity(tf.gather(Z, nines[:,0]), 
        tf.gather(Z, nines[:,2]), sim_params, attention_weights)
        Sqc = self.similarity(tf.gather(Z, nines[:,0]), 
        tf.gather(Z, nines[:,3]), sim_params, attention_weights)
        Sqd = self.similarity(tf.gather(Z, nines[:,0]), 
        tf.gather(Z, nines[:,4]), sim_params, attention_weights)
        Sqe = self.similarity(tf.gather(Z, nines[:,0]), 
        tf.gather(Z, nines[:,5]), sim_params, attention_weights)
        Sqf = self.similarity(tf.gather(Z, nines[:,0]), 
        tf.gather(Z, nines[:,6]), sim_params, attention_weights)
        Sqg = self.similarity(tf.gather(Z, nines[:,0]), 
        tf.gather(Z, nines[:,7]), sim_params, attention_weights)
        Sqh = self.similarity(tf.gather(Z, nines[:,0]), 
        tf.gather(Z, nines[:,8]), sim_params, attention_weights)

        # Probility of behavior
        def f2(): return (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh))
        def f3(): return (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh))
        def f4(): return (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh))
        def f5(): return (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqe / (Sqe + Sqf + Sqg + Sqh))
        def f6(): return (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqe / (Sqe + Sqf + Sqg + Sqh)) \
            * (Sqf / (Sqf + Sqg + Sqh))
        def f7(): return (Sqa / (Sqa + Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqb / (Sqb + Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqc / (Sqc + Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqd / (Sqd + Sqe + Sqf + Sqg + Sqh)) \
            * (Sqe / (Sqe + Sqf + Sqg + Sqh)) \
            * (Sqf / (Sqf + Sqg + Sqh)) \
            * (Sqg / (Sqg + Sqh))
            
        P = tf.case({tf.equal(N, tf.constant(2)): f2, 
            tf.equal(N, tf.constant(3)): f3, 
            tf.equal(N, tf.constant(4)): f4, 
            tf.equal(N, tf.constant(5)): f5, 
            tf.equal(N, tf.constant(6)): f6, 
            tf.equal(N, tf.constant(7)): f7, 
            tf.equal(N, tf.constant(8)): f7}, default=f2, exclusive=True)

        # Cost function
        cap = tf.constant(2.2204e-16)
        J = tf.negative(tf.reduce_sum(tf.log(tf.maximum(P, cap))))
        return J

    def _core_model(self):
        '''TODO
        '''
        with tf.variable_scope("model") as scope:
            # Similarity function variables            
            (sim_params, sim_constraints) = self._get_similarity_parameters()
            attention_weights = self._get_attention_weights()
            Z = self._get_embedding()

            # scope.reuse_variables() TODO

            tf_displays = tf.placeholder(tf.int32, [None, 9], name='displays')
            tf_n_reference = tf.placeholder(tf.int32, name='n_reference')
            tf_n_selected = tf.placeholder(tf.int32, name='n_selected')
            tf_is_ranked = tf.placeholder(tf.int32, name='is_ranked')
            tf_group_id = tf.placeholder(tf.int32, name='group_id')

            # Get indices of different display configurations
            idx_8c2 = tf.squeeze(tf.where(tf.logical_and(
                tf.equal(tf_n_reference, tf.constant(8)), 
                tf.equal(tf_n_selected, tf.constant(2)))))
            idx_2c1 = tf.squeeze(tf.where(tf.equal(tf_n_reference, tf.constant(2))))

            # Get displays
            disp_8c2 = tf.gather(tf_displays, idx_8c2)
            
            disp_2c1 = tf.gather(tf_displays, idx_2c1)
            disp_2c1 = disp_2c1[:, 0:3]

            # Expand attention weights
            group_idx_2c1 = tf.gather(tf_group_id, idx_2c1)
            group_idx_2c1 = tf.reshape(group_idx_2c1, [tf.shape(group_idx_2c1)[0],1])
            weights_2c1 = tf.gather_nd(attention_weights, group_idx_2c1)
            group_idx_8c2 = tf.gather(tf_group_id, idx_8c2)
            group_idx_8c2 = tf.reshape(group_idx_8c2, [tf.shape(group_idx_8c2)[0],1])
            weights_8c2 = tf.gather_nd(attention_weights, group_idx_8c2)            

            # Cost function
            J = (self._cost_2c1(Z, disp_2c1, sim_params, weights_2c1) + 
            self._cost_8cN(Z, disp_8c2, tf.constant(2), sim_params, weights_8c2))

            # TODO constraint_weights
            # constraint_weights = attention_weights.assign(self._project_attention_weights(attention_weights))

        return (J, Z, attention_weights, sim_params, sim_constraints, tf_displays, tf_n_reference, tf_n_selected, tf_is_ranked, tf_group_id)
    
    def _concrete_evaluate(self, obs):
        '''Evaluate observations using the current state of the embedding object.
        TODO merge with evaluate method?
        '''

        # Is this really necessary?
        old_do_reuse = self.do_reuse
        old_init_scale = self.init_scale

        self.do_reuse = True
        self.init_scale = 0.
        # old_infer_Z = self.infer_Z
        # old_infer_attention_weighst = self.infer_attention_weights
        # old_infer_rho = self.infer_rho
        # old_infer_tau = self.infer_tau
        # old_infer_beta = self.infer_beta
        # old_infer_gamma = self.infer_gamma

        # self.infer_Z = False
        # self.infer_attention_weights = False
        # self.infer_rho = False
        # self.infer_tau = False
        # self.infer_beta = False
        # self.infer_gamma = False
        
        (J, _, _, _, _, tf_displays, tf_n_reference, tf_n_selected, 
        tf_is_ranked, tf_group_id) = self._core_model()

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        J_all =  sess.run(J, feed_dict={
                tf_displays: obs.displays, 
                tf_n_reference: obs.n_reference,
                tf_n_selected: obs.n_selected, 
                tf_is_ranked: obs.is_ranked, 
                tf_group_id: obs.group_id})

        sess.close()
        tf.reset_default_graph()
        
        self.do_reuse = old_do_reuse
        self.init_scale = old_init_scale
        # self.infer_Z = old_infer_Z
        # self.infer_attention_weights = old_infer_attention_weighst
        # self.infer_rho = old_infer_rho
        # self.infer_tau = old_infer_tau
        # self.infer_beta = old_infer_beta
        # self.infer_gamma = old_infer_gamma

        return J_all


class Exponential(PsychologicalEmbedding):
    '''An exponential family stochastic display embedding algorithm. 
    
    This embedding technique uses the following similarity kernel: 
      s(x,y) = exp(-beta .* norm(x - y, rho).^tau) + gamma, 
    where x and y are n-dimensional vectors. The similarity function has four 
    free parameters: rho, tau, gamma, and beta. The exponential family is 
    obtained by integrating across various psychological theores [1,2,3,4].

    References:
    [1] Jones, M., Love, B. C., & Maddox, W. T. (2006). Recency effects as a 
      window to generalization: Separating decisional and perceptual sequential
      effects in category learning. Journal of Experimental Psychology: 
      Learning, Memory, & Cognition, 32 , 316-332.
    [2] Jones, M., Maddox, W. T., & Love, B. C. (2006). The role of similarity 
      in generalization. In Proceedings of the 28th annual meeting of the 
      cognitive science society (pp. 405-410). 
    [3] Nosofsky, R. M. (1986). Attention, similarity, and the identication-
      categorization relationship. Journal of Experimental Psychology: General,
      115, 39-57.
    [4] Shepard, R. N. (1987). Toward a universal law of generalization for 
      psychological science. Science, 237, 1317-1323.
    '''

    def __init__(self, n_stimuli, dimensionality=2, n_group=1):

        '''
        Initialize

        Parameters:
          n_stimuli: An integer indicating the total number of unique stimuli
            that will be embedded.
          dimensionality: An integer indicating the dimensionalty of the 
            embedding. The dimensionality can be inferred using the function
            "suggest_dimensionality".
          n_group: (default: 1) An integer indicating the number of different
            population groups in the embedding. A separate set of attention 
            weights will be inferred for each group.
        '''

        PsychologicalEmbedding.__init__(self, n_stimuli, dimensionality, n_group)
        
        # Default parameter settings.
        self.rho = 2.
        self.tau = 1.
        self.gamma = 0.
        self.beta = 10.
        
        # Default inference settings.
        self.infer_rho = True
        self.infer_tau = True
        self.infer_gamma = True
        self.infer_beta = True
        self.lr = 0.00001
        self.max_n_epoch = 2000
        self.patience = 10
    
    def _get_similarity_parameters(self):
        '''Returns a dictionary and TensorFlow operation.
        
        This method encapsulates the creation of algorithm-specific free
        parameters governing the similarity kernel.

        Returns:
          sim_params: A dictionary of algorithm-specific TensorFlow variables.
          sim_constraints: A TensorFlow operation that imposes boundary 
            constraints on the algorithm-specific free parameters during
            inference.
        '''
        with tf.variable_scope("similarity_params"):
            if self.do_reuse:
                rho = tf.get_variable("rho", [1], initializer=tf.constant_initializer(self.rho), trainable=True)
                tau = tf.get_variable("tau", [1], initializer=tf.constant_initializer(self.tau), trainable=True)
                gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(self.gamma), trainable=True)
                beta = tf.get_variable("beta", [1], initializer=tf.constant_initializer(self.beta), trainable=True)
            else:
                if self.infer_rho:
                    rho = tf.get_variable("rho", [1], initializer=tf.random_uniform_initializer(1.,3.))
                else:
                    rho = tf.get_variable("rho", [1], initializer=tf.constant_initializer(self.rho), trainable=False)
                if self.infer_tau:
                    tau = tf.get_variable("tau", [1], initializer=tf.random_uniform_initializer(1.,2.))
                else:
                    tau = tf.get_variable("tau", [1], initializer=tf.constant_initializer(self.tau), trainable=False)
                if self.infer_gamma:
                    gamma = tf.get_variable("gamma", [1], initializer=tf.random_uniform_initializer(0.,.001))
                else:
                    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(self.gamma), trainable=False)
                if self.infer_beta:
                    beta = tf.get_variable("beta", [1], initializer=tf.random_uniform_initializer(1.,30.))
                else:
                    beta = tf.get_variable("beta", [1], initializer=tf.constant_initializer(self.beta), trainable=False)
        sim_params = {'rho':rho, 'tau':tau, 'gamma':gamma, 'beta':beta}

        # TensorFlow operation to enforce free parameter constraints.
        constraint_rho = rho.assign(tf.maximum(1., rho))
        constraint_tau = tau.assign(tf.maximum(1., tau))
        constraint_gamma = gamma.assign(tf.maximum(0., gamma))
        constraint_beta = beta.assign(tf.maximum(1., beta))
        sim_constraints = tf.group(constraint_rho, constraint_tau, constraint_gamma, constraint_beta)

        return (sim_params, sim_constraints)

    def _set_parameters(self, params):
        '''A state changing method that sets algorithm-specific parameters.
        
        This method encapsulates the setting of algorithm-specific free 
        parameters governing the similarity kernel.
        '''
        self.rho = params['rho']
        self.tau = params['tau']
        self.beta = params['beta']
        self.gamma = params['gamma']

    def freeze(self, rho=None, tau=None, gamma=None, beta=None, Z=None):
        '''A state changing method specifing which parameters are fixed.

        During inference, you may want to freeze some parameters at specific
        values. To freeze a particular parameter, pass in a value for it. If you
        would like to freeze multiple parameters, they must be passed in at the 
        same time. If no value is provided for a parameter, it will not be fixed.
        
        Parameters:
          rho
          tau
          gamma
          beta
          Z
        '''
        
        self.infer_rho = True
        self.infer_tau = True
        self.infer_gamma = True
        self.infer_beta = True
        self.infer_Z = True

        if rho is not None:
            self.rho = rho
            self.infer_rho = False
        if tau is not None:
            self.tau = tau
            self.infer_tau = False
        if gamma is not None:
            self.gamma = gamma
            self.infer_gamma = False
        if beta is not None:
            self.beta = beta
            self.infer_beta = False
        if Z is not None:
            self.Z = Z
            self.infer_Z = False

    def similarity(self, z_q, z_ref, sim_params, attention_weights):
        '''Exponential family similarity kernel.

        Parameters:
          z_q: A set of embedding points.
            shape = (n_sample, dimensionality)
          z_ref: A set of embedding points.
            shape = (n_sample, dimensionality)
          sim_params: A dictionary of algorithm-specific parameters governing
            the similarity kernel.
          attention_weights: The weights allocated to each dimension in a 
            weighted minkowski metric.
            shape = (n_sample, dimensionality)
        Returns:
          similarity: The corresponding similairty between rows of embedding
            points.
            shape = (n_sample,)
        '''

        # Algorithm-specific parameters governing the similarity kernel.
        rho = sim_params['rho']
        tau = sim_params['tau']
        gamma = sim_params['gamma']
        beta = sim_params['beta']

        # Weighted Minkowski distance.
        d_qref = tf.pow(tf.abs(z_q - z_ref), rho)
        d_qref = tf.multiply(d_qref, attention_weights)
        d_qref = tf.pow(tf.reduce_sum(d_qref,axis=1), 1. / rho)

        # Exponential family similarity kernel.
        s_qref = tf.exp(tf.negative(beta) * tf.pow(d_qref, tau) + gamma)
        return s_qref

class HeavyTailed(PsychologicalEmbedding):
    '''A heavy-tailed family stochastic display embedding algorithm. 
    
    This embedding technique uses the following similarity kernel: 
      s(x,y) = (kappa + (norm(x-y, rho).^tau)).^(-alpha), 
    where x and y are n-dimensional vectors. The similarity function has four 
    free parameters: rho, tau, kappa, and alpha. The heavy-tailed family is a 
    further generalization of the Student-t family.
    '''

    def __init__(self, n_stimuli, dimensionality=2, n_group=1):

        '''
        Initialize

        Parameters:
          n_stimuli: An integer indicating the total number of unique stimuli
            that will be embedded.
          dimensionality: An integer indicating the dimensionalty of the 
            embedding. The dimensionality can be inferred using the function
            "suggest_dimensionality".
          n_group: (default: 1) An integer indicating the number of different
            population groups in the embedding. A separate set of attention 
            weights will be inferred for each group.
        '''

        PsychologicalEmbedding.__init__(self, n_stimuli, dimensionality, n_group)
        
        # Default parameter settings.
        self.rho = 2.
        self.tau = 1.
        self.kappa = 2.
        self.alpha = 30.
        
        # Default inference settings.
        self.infer_rho = True
        self.infer_tau = True
        self.infer_kappa = True
        self.infer_alpha = True
        self.lr = 0.00001
        self.max_n_epoch = 2000
        self.patience = 10
    
    def _get_similarity_parameters(self):
        '''Returns a dictionary and TensorFlow operation.
        
        This method encapsulates the creation of algorithm-specific free
        parameters governing the similarity kernel.

        Returns:
          sim_params: A dictionary of algorithm-specific TensorFlow variables.
          sim_constraints: A TensorFlow operation that imposes boundary 
            constraints on the algorithm-specific free parameters during
            inference.
        '''
        with tf.variable_scope("similarity_params"):
            if self.do_reuse:
                rho = tf.get_variable("rho", [1], initializer=tf.constant_initializer(self.rho), trainable=True)
                tau = tf.get_variable("tau", [1], initializer=tf.constant_initializer(self.tau), trainable=True)
                kappa = tf.get_variable("kappa", [1], initializer=tf.constant_initializer(self.kappa), trainable=True)
                alpha = tf.get_variable("alpha", [1], initializer=tf.constant_initializer(self.alpha), trainable=True)
            else:
                if self.infer_rho:
                    rho = tf.get_variable("rho", [1], initializer=tf.random_uniform_initializer(1.,3.))
                else:
                    rho = tf.get_variable("rho", [1], initializer=tf.constant_initializer(self.rho), trainable=False)
                if self.infer_tau:
                    tau = tf.get_variable("tau", [1], initializer=tf.random_uniform_initializer(1.,2.))
                else:
                    tau = tf.get_variable("tau", [1], initializer=tf.constant_initializer(self.tau), trainable=False)
                if self.infer_kappa:
                    kappa = tf.get_variable("kappa", [1], initializer=tf.random_uniform_initializer(1.,11.))
                else:
                    kappa = tf.get_variable("kappa", [1], initializer=tf.constant_initializer(self.kappa), trainable=False)
                if self.infer_alpha:
                    alpha = tf.get_variable("alpha", [1], initializer=tf.random_uniform_initializer(10.,60.))
                else:
                    alpha = tf.get_variable("alpha", [1], initializer=tf.constant_initializer(self.alpha), trainable=False)
        sim_params = {'rho':rho, 'tau':tau, 'kappa':kappa, 'alpha':alpha}

        # TensorFlow operation to enforce free parameter constraints.
        constraint_rho = rho.assign(tf.maximum(1., rho))
        constraint_tau = tau.assign(tf.maximum(1., tau))
        constraint_kappa = kappa.assign(tf.maximum(0., kappa))
        constraint_alpha = alpha.assign(tf.maximum(0., alpha))
        sim_constraints = tf.group(constraint_rho, constraint_tau, constraint_kappa, constraint_alpha)

        return (sim_params, sim_constraints)

    def _set_parameters(self, params):
        '''A state changing method that sets algorithm-specific parameters.
        
        This method encapsulates the setting of algorithm-specific free 
        parameters governing the similarity kernel.
        '''
        self.rho = params['rho']
        self.tau = params['tau']
        self.kappa = params['kappa']
        self.alpha = params['alpha']

    def freeze(self, rho=None, tau=None, kappa=None, alpha=None, Z=None):
        '''A state changing method specifing which parameters are fixed.

        During inference, you may want to freeze some parameters at specific
        values. To freeze a particular parameter, pass in a value for it. If you
        would like to freeze multiple parameters, they must be passed in at the 
        same time. If no value is provided for a parameter, it will not be fixed.
        
        Parameters:
          rho
          tau
          kappa
          alpha
          Z
        '''

        self.infer_rho = True
        self.infer_tau = True
        self.infer_kappa = True
        self.infer_alpha = True
        self.infer_Z = True

        if rho is not None:
            self.rho = rho
            self.infer_rho = False            
        if tau is not None:
            self.tau = tau
            self.infer_tau = False            
        if kappa is not None:
            self.kappa = kappa
            self.infer_kappa = False            
        if alpha is not None:
            self.alpha = alpha
            self.infer_alpha = False            
        if Z is not None:
            self.Z = Z
            self.infer_Z = False        

    def similarity(self, z_q, z_ref, sim_params, attention_weights):
        '''Heavy-tailed family similarity kernel.

        Parameters:
          z_q: A set of embedding points.
            shape = (n_sample, dimensionality)
          z_ref: A set of embedding points.
            shape = (n_sample, dimensionality)
          sim_params: A dictionary of algorithm-specific parameters governing
            the similarity kernel.
          attention_weights: The weights allocated to each dimension in a 
            weighted minkowski metric.
            shape = (n_sample, dimensionality)
        Returns:
          similarity: The corresponding similairty between rows of embedding
            points.
            shape = (n_sample,)
        '''

        # Algorithm-specific parameters governing the similarity kernel.
        rho = sim_params['rho']
        tau = sim_params['tau']
        kappa = sim_params['kappa']
        alpha = sim_params['alpha']

        # Weighted Minkowski distance.
        d_qref = tf.pow(tf.abs(z_q - z_ref), rho)
        d_qref = tf.multiply(d_qref, attention_weights)
        d_qref = tf.pow(tf.reduce_sum(d_qref,axis=1), 1. / rho)

        # Heavy-tailed family similarity kernel.
        s_qref = tf.pow(kappa + tf.pow(d_qref, tau), (tf.negative(alpha)))
        return s_qref

class StudentT(PsychologicalEmbedding):
    '''A Student-t family stochastic display embedding algorithm.

    The embedding technique uses the following simialrity kernel:
      s(x,y) = (1 + (((norm(x-y, rho)^tau) / alpha))^(-(alpha + 1)/2),
    where x and y are n-dimensional vectors. The similarity kernel has three 
    free parameters: rho, tau, and alpha. The original Student-t kernel was
    was proposed by van der Maaten [1] to handle "triplet" similarity 
    judgments. The similarity kernel used here is a slight generalization of
    the orignal case where rho=2, tau=2, and alpha=dimensionality-1.

    References:
      [1] van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic triplet 
        embedding. In Machine learning for signal processing (mlsp), 2012 IEEE
        international workshop on (p. 1-6). doi:10.1109/MLSP.2012.6349720
    '''

    def __init__(self, n_stimuli, dimensionality=2, n_group=1):

        '''
        Initialize

        Parameters:
          n_stimuli: An integer indicating the total number of unique stimuli
            that will be embedded.
          dimensionality: An integer indicating the dimensionalty of the 
            embedding. The dimensionality can be inferred using the function
            "suggest_dimensionality".
          n_group: (default: 1) An integer indicating the number of different
            population groups in the embedding. A separate set of attention 
            weights will be inferred for each group.
        '''

        PsychologicalEmbedding.__init__(self, n_stimuli, dimensionality, n_group)
        
        # Default parameter settings.
        self.rho = 2.
        self.tau = 2.
        self.alpha = 5.
        
        # Default inference settings.
        self.infer_rho = True
        self.infer_tau = True
        self.infer_alpha = True
        self.lr = 0.00001
        self.max_n_epoch = 2000
        self.patience = 10

    def _get_similarity_parameters(self):
        '''Returns a dictionary and TensorFlow operation.
        
        This method encapsulates the creation of algorithm-specific free
        parameters governing the similarity kernel.

        Returns:
          sim_params: A dictionary of algorithm-specific TensorFlow variables.
          sim_constraints: A TensorFlow operation that imposes boundary 
            constraints on the algorithm-specific free parameters during
            inference.
        '''
        with tf.variable_scope("similarity_params"):
            if self.do_reuse:
                rho = tf.get_variable("rho", [1], initializer=tf.constant_initializer(self.rho), trainable=True)
                tau = tf.get_variable("tau", [1], initializer=tf.constant_initializer(self.tau), trainable=True)
                alpha = tf.get_variable("alpha", [1], initializer=tf.constant_initializer(self.alpha), trainable=True)
            else:
                if self.infer_rho:
                    rho = tf.get_variable("rho", [1], initializer=tf.random_uniform_initializer(1.,3.))
                else:
                    rho = tf.get_variable("rho", [1], initializer=tf.constant_initializer(self.rho), trainable=False)
                if self.infer_tau:
                    tau = tf.get_variable("tau", [1], initializer=tf.random_uniform_initializer(1.,2.))
                else:
                    tau = tf.get_variable("tau", [1], initializer=tf.constant_initializer(self.tau), trainable=False)
                if self.infer_alpha:
                    alpha = tf.get_variable("alpha", [1], initializer=tf.random_uniform_initializer(1.,30.))
                else:
                    alpha = tf.get_variable("alpha", [1], initializer=tf.constant_initializer(self.alpha), trainable=False)
        sim_params = {'rho':rho, 'tau':tau, 'alpha':alpha}

        # TensorFlow operation to enforce free parameter constraints.
        eps = tf.constant(.000001)
        constraint_rho = rho.assign(tf.maximum(1., rho))
        constraint_tau = tau.assign(tf.maximum(1., tau))
        constraint_alpha = alpha.assign(tf.maximum(eps, alpha))
        sim_constraints = tf.group(constraint_rho, constraint_tau, constraint_alpha)

        return (sim_params, sim_constraints)

    def _set_parameters(self, params):
        '''A state changing method that sets algorithm-specific parameters.
        
        This method encapsulates the setting of algorithm-specific free 
        parameters governing the similarity kernel.
        '''
        self.rho = params['rho']
        self.tau = params['tau']
        self.alpha = params['alpha']

    def freeze(self, rho=None, tau=None, alpha=None, Z=None):
        '''A state changing method specifing which parameters are fixed.

        During inference, you may want to freeze some parameters at specific
        values. To freeze a particular parameter, pass in a value for it. If you
        would like to freeze multiple parameters, they must be passed in at the 
        same time. If no value is provided for a parameter, it will not be fixed.
        
        Parameters:
          rho
          tau
          alpha
          Z
        '''

        self.infer_rho = True
        self.infer_tau = True
        self.infer_alpha = True
        self.infer_Z = True

        if rho is not None:
            self.rho = rho
            self.infer_rho = False
        if tau is not None:
            self.tau = tau
            self.infer_tau = False
        if alpha is not None:
            self.alpha = alpha
            self.infer_alpha = False
        if Z is not None:
            self.Z = Z
            self.infer_Z = False

    def similarity(self, z_q, z_ref, sim_params, attention_weights):
        '''Student-t family similarity kernel.

        Parameters:
          z_q: A set of embedding points.
            shape = (n_sample, dimensionality)
          z_ref: A set of embedding points.
            shape = (n_sample, dimensionality)
          sim_params: A dictionary of algorithm-specific parameters governing
            the similarity kernel.
          attention_weights: The weights allocated to each dimension in a 
            weighted minkowski metric.
            shape = (n_sample, dimensionality)
        Returns:
          similarity: The corresponding similairty between rows of embedding
            points.
            shape = (n_sample,)
        '''

        # Algorithm-specific parameters governing the similarity kernel.
        rho = sim_params['rho']
        tau = sim_params['tau']
        alpha = sim_params['alpha']

        # Weighted Minkowski distance.
        d_qref = tf.pow(tf.abs(z_q - z_ref), rho)
        d_qref = tf.multiply(d_qref, attention_weights)
        d_qref = tf.pow(tf.reduce_sum(d_qref,axis=1), 1. / rho)

        # Student-t family similarity kernel.
        s_qref = tf.pow(1 + (tf.pow(d_qref, tau) / alpha), tf.negative(alpha + 1)/2)
        return s_qref
