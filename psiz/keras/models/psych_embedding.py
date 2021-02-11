# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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
# ============================================================================
"""Module of psychological embedding models.

Classes:
    PsychologicalEmbedding: Abstract base class for a psychological
        embedding model.

Functions:
    load_model: Load a hdf5 file, that was saved with the `save`
        class method, as a PsychologicalEmbedding object.

"""

import copy
import json
import os
from pathlib import Path
import warnings

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
import tensorflow_probability as tfp

import psiz.keras.layers
import psiz.trials


class PsychologicalEmbedding(tf.keras.Model):
    """A pscyhological embedding model.

    This model can be subclassed to infer a psychological embedding
    from different types similarity judgment data.

    Attributes:
        embedding: An embedding layer.
        kernel: A kernel layer.
        behavior: A behavior layer.
        n_stimuli: The number of stimuli.
        n_dim: The dimensionality of the embedding.
        n_sample: The number of samples to draw on probalistic layers.
            This attribute is only advantageous if using probabilistic
            layers.

    """

    def __init__(
            self, stimuli=None, kernel=None, behavior=None, n_sample=1,
            **kwargs):
        """Initialize.

        Arguments:
            stimuli: An embedding layer representing the stimuli. Must
                agree with n_stimuli, n_dim, n_group.
            kernel (optional): A kernel layer.
            behavior (optional): A behavior layer.
            n_sample (optional): An integer indicating the
                number of samples to use on the forward pass. This
                argument is only advantageous if using stochastic
                layers (e.g., variational models).
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super().__init__(**kwargs)

        # Assign layers.
        self.stimuli = stimuli
        self.kernel = kernel
        self.behavior = behavior

        self._kl_weight = 0.
        self._n_sample = n_sample

    @property
    def n_stimuli(self):
        """Convenience method for `n_stimuli`."""
        try:
            # Assume Stimuli layer.
            n_stimuli = self.stimuli.input_dim
            if self.stimuli.mask_zero:
                n_stimuli -= 1
        except AttributeError:
            try:
                # Assume peel.
                n_stimuli = self.stimuli.net.input_dim
                if self.stimuli.net.mask_zero:
                    n_stimuli -= 1
            except AttributeError:
                # Assume gate.
                n_stimuli = self.stimuli.subnets[0].input_dim
                if self.stimuli.subnets[0].mask_zero:
                    n_stimuli -= 1

        return n_stimuli

    @property
    def n_dim(self):
        """Convenience method for `n_dim`."""
        try:
            # Assume Stimuli layer.
            output_dim = self.stimuli.output_dim
        except AttributeError:
            try:
                # Assume peel.
                output_dim = self.stimuli.net.output_dim
            except AttributeError:
                # Assume gate.
                output_dim = self.stimuli.subnets[0].output_dim

        return output_dim

    @property
    def n_sample(self):
        return self._n_sample

    @n_sample.setter
    def n_sample(self, n_sample):
        self._n_sample = n_sample

    @property
    def kl_weight(self):
        return self._kl_weight

    @kl_weight.setter
    def kl_weight(self, kl_weight):
        self._kl_weight = kl_weight
        _submodule_setattr(self.layers, 'kl_weight', kl_weight)

    def train_step(self, data):
        """Logic for one training step.

        Arguments:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`.
            Typically, the values of the `Model`'s metrics are
            returned. Example: `{'loss': 0.2, 'accuracy': 0.7}`.

        Notes:
            It is assumed that the loss uses SUM_OVER_BATCH_SIZE.
            In the variational inference case, the loss for the entire
            training set is essentially

                loss = KL - CCE,                                (Eq. 1)

            where CCE denotes the sum of the CCE for all observations.
            It should be noted that CCE_all is also an expectation over
            posterior samples.

            There are two issues:

            1) We are using a *batch* update strategy, which slightly
            alters the equation and means we need to be careful not to
            overcount the KL contribution. The `b` subscript indicates
            an arbitrary batch:

                loss_b = (KL / n_batch) - CCE_b.                (Eq. 2)

            2) The default TF reduction strategy `SUM_OVER_BATCH_SIZE`
            means that we are not actually computing a sum `CCE_b`, but
            an average: CCE_bavg = CCE_b / batch_size. To fix this, we
            need to proportionately scale the KL term,

                loss_b = KL / (n_batch * batch_size) - CCE_bavg (Eq. 3)

            Expressed more simply,

            loss_batch = kl_weight * KL - CCE_bavg              (Eq. 4)

            where kl_weight = 1 / train_size.

            But wait, there's more! Observations may be weighted
            differently, which yields a Frankensteinian CCE_bavg since
            a proper average would divide by `effective_batch_size`
            (i.e., the sum of the weights) not `batch_size`. There are
            a few imperfect remedies:
            1) Do not use `SUM_OVER_BATCH_SIZE`. This has many
            side-effects: must manually handle regularization and
            computation of mean loss. Mean loss is desirable for
            optimization stability reasons, although it is not strictly
            necessary.
            2) Require the weights sum to n_sample. Close, but
            not actually correct. To be correct you would actually
            need the weights of each batch to sum to `batch_size`,
            which means the semantics of the weights changes from batch
            to batch.
            3) Multiply Eq. 3 by (batch_size / effective_batch_size).
            This is tricky since this must be done before non-KL
            regularization is applied, which is handled inside
            TF's `compiled_loss`. Could hack this by writing a custom
            CCE that "pre-applies" correction term.

                loss_b = KL / (n_batch * effective_batch_size) -
                        (batch_size / effective_batch_size) * CCE_bavg

                loss_b = KL / (effective_train_size) -
                        (batch_size / effective_batch_size) * CCE_bavg

            4) Pretend it's not a problem since both terms are being
            divided by the same incorrect `effective_batch_size`.

        """

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # NOTE: During computation of gradients, IndexedSlices are
        # created which generates a TensorFlow warning. I cannot
        # find an implementation that avoids IndexedSlices. The
        # following catch environment silences the offending
        # warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, module=r'.*indexed_slices'
            )
            with backprop.GradientTape() as tape:
                # Average over samples.
                y_pred = tf.reduce_mean(self(x, training=True), axis=1)
                loss = self.compiled_loss(
                    y, y_pred, sample_weight, regularization_losses=self.losses
                )

            # Custom training steps:
            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            # NOTE: There is an open issue for using constraints with
            # embedding-like layers (e.g., tf.keras.layers.Embedding)
            # see:
            # https://github.com/tensorflow/tensorflow/issues/33755.
            # There are also issues when using Eager Execution. A
            # work-around is to convert the problematic gradients, which
            # are returned as tf.IndexedSlices, into dense tensors.
            for idx, grad in enumerate(gradients):
                if gradients[idx].__class__.__name__ == 'IndexedSlices':
                    gradients[idx] = tf.convert_to_tensor(
                        gradients[idx]
                    )

        self.optimizer.apply_gradients(zip(
            gradients, trainable_variables)
        )

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """The logic for one evaluation step.

        Standard prediction is performmed with one sample. To
        accommodate variational inference, the log probability of the
        data is computed by averaging over samples from the model:
        p(heldout | train) = int_model p(heldout|model) p(model|train)
                          ~= 1/n * sum_{i=1}^n p(heldout | model_i)
        where model_i is a draw from the posterior p(model|train).

        Arguments:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`.
            Typically, the values of the `Model`'s metrics are
            returned.

        """
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # NOTE The first dimension of the Tensor returned from calling the
        # model is assumed to be `sample_size`. If this is a singleton
        # dimension, taking the mean is equivalent to a squeeze
        # operation.
        y_pred = tf.reduce_mean(self(x, training=False), axis=1)
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses
        )
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        """The logic for one inference step.

        Standard prediction is performmed with one sample. To
        accommodate variational inference, the predictions are averaged
        over multiple samples from the model.

        Arguments:
            data: A nested structure of `Tensor`s.

        Returns:
            The result of one inference step, typically the output of
            calling the `Model` on data.

        """
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = tf.reduce_mean(self(x, training=False), axis=1)
        return y_pred

    def get_config(self):
        """Return model configuration."""
        layer_configs = {
            'stimuli': tf.keras.utils.serialize_keras_object(
                self.stimuli
            ),
            'kernel': tf.keras.utils.serialize_keras_object(
                self.kernel
            ),
            'behavior': tf.keras.utils.serialize_keras_object(self.behavior)
        }

        config = {
            'name': self.name,
            'class_name': self.__class__.__name__,
            'n_sample': self.n_sample,
            'layers': copy.deepcopy(layer_configs)
        }
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration.

        Arguments:
            config: A hierarchical configuration dictionary.

        Returns:
            An instantiated and configured TensorFlow model.

        """
        model_config = copy.deepcopy(config)
        model_config.pop('class_name', None)

        # Deserialize layers.
        layer_configs = model_config.pop('layers', None)
        built_layers = {}
        for layer_name, layer_config in layer_configs.items():
            layer = tf.keras.layers.deserialize(layer_config)
            # Convert old saved models.
            if layer_name == 'embedding':
                layer_name = 'stimuli'
            built_layers[layer_name] = layer

        model_config.update(built_layers)
        return cls(**model_config)

    # Deprecated in favor of TF 2.4 default `save` method. This method was
    # previousely called `save`.
    def save_model(self, filepath, overwrite=False):
        """Save the PsychologialEmbedding model.

        Ideally we would use TensorFlow model's defualt save
        method. Unfortunately, this does not always work for the
        sub-classed model appraoch. A custom implementation is used
        here.

        Arguments:
            filepath: String specifying the path to save the model.
            overwrite (optional): Whether to overwrite exisitng files.

        """
        # Make directory.
        Path(filepath).mkdir(parents=True, exist_ok=overwrite)

        fp_config = os.path.join(filepath, 'config.h5')
        fp_weights = os.path.join(filepath, 'weights')

        # Ideally, we could guarantee types during get_config() method call,
        # making this recursive check unnecessary.
        def _convert_to_64(d):
            for k, v in d.items():
                if isinstance(v, np.float32):
                    if 'name' in d:
                        component_name = d['name'] + ':' + k
                    else:
                        component_name = k
                    print(
                        'WARNING: Model component `{0}` had type float32.'
                        ' Please check the corresponding get_config method for'
                        ' appropriate float casting.'.format(component_name)
                    )
                    d[k] = float(v)
                if isinstance(v, np.int64):
                    if 'name' in d:
                        component_name = d['name'] + ':' + k
                    else:
                        component_name = k
                    print(
                        'WARNING: Model component `{0}` had type int64. Please'
                        ' check the corresponding get_config method for'
                        ' appropriate int casting.'.format(component_name)
                    )
                    d[k] = int(v)
                elif isinstance(v, dict):
                    d[k] = _convert_to_64(v)
            return d

        # Save configuration.
        model_config = self.get_config()
        model_config = _convert_to_64(model_config)
        json_model_config = json.dumps(model_config)
        json_loss_config = json.dumps(
            tf.keras.losses.serialize(self.loss)
        )
        optimizer_config = tf.keras.optimizers.serialize(self.optimizer)
        # HACK: numpy.float32 is not serializable
        for k, v in optimizer_config['config'].items():
            if isinstance(v, np.float32):
                optimizer_config['config'][k] = float(v)
        json_optimizer_config = json.dumps(optimizer_config)
        f = h5py.File(fp_config, "w")
        f.create_dataset('model_type', data='psiz')
        f.create_dataset('psiz_version', data='0.4.0')
        f.create_dataset('config', data=json_model_config)
        f.create_dataset('loss', data=json_loss_config)
        f.create_dataset('optimizer', data=json_optimizer_config)
        f.close()

        # Save weights.
        self.save_weights(
            fp_weights, overwrite=overwrite, save_format='tf'
        )


# This function is deprecated and should only be used to load old models.
def load_model(filepath, custom_objects={}, compile=False):
    """Load embedding model saved via the save method.

    Arguments:
        filepath: The location of the hdf5 file to load.
        custom_objects (optional): A dictionary mapping the string
            class name to the Python class
        compile: Boolean indicating if model should be compiled.

    Returns:
        A TensorFlow Keras model.

    Raises:
        ValueError

    """
    # Check if directory.
    if os.path.isdir(filepath):
        # NOTE: Ideally we could call TensorFlow's `load_model` method as used
        # in the snippet below, but our sub-classed model does not play
        # nice.
        #
        # with tf.keras.utils.custom_object_scope(custom_objects):
        #     model = tf.keras.models.load_model(filepath, compile=compile)
        # return model
        #
        # Storage format for psiz_version == 0.4.0
        fp_config = os.path.join(filepath, 'config.h5')
        fp_weights = os.path.join(filepath, 'weights')

        # Load configuration.
        f = h5py.File(fp_config, 'r')
        psiz_version = f['psiz_version'][()]
        config = json.loads(f['config'][()])
        loss_config = json.loads(f['loss'][()])
        optimizer_config = json.loads(f['optimizer'][()])

        model_class_name = config.get('class_name')
        # Load model.
        if model_class_name in custom_objects:
            model_class = custom_objects[model_class_name]
        else:
            model_class = getattr(psiz.keras.models, model_class_name)
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = model_class.from_config(config)
            loss = tf.keras.losses.deserialize(loss_config)
            optimizer = tf.keras.optimizers.deserialize(optimizer_config)

        if compile:
            model.compile(loss=loss, optimizer=optimizer)
        # Build lazy layers in order to set weights.
        model.stimuli.build(input_shape=[None, None, None])

        # Load weights.
        model.load_weights(fp_weights).expect_partial()

    else:
        raise ValueError(
            'The argument `filepath` must be a directory. The provided'
            ' {0} is not a directory.'.format(filepath)
        )

    return model


def _submodule_setattr(layers, attribute, val):
    """Iterate over layers and submodules to set attribute."""
    for layer in layers:
        for sub in layer.submodules:
            if hasattr(sub, attribute):
                setattr(sub, attribute, val)
