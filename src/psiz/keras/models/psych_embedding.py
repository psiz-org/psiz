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
import warnings

import tensorflow as tf
from tensorflow.python.eager import backprop


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
            use_group_stimuli=False, use_group_kernel=False,
            use_group_behavior=False, **kwargs):
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
            use_group_stimuli (optional): Boolean indicating if group
                information should be piped to `stimuli` layer.
            use_group_kernel (optional): Boolean indicating if group
                information should be piped to `kernel` layer.
            use_group_behavior (optional): Boolean indicating if group
                information should be piped to `behavior` layer.
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super().__init__(**kwargs)

        # Assign layers.
        self.stimuli = stimuli
        self.kernel = kernel
        self.behavior = behavior

        # Handle layer switches.
        self._use_group = {
            'stimuli': use_group_stimuli,
            'kernel': use_group_kernel,
            'behavior': use_group_behavior
        }

        self._kl_weight = 0.
        self._n_sample = n_sample

    @property
    def n_stimuli(self):
        """Convenience method for `n_stimuli`."""
        try:
            n_stimuli = self.stimuli.n_stimuli
        except AttributeError:
            try:
                # Assume embedding layer.
                n_stimuli = self.stimuli.input_dim
                if self.stimuli.mask_zero:
                    n_stimuli -= 1
            except AttributeError:
                # Assume Gate or GateMulti layer.
                n_stimuli = self.stimuli.subnets[0].input_dim
                if self.stimuli.subnets[0].mask_zero:
                    n_stimuli -= 1

        return n_stimuli

    @property
    def n_dim(self):
        """Convenience method for `n_dim`."""
        try:
            # Assume embedding layer.
            output_dim = self.stimuli.output_dim
        except AttributeError:
            # Assume Gate or GateMulti layer.
            output_dim = self.stimuli.subnets[0].output_dim

        return output_dim

    @property
    def n_sample(self):
        return self._n_sample

    @n_sample.setter
    def n_sample(self, n_sample):
        self._n_sample = n_sample

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
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

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
            for idx in range(len(gradients)):
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
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
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
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
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
            'psiz_version': '0.5.0',
            'n_sample': self.n_sample,
            'layers': copy.deepcopy(layer_configs),
            'use_group_stimuli': self._use_group['stimuli'],
            'use_group_kernel': self._use_group['kernel'],
            'use_group_behavior': self._use_group['behavior'],
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
        model_config.pop('psiz_version', None)

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


def _submodule_setattr(layers, attribute, val):
    """Iterate over layers and submodules to set attribute."""
    for layer in layers:
        for sub in layer.submodules:
            if hasattr(sub, attribute):
                setattr(sub, attribute, val)
