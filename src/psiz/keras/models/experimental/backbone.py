# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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
    Backbone:  A backbone-based psychological embedding model.

"""

import copy
from importlib.metadata import version
import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.eager import backprop

from psiz.keras.layers.experimental.braid_gate import BraidGate
from psiz.utils import expand_dim_repeat


class Backbone(tf.keras.Model):
    """A backbone-based psychological embedding model.

    This model can be subclassed to infer a psychological embedding
    from different types behavioral data.

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

        Args:
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

        # TODO could change Backbone signature such that use specifies on the module whether to `use_group_xxx`, easy for kernel and behavior, maybe tricky for stimuli
        # TODO this won't work for list.
        # self.behavior.kernel = kernel  # TODO is this ok, should I use setter for better protection?
        # self.behavior.use_group = use_group_behavior
        # PROBLEM: groups needs to be handled outside stimuli,kernel,behavior layer since those layers are blind to group info

        # Handle module switches.
        def supports_groups(layer):
            if hasattr(layer, 'supports_groups'):
                return layer.supports_groups
            else:
                return False

        self._pass_groups = {
            'stimuli': supports_groups(stimuli),
            'kernel': supports_groups(kernel),
            'behavior': supports_groups(behavior)
        }

        self._kl_weight = 0.
        self._n_sample = n_sample

    @property
    def n_dim(self):
        """Convenience method for `n_dim`."""
        try:
            # Assume embedding layer.
            output_dim = self.stimuli.output_dim
        except AttributeError:
            # Assume BraidGate layer.
            output_dim = self.stimuli.subnets[0].output_dim

        return output_dim

    @property
    def n_sample(self):
        return self._n_sample

    @n_sample.setter
    def n_sample(self, n_sample):
        self._n_sample = n_sample

    def call(self, inputs):
        """Call.

        Args:
            inputs: A dictionary of inputs. At a minimum, must contain:
                stimulus_set: dtype=tf.int32, consisting of the
                    integers on the interval [0, n_stimuli[
                    shape=(batch_size, n_max_reference + 1, n_outcome)
                groups: dtype=tf.int32, Integers indicating the
                    group membership of a trial.
                    shape=(batch_size, k)

        """
        # Grab universal inputs.
        stimulus_set = inputs['stimulus_set']
        groups = inputs['groups']

        # Repeat `stimulus_set` `n_sample` times in a newly inserted
        # axis (axis=1).
        stimulus_set = expand_dim_repeat(
            stimulus_set, self.n_sample, axis=1
        )
        # TensorShape([batch_size, n_sample, n_ref + 1, n_outcome])  TODO
        # Update `stimulus_set` key.
        inputs['stimulus_set'] = stimulus_set

        # Enbed stimuli indices in n-dimensional space:
        if self._pass_groups['stimuli']:
            z = self.stimuli([stimulus_set, groups])
        else:
            z = self.stimuli(stimulus_set)
        # TensorShape([batch_size, n_sample, n_ref + 1, n_outcome, n_dim])  TODO

        # TODO from here on is very ugly.
        # * does it make sense to roll things into behavior?
        # * maybe ugly isn' a problem, so long as it works well

        # Prep retrieved embeddings for kernel op based on behavior.
        # NOTE: This assumes all subnets the same behavior, which should be
        # true.
        if self._pass_groups['behavior']:
            z_q, z_r = self.behavior.subnets[0].on_kernel_begin(z)
        else:
            z_q, z_r = self.behavior.on_kernel_begin(z)

        # Pass through similarity kernel.
        if self._pass_groups['kernel']:
            sim_qr = self.kernel([z_q, z_r, groups])
        else:
            sim_qr = self.kernel([z_q, z_r])
        # TensorShape([batch_size, sample_size, n_ref, n_outcome])  TODO

        # Given behavior, prepare retrieved embeddings for kernel op.
        # TODO this assumes all subnets the same behavior (should be true)
        # TODO this assumes nesting is only one deep (bad assumption?)
        if self._pass_groups['behavior']:
            inputs_list = self.behavior.subnets[0].format_inputs(inputs)
        else:
            inputs_list = self.behavior.format_inputs(inputs)

        # Predict behavioral outcomes.
        # TODO problem: my dispatcher can't route inputs of type dictionary
        if self._pass_groups['behavior']:
            y_pred = self.behavior([sim_qr, *inputs_list, groups])
        else:
            y_pred = self.behavior([sim_qr, *inputs_list])

        return y_pred

    def train_step(self, data):
        """Logic for one training step.

        Args:
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

        Args:
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

        Args:
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
        ver = version("psiz")
        ver = '.'.join(ver.split('.')[:3])

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
            'psiz_version': ver,
            'n_sample': self.n_sample,
            'layers': copy.deepcopy(layer_configs)
        }
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration.

        Args:
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
                layer_name = 'stimuli'  # pragma: no cover
            built_layers[layer_name] = layer

        model_config.update(built_layers)
        return cls(**model_config)
