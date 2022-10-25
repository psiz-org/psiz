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
"""Module for a Keras mixins.

Classes:
    StochasticMixin: A mixin for layers that use stochastic sampling.

"""

import tensorflow as tf


class StochasticMixin():
    """A mixin for layers that use stochastic sampling.

    A companion mixin for `psiz.keras.models.StochasticModel`.

    Attributes:
        sample_axis_outermost: Integer indicating the axis used for
            samples at the outermost level of the model. This is the
            equivalent to the `sample_axis` value used to intialize
            a `StochasticModel`.
        sample_axis: Integer indicating the axis used for samples in
            the current context. If the call enters an RNN, the axis
            used for samples will be decreased by one to reflect the
            fact that the timstep axis has been dropped.
        n_sample: The number of samples on the sample axis.
        is_inside_rnn: A Boolean indicating if the context of the call
            is inside an RNN layer.

    Notes:
        Attributes of mixin are set when `build` method of
            `StochasticModel` is called. Before `build` is called, the
            attributes will be `None`.
        You can not set `sample_axis` directly, you can only set
            `sample_axis_outermost`. This is intentional to allow
            the correct determination of the sample axis based on
            context of the call (i.e., whether the call is inside an
            RNN or not).

    """

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self._sample_axis_outermost = None
        self._n_sample = None
        self._is_inside_rnn = None
        self._has_sample_axis = tf.constant(False)

    def set_stochastic_mixin(
        self, sample_axis_outermost, n_sample, is_inside_rnn
    ):
        """Set attributes."""
        self.sample_axis_outermost = sample_axis_outermost
        self.n_sample = n_sample
        self.is_inside_rnn = is_inside_rnn

        if self.sample_axis_outermost is not None:
            self._has_sample_axis = tf.constant(True)

    @property
    def sample_axis(self):
        if self._is_inside_rnn:
            return self._sample_axis_outermost - 1
        else:
            return self._sample_axis_outermost

    @property
    def sample_axis_outermost(self):
        return self._sample_axis_outermost

    @property
    def n_sample(self):
        return self._n_sample

    @property
    def is_inside_rnn(self):
        return bool(self._is_inside_rnn)

    @sample_axis_outermost.setter
    def sample_axis_outermost(self, sample_axis_outermost):
        self._sample_axis_outermost = int(sample_axis_outermost)

    @n_sample.setter
    def n_sample(self, n_sample):
        self._n_sample = int(n_sample)

    @is_inside_rnn.setter
    def is_inside_rnn(self, is_inside_rnn):
        self._is_inside_rnn = tf.constant(is_inside_rnn)
