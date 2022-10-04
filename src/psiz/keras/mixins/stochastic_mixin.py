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


class StochasticMixin():
    """A mixin for layers that use stochastic sampling.

    A companion mixin for `psiz.keras.models.Stochastic` model.

    Attributes:
        sample_axis: The axis used for samples (outside of RNN cell).
        sample_axis_in_cell: The axis used for samples (inside of RNN cell,
            where timstep axis has been dropped).
        n_sample: The number of samples on the sample axis.

    Notes:
        Attributes of mixin are set when `build` method of `Stochastic`
        model is called. Before `build` is called, the attributes will
        have value `None`.

    """

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self._sample_axis = None
        self._n_sample = None

    @property
    def sample_axis(self):
        return self._sample_axis

    @property
    def sample_axis_in_cell(self):
        return self._sample_axis - 1

    @property
    def n_sample(self):
        return self._n_sample

    @sample_axis.setter
    def sample_axis(self, sample_axis):
        self._sample_axis = int(sample_axis)

    @n_sample.setter
    def n_sample(self, n_sample):
        self._n_sample = int(n_sample)
