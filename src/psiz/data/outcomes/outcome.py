# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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
"""Module for data.

Classes:
    Outcome: Base class for outcome data.

"""

import numpy as np

from psiz.data.dataset_component import DatasetComponent
from psiz.data.unravel_timestep import unravel_timestep


class Outcome(DatasetComponent):
    """Base class for outcome data."""

    def __init__(self, name=None, sample_weight=None):
        """Initialize.

        Args:
            name: A string indicating the name of the outcome variable.
            sample_weight (optional): A 1D or 2D np.ndarray of floats.
                shape=(samples, [sequence_length])

        """
        DatasetComponent.__init__(self)

        self.name = name
        self._sample_weight = sample_weight

    def process_sample_weight(self):
        """Process `sample_weight`.

        NOTE: Objects that subclass `Outcome` must call this method in
        the __init__ method after setting `n_sample` and
        `sequence_length`.

        """
        # NOTE: If there is no sample weight, we assume full weight for
        # all elements.
        sample_weight = self._sample_weight
        if sample_weight is None:
            sample_weight = np.ones(
                [self.n_sample, self.sequence_length], dtype=np.float32
            )
        else:
            if sample_weight.ndim == 1:
                # Assume trials are independent and add singleton dimension for
                # timestep axis.
                sample_weight = np.expand_dims(sample_weight, axis=self.timestep_axis)
        self._sample_weight = self._validate_sample_weight(sample_weight)

    def _validate_sample_weight(self, sample_weight):
        """Validate `sample_weight`."""
        # Cast `sample_weight` to float if necessary.
        sample_weight = sample_weight.astype(float)

        # Check rank of `sample_weight`.
        if not (sample_weight.ndim == 2):
            raise ValueError("The argument 'sample_weight' must be a rank-2 ND array.")

        # Check shape agreement.
        if not (sample_weight.shape[0] == self.n_sample):
            raise ValueError(
                "The argument 'sample_weight' must have "
                "shape=(samples, sequence_length) that agrees with the rest "
                "of the object."
            )
        if not (sample_weight.shape[1] == self.sequence_length):
            raise ValueError(
                "The argument 'sample_weight' must have "
                "shape=(samples, sequence_length) that agrees with the rest "
                "of the object."
            )
        return sample_weight

    @property
    def sample_weight(self):
        """Return sample weight."""
        return self._sample_weight

    def numpy(self, with_timestep_axis=None):
        """Export sample weight as NumPy arrays.

        Subclasses of Outcome must call super().numpy(...) in order
        to obtain sample weights.

        Args:
            with_timestep_axis (optional): Boolean indicating if data
                should be returned with a timestep axis. By default,
                data is exported in the same format as it was
                provided at initialization. Callers can override
                default behavior by setting this argument.

        """
        if with_timestep_axis is None:
            with_timestep_axis = self._export_with_timestep_axis

        sample_weight = self.sample_weight
        if with_timestep_axis is False:
            sample_weight = unravel_timestep(sample_weight)
        sample_weight = sample_weight.astype(np.float32)
        return {self.name: sample_weight}

    def _subset_sample_weight(self, idx):
        """Subset of sample weight."""
        return self.sample_weight[idx]
