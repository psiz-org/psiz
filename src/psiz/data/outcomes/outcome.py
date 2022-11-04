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
"""Module for data.

Classes:
    Outcome: Base class for outcome data.

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from psiz.data.trial_component import TrialComponent
from psiz.data.unravel_timestep import unravel_timestep


class Outcome(TrialComponent):
    """Base class for outcome data."""

    def __init__(self, name=None, sample_weight=None):
        """Initialize.

        Args:
            name: A string indicating the name of the outcome variable.
            sample_weight (optional): A 1D or 2D np.ndarray of floats.
                shape=(samples, [sequence_length])

        """
        TrialComponent.__init__(self)

        # TODO raise error at export if name unavailable?
        # if name is None:
        #     raise ValueError('')
        self.name = name
        self._sample_weight = sample_weight

    def process_sample_weight(self):
        """Process `sample_weight`.

        NOTE: Objects that subclass `Outcome` must call this method in
        the __init__ method after setting `n_sequence` and
        `sequence_length`.

        """
        # NOTE: If there is no sample weight, we assume full weight for
        # all elements.
        sample_weight = self._sample_weight
        if sample_weight is None:
            sample_weight = np.ones(
                [self.n_sequence, self.sequence_length], dtype=np.float32
            )
        self._sample_weight = self._validate_sample_weight(sample_weight)

    def _validate_sample_weight(self, sample_weight):
        """Validite `sample_weight`."""
        # Cast `sample_weight` to float if necessary.
        sample_weight = sample_weight.astype(float)

        # Check rank of `sample_weight`.
        if not (sample_weight.ndim == 2):
            raise ValueError(
                "The argument 'sample_weight' must be a rank-2 ND array."
            )

        # Check shape agreement.
        if not (sample_weight.shape[0] == self.n_sequence):
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

    def export(
        self, export_format='tf', with_timestep_axis=True
    ):
        """Export sample_weight.

        Subclasses of Outcome must call super().export(...) in order
        to obtain sample weights.

        """
        sample_weight = self.sample_weight
        if with_timestep_axis is False:
            sample_weight = unravel_timestep(sample_weight)

        if export_format == 'tf':
            sample_weight = tf.constant(sample_weight, dtype=K.floatx())
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return {self.name: sample_weight}

    def _subset_sample_weight(self, idx):
        """Subset of sample weight."""
        return self.sample_weight[idx]

    # TODO delete stack
    # def _stack_sample_weight(self, trials_list, sequence_length):
    #     """Stack `sample_weight` data."""
    #     # Start by padding first entry in list.
    #     timestep_pad = sequence_length - trials_list[0].sequence_length
    #     pad_width = ((0, 0), (0, timestep_pad))
    #     sample_weight = np.pad(
    #         trials_list[0].sample_weight,
    #         pad_width, mode='constant', constant_values=0
    #     )

    #     # Loop over remaining list.
    #     for i_trials in trials_list[1:]:
    #         timestep_pad = sequence_length - i_trials.sequence_length
    #         pad_width = ((0, 0), (0, timestep_pad))
    #         curr_sample_weight = np.pad(
    #             i_trials.sample_weight,
    #             pad_width, mode='constant', constant_values=0
    #         )

    #         sample_weight = np.concatenate(
    #             (sample_weight, curr_sample_weight), axis=0
    #         )

    #     return sample_weight
