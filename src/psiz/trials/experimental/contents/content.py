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
"""Module for trials.

Classes:
    Content: Abstract class for trial content data.

"""

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

from psiz.trials.experimental.trial_component import TrialComponent


class Content(TrialComponent, metaclass=ABCMeta):
    """Abstract class for trial content data."""

    def __init__(self):
        """Initialize."""
        TrialComponent.__init__(self)

        # Additional Attributes determined by concrete class.
        self.stimulus_set = None

        # Immutable attributes.
        self._mask_zero = True
        self._placeholder = 0

    @property
    def mask_zero(self):
        """Getter method for `mask_zero`."""
        return self._mask_zero

    @property
    def placeholder(self):
        """Getter method for `placeholder`."""
        return self._placeholder

    @abstractmethod
    def is_actual(self):
        """Return 2D Boolean array indicating trials with actual content.

        Returns:
            is_actual:
                shape=(n_sequence, max_timestep)

        """

    def unique_configurations(self):
        """Generate a unique ID for each content configuration.

        Convenience function that generates a unique ID for each unique
        content configuration.

        Will call subclass `_config_attrs` in order to determine
        unique configuraitons. It is assumed that all return attributes
        have shape=(n_sequence, max_timestep)

        Returns:
            config_idx: A unique index for each type of trial
                configuration.
                shape=(n_sequence, max_timestep)
            df_config: A DataFrame containing all the unique
                trial configurations.

        """
        attr_list = self._config_attrs()
        if len(attr_list) == 0:
            config_idx = np.zeros([self.n_sequence, self.max_timestep])
            df_config = None
        else:
            # Assemble dictionary of relevant attributes (as flattened array).
            n_trial = self.n_sequence * self.max_timestep
            d = {}
            for attr in attr_list:
                d.update({
                    attr: np.reshape(getattr(self, attr), [n_trial])
                })
            # Determine unique content configurations by exploiting pandas
            # DataFrame.
            df_config = pd.DataFrame(d)
            df_config = df_config.drop_duplicates()
            df_config = df_config.sort_values(by=attr_list[0], axis=0)
            df_config = df_config.reset_index(drop=True)

            # Loop over distinct configurations in order to determine
            # configuration index for all trials.
            config_idx = np.zeros(
                [self.n_sequence, self.max_timestep], dtype=np.int32
            )
            for index, row in df_config.iterrows():
                bidx = self._find_trials_matching_config(row)
                config_idx[bidx] = index

        return config_idx, df_config

    def _config_attrs(self):
        """Return attributes that govern trial configurations."""
        return []

    def _find_trials_matching_config(self, row):
        """Find trials matching configuration.

        Arguments:
            row: A pandas.Series object representing a trial
                configuration.

        """
        bidx = np.ones([self.n_sequence, self.max_timestep], dtype=bool)
        for index, value in row.items():
            bidx_key = np.equal(getattr(self, index), value)
            # Determine intersection.
            bidx = np.logical_and(bidx, bidx_key)
        return bidx
