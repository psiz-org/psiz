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
    TrialDataset: Generic composite class for trial data.

"""

import h5py
from importlib.metadata import version
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from psiz.trials.experimental.contents.rank_similarity import RankSimilarity
from psiz.trials.experimental.contents.rate_similarity import RateSimilarity
from psiz.trials.experimental.outcomes.continuous import Continuous
from psiz.trials.experimental.outcomes.sparse_categorical import SparseCategorical
from psiz.trials.experimental.unravel_timestep import unravel_timestep


class TrialDataset(object):
    """Generic composite class for trial data."""

    def __init__(self, content, groups=None, outcome=None, weight=None):
        """Initialize.

        Arguments:
            content: A subclass of a psiz.trials.Content object.
            groups (optional): A np.ndarray of integers. Must be rank-2
                or rank-3.
                shape=(n_sequence, [max_timestep], n_col)
            outcome (optional): A subclass of a psiz.trials.Outcome
                object.
            weight (optional): 2D np.ndarray of floats.
                shape=(n_sequence, max_timestep)

        """
        self.content = content

        # Anchor initialization on `content`.
        self.n_sequence = content.n_sequence
        self.max_timestep = content.max_timestep

        # Handle `groups` intialization.
        if groups is None:
            groups = np.zeros(
                [self.n_sequence, self.max_timestep, 1], dtype=int
            )
        else:
            groups = self._check_groups(groups)
        self.groups = groups

        # Handle outcome initialization.
        if outcome is not None:
            self._check_outcome(outcome)
        self.outcome = outcome

        # Handle `weight` initialization.
        if weight is None:
            weight = self.content.is_actual().astype(np.float)
        else:
            weight = self._check_weight(weight)
        self.weight = weight

    def as_dataset(self, input_only=False, timestep=True, format='tf'):
        """Format trial data as model-consumable object.

        Arguments:
            input_only: Boolean indicating if only the input should be
                returned.
            timestep: Boolean indicating if data should be returned
                with a timestep axis. If `False`, data is reshaped.
            format: The format of the dataset. By default the dataset
            is formatted as a tf.data.Dataset object.

        Returns:
            ds: A dataset that can be consumed by a model.

        """
        if format == 'tf':
            # Assemble model input.
            x = self.content._for_dataset(format=format, timestep=timestep)
            groups = self.groups
            if timestep is False:
                groups = unravel_timestep(groups)
            x.update({
                'groups': tf.constant(groups, dtype=tf.int32)
            })

            if not input_only:
                # Assemble model output
                if self.outcome is not None:
                    y = self.outcome._for_dataset(
                        format=format, timestep=timestep
                    )
                else:
                    raise ValueError("No outcome has been specified.")

                # Assemble weights.
                w = self.weight
                if timestep is False:
                    w = unravel_timestep(w)
                w = tf.constant(w, dtype=K.floatx())

                ds = tf.data.Dataset.from_tensor_slices((x, y, w))
            else:
                ds = tf.data.Dataset.from_tensor_slices((x))
        else:
            raise ValueError(
                "Unrecognized format '{0}'.".format(format)
            )
        return ds

    def is_actual(self):
        """Return 2D Boolean array indicating trials with actual content.

        Returns:
            is_actual:
                shape=(n_sequence, max_timestep)

        """
        return self.content.is_actual()

    @classmethod
    def load(cls, filepath):
        """Load trials.

        Arguments:
            filepath: The location of the hdf5 file to load.

        """
        f = h5py.File(filepath, "r")
        content = cls._load_h5_group(f['content'])
        groups = f["groups"][()]
        outcome = cls._load_h5_group(f['outcome'])
        weight = f["weight"][()]

        trials = TrialDataset(
            content, groups=groups, outcome=outcome, weight=weight,
        )
        return trials

    def save(self, filepath):
        """Save the TrialDataset object as an HDF5 file.

        Arguments:
            filepath: String specifying the path tosave the data.

        """
        ver = version("psiz")
        ver = '.'.join(ver.split('.')[:3])

        f = h5py.File(filepath, "w")
        f.create_dataset("class_name", data="TrialDataset")
        f.create_dataset("version", data=ver)
        grp_content = f.create_group("content")
        self.content._save(grp_content)
        f.create_dataset("groups", data=self.groups)
        grp_outcome = f.create_group("outcome")
        if self.outcome is not None:
            self.outcome._save(grp_outcome)
        else:
            grp_outcome.create_dataset("class_name", data="None")
        f.create_dataset("weight", data=self.weight)
        f.close()

    def stack(self, trials_list):
        """Return new object with sequence-stacked data.

        Arguments:
            trials_list: A tuple of TrialDataset objects to be
                stacked. All objects must be the same class.

        Returns:
            A new object.

        """
        # Determine maximum number of timesteps and decouple content and
        # outputs.
        max_timestep = 0
        content_list = []
        outcome_list = []
        for i_trials in trials_list:
            if i_trials.max_timestep > max_timestep:
                max_timestep = i_trials.max_timestep
            content_list.append(i_trials.content)
            outcome_list.append(i_trials.outcome)

        groups = self._stack_groups(trials_list, max_timestep)
        weight = self._stack_weight(trials_list, max_timestep)
        content = content_list[0].stack(content_list)
        outcome = outcome_list[0].stack(outcome_list)
        stacked = TrialDataset(
            content, groups=groups, outcome=outcome, weight=weight
        )
        return stacked

    def subset(self, idx):
        """Return subset of trials as a new RankObservations object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new RankObservations object.

        """
        content_sub = self.content.subset(idx)

        outcome_sub = None
        if self.outcome is not None:
            outcome_sub = self.outcome.subset(idx)

        return TrialDataset(
            content_sub,
            groups=self.groups[idx],
            outcome=outcome_sub,
            weight=self.weight[idx]
        )

    def _check_weight(self, weight):
        """Check the validity of `weight`."""
        # Cast `weight` to float if necessary.
        weight = weight.astype(np.float)

        # Check rank of `weight`.
        if not (weight.ndim == 2):
            raise ValueError(
                "The argument 'weight' must be a rank 2 ND array."
            )

        # Check shape agreement.
        if not (weight.shape[0] == self.n_sequence):
            raise ValueError(
                "The argument 'weight' must have "
                "shape=(n_squence, max_timestep) as determined by `content`."
            )
        if not (weight.shape[1] == self.max_timestep):
            raise ValueError(
                "The argument 'weight' must have "
                "shape=(n_squence, max_timestep) as determined by `content`."
            )
        return weight

    def _check_groups(self, groups):
        """Check the validity of `groups`."""
        # Cast `groups` to int if necessary.
        groups = groups.astype(np.int32)

        if groups.ndim == 2:
            # Assume independent trials and add singleton timestep axis.
            groups = np.expand_dims(groups, axis=1)

        # Check rank of `groups`.
        if not (groups.ndim == 3):
            raise ValueError(
                "The argument 'groups' must be a rank 3 ND array."
            )

        # Check shape agreement.
        if not (groups.shape[0] == self.n_sequence):
            raise ValueError(
                "The argument 'groups' must have "
                "shape=(n_squence, max_timestep) as determined by `content`."
            )
        if not (groups.shape[1] == self.max_timestep):
            raise ValueError(
                "The argument 'groups' must have "
                "shape=(n_squence, max_timestep) as determined by `content`."
            )

        # Check lowerbound support limit.
        bad_locs = groups < 0
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError(
                "The parameter 'groups' contains integers less than 0. "
                "Found {0} bad trial(s).".format(n_bad)
            )
        return groups

    def _check_outcome(self, outcome):
        # Check rank of `groups`.
        if outcome.n_sequence != self.n_sequence:
            raise ValueError(
                "The user-provided 'outcome' object must agree with the "
                "`n_sequence` attribute of the user-provided"
                "`content` object."
            )

        if outcome.max_timestep != self.max_timestep:
            raise ValueError(
                "The user-provided 'outcome' object must agree with the "
                "`max_timestep` attribute of the user-provided "
                "`content` object."
            )

    @staticmethod
    def _load_h5_group(grp):
        # Encoding/read rules changed in h5py 3.0, requiring asstr() call.
        try:
            class_name = grp["class_name"].asstr()[()]
        except AttributeError:
            class_name = grp["class_name"][()]
        custom_objects = {
            'RankSimilarity': RankSimilarity,
            'RateSimilarity': RateSimilarity,
            'SparseCategorical': SparseCategorical,
            'Continuous': Continuous,
        }
        if class_name == 'None':
            return None
        else:
            if class_name in custom_objects:
                group_class = custom_objects[class_name]
            else:
                raise NotImplementedError

            return group_class._load(grp)

    def _stack_groups(self, trials_list, max_timestep):
        """Stack `groups` data."""
        # Before doing anything, check that `groups` shape is compatible.
        n_group = trials_list[0].groups.shape[2]
        for i_trials in trials_list[1:]:
            if trials_list[0].groups.shape[2] != n_group:
                raise ValueError(
                    'The shape of `groups` for the different TrialDatasets '
                    'must be identical on axis=2.'
                )

        # Start by padding first entry in list.
        timestep_pad = max_timestep - trials_list[0].max_timestep
        pad_width = ((0, 0), (0, timestep_pad), (0, 0))
        groups = np.pad(
            trials_list[0].groups,
            pad_width, mode='constant', constant_values=0
        )

        # Loop over remaining list.
        for i_trials in trials_list[1:]:
            timestep_pad = max_timestep - i_trials.max_timestep
            pad_width = ((0, 0), (0, timestep_pad), (0, 0))
            curr_groups = np.pad(
                i_trials.groups,
                pad_width, mode='constant', constant_values=0
            )

            groups = np.concatenate(
                (groups, curr_groups), axis=0
            )

        return groups

    def _stack_weight(self, trials_list, max_timestep):
        """Stack `weight` data."""
        # Start by padding first entry in list.
        timestep_pad = max_timestep - trials_list[0].max_timestep
        pad_width = ((0, 0), (0, timestep_pad))
        weight = np.pad(
            trials_list[0].weight,
            pad_width, mode='constant', constant_values=0
        )

        # Loop over remaining list.
        for i_trials in trials_list[1:]:
            timestep_pad = max_timestep - i_trials.max_timestep
            pad_width = ((0, 0), (0, timestep_pad))
            curr_groups = np.pad(
                i_trials.weight,
                pad_width, mode='constant', constant_values=0
            )

            weight = np.concatenate(
                (weight, curr_groups), axis=0
            )

        return weight
