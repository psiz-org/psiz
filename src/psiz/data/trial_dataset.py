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
"""Module for trials.

Classes:
    TrialDataset: Generic composite class for trial data.

"""

import warnings

import h5py
from importlib.metadata import version
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from psiz.data.contents.rank_similarity import RankSimilarity
from psiz.data.contents.rate_similarity import RateSimilarity
from psiz.data.outcomes.continuous import Continuous
from psiz.data.outcomes.sparse_categorical import (
    SparseCategorical
)
from psiz.data.unravel_timestep import unravel_timestep


class TrialDataset(object):
    """Generic composite class for trial data."""

    def __init__(self, content, outcome=None, groups=None, weight=None):
        """Initialize.

        Args:
            content: A subclass of a psiz.trials.Content object.
            outcome (optional): A subclass of a psiz.trials.Outcome
                object.
            groups (optional): A dictionary composed of values where
                each value is an np.ndarray of integers that must be
                rank-2 or rank-3.
                dict value shape=(n_sequence, [max_timestep], n_col)
            weight (optional): 2D np.ndarray of floats.
                shape=(n_sequence, max_timestep)

        """
        self._timestep_axis = 1  # TODO
        self.content = content

        # Anchor initialization on `content`.
        self.n_sequence = content.n_sequence
        self.max_timestep = content.max_timestep

        # Handle `groups` intialization.
        if groups is not None:
            for gate_key, gate_weights in groups.items():
                gate_weights = self._validate_gate_weights(
                    gate_key, gate_weights
                )
                groups[gate_key] = gate_weights
        self.groups = groups

        # Handle outcome initialization.
        if outcome is not None:
            self._check_outcome(outcome)
        self.outcome = outcome

        # Handle `weight` initialization.
        if weight is None:
            weight = self.content.is_actual.astype(float)
        else:
            weight = self._check_weight(weight)
        self.weight = weight

    def export(self, timestep=True, export_format='tf', inputs_only=False):
        """Export trial data as model-consumable object.

        Args:
            timestep (optional): Boolean indicating if data should be
                returned with a timestep axis. If `False`, data is
                reshaped.
            export_format (optional): The output format of the dataset.
                By default the dataset is formatted as a
                    tf.data.Dataset object.
            inputs_only (optional): Boolean indicating if only the input
                should be returned.
        Returns:
            ds: A dataset that can be consumed by a model.

        """
        # Assemble model input.
        x = self.content.export(
            export_format=export_format, timestep=timestep
        )
        if self.groups is not None:
            groups = self._export_groups(
                export_format=export_format, timestep=timestep
            )
            x.update(groups)

        # Assemble model output
        if self.outcome is not None and not inputs_only:
            y = self.outcome.export(
                export_format=export_format, timestep=timestep
            )

        # Assemble weights.
        if not inputs_only:
            w = self._export_weight(
                export_format=export_format, timestep=timestep
            )

        if export_format == 'tf':
            try:
                ds = tf.data.Dataset.from_tensor_slices((x, y, w))
            except NameError:
                ds = tf.data.Dataset.from_tensor_slices((x))
        elif export_format == 'tensors':  # TODO HACK
            try:
                ds = (x, y, w)
            except NameError:
                ds = (x)
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return ds

    @property
    def is_actual(self):
        """Return 2D Boolean array indicating trials with actual content.

        Returns:
            is_actual:
                shape=(n_sequence, max_timestep)

        """
        return self.content.is_actual

    @classmethod
    def load(cls, filepath):
        """Load trials.

        Args:
            filepath: The location of the hdf5 file to load.

        """
        f = h5py.File(filepath, "r")
        content = cls._load_h5_group(f, 'content')
        groups = cls._load_h5_group(f, 'groups')
        outcome = cls._load_h5_group(f, 'outcome')
        weight = f["weight"][()]

        trials = TrialDataset(
            content, groups=groups, outcome=outcome, weight=weight,
        )
        return trials

    def save(self, filepath):
        """Save the TrialDataset object as an HDF5 file.

        Args:
            filepath: String specifying the path to save the data.

        """
        f = h5py.File(filepath, "w")

        # Add class name and versioning information.
        ver = version("psiz")
        ver = '.'.join(ver.split('.')[:3])
        f.create_dataset("class_name", data="TrialDataset")
        f.create_dataset("psiz_version", data=ver)

        # Add content (always exists).
        grp_content = f.create_group("content")
        self.content.save(grp_content)

        # Add groups (sometimes exsits).
        if self.groups is not None:
            grp_groups = f.create_group("groups")
            self._save_groups(grp_groups)

        # Add outcomes (sometimes exists).
        if self.outcome is not None:
            grp_outcome = f.create_group("outcome")
            self.outcome.save(grp_outcome)

        # Add weights (always exists because of default initialization).
        f.create_dataset("weight", data=self.weight)
        f.close()

    def stack(self, trials_list):
        """Return new object with sequence-stacked data.

        Args:
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
        """Return subset of sequences as a new TrialDataset object.

        Args:
            idx: The indices corresponding to the subset.

        Returns:
            A new TrialDataset object.

        """
        content_sub = self.content.subset(idx)

        outcome_sub = None
        if self.outcome is not None:
            outcome_sub = self.outcome.subset(idx)

        groups_sub = None
        if self.groups is not None:
            groups_sub = {}
            for key, value in self.groups.items():
                groups_sub[key] = value[idx]

        return TrialDataset(
            content_sub,
            groups=groups_sub,
            outcome=outcome_sub,
            weight=self.weight[idx]
        )

    def _check_weight(self, weight):
        """Check the validity of `weight`."""
        # Cast `weight` to float if necessary.
        weight = weight.astype(float)

        # Check rank of `weight`.
        if not (weight.ndim == 2):
            raise ValueError(
                "The argument 'weight' must be a rank-2 ND array."
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

    def _validate_gate_weights(self, gate_key, gate_weights):
        """Validate gate weight values."""
        if gate_weights.ndim == 2:
            # Assume independent trials and add singleton timestep axis.
            gate_weights = np.expand_dims(
                gate_weights, axis=self._timestep_axis
            )

        # Check rank of `gate_weights`.
        if not (gate_weights.ndim == 3):
            raise ValueError(
                "The gate weights for the dictionary key '{0}' must be a "
                "rank-2 or rank-3 ND array. If using a sparse coding format, "
                "make sure you have a trailing singleton dimension to meet "
                "this requirement.".format(gate_key)
            )

        # If `gate_weights` looks like sparse coding format, check data type.
        if gate_weights.shape[-1] == 1:
            if not isinstance(gate_weights[0, 0, 0], (int, np.integer)):
                warnings.warn(
                    "The gate weights for the dictionary key '{0}' appear to "
                    "use a sparse coding. To improve efficiency, these "
                    "weights should have an integer dtype.".format(gate_key)
                )

        # Check shape agreement.
        if not (gate_weights.shape[0] == self.n_sequence):
            raise ValueError(
                "The gate weights for the dictionary key '{0}' must have "
                "a shape that agrees with 'n_squence' of the 'content'"
                ".".format(gate_key)
            )
        if not (gate_weights.shape[1] == self.max_timestep):
            raise ValueError(
                "The gate weights for the dictionary key '{0}' must have "
                "a shape that agrees with 'max_timestep' of the 'content'"
                ".".format(gate_key)
            )

        # Check lowerbound support limit.
        bad_locs = gate_weights < 0
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError(
                "The gate weights for the dictionary key '{0}' contain "
                "values less than 0. Found {1} bad trial(s).".format(
                    gate_key, n_bad
                )
            )
        return gate_weights

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

    def _save_groups(self, h5_grp):
        """Add relevant data to H5 group.

        Args:
            h5_grp: H5 group for saving data.

        """
        for gate_key, gate_weights in self.groups.items():
            h5_grp.create_dataset(gate_key, data=gate_weights)
        return None

    def _export_groups(self, export_format='tf', timestep=True):
        """Export groups."""
        groups = self.groups
        for gating_key, gate_weights in groups.items():
            if timestep is False:
                groups[gating_key] = unravel_timestep(gate_weights)
            groups[gating_key] = tf.constant(groups[gating_key])
        return groups

    def _export_weight(self, export_format='tf', timestep=True):
        """Export weight."""
        w = self.weight
        if timestep is False:
            w = unravel_timestep(w)
        return tf.constant(w, dtype=K.floatx())

    @staticmethod
    def _load_h5_group(f, grp_name):
        """Load H5 group."""
        try:
            h5_grp = f[grp_name]
        except KeyError:
            return None

        try:
            # NOTE: Encoding/read rules changed in h5py 3.0, requiring asstr()
            # call. The minimum requirements are reflected in `setup.cfg`.
            class_name = h5_grp["class_name"].asstr()[()]
            custom_objects = {
                'RankSimilarity': RankSimilarity,
                'RateSimilarity': RateSimilarity,
                'SparseCategorical': SparseCategorical,
                'Continuous': Continuous,
            }
            if class_name in custom_objects:
                group_class = custom_objects[class_name]
            else:
                raise NotImplementedError

            return group_class.load(h5_grp)
        except KeyError:
            # Assume a dictionary.
            d_keys = list(h5_grp.keys())
            d = {}
            for key in d_keys:
                d[key] = h5_grp[key][()]
            return d

    def _stack_groups(self, trials_list, max_timestep):
        """Stack `groups` data."""
        # First check that groups keys are compatible.
        # NOTE: It is not safe to simply pad an missing key with zeros, since
        # zero likely has user-defined semantics.
        gating_keys = trials_list[0].groups.keys()
        for i_trials in trials_list[1:]:
            i_gating_keys = i_trials.groups.keys()
            if gating_keys != i_gating_keys:
                raise ValueError(
                    'The dictionary keys of `groups` must be identical '
                    'for all TrialDatasets. Got a mismatch: {0} and '
                    '{1}.'.format(str(gating_keys), str(i_gating_keys))
                )

        # Loop over each key in groups.
        groups_stacked = {}
        for key in gating_keys:
            # Check that shapes are compatible.
            value_shape = trials_list[0].groups[key].shape
            for i_trials in trials_list[1:]:
                i_value_shape = i_trials.groups[key].shape
                # is_axis_1_ok = value_shape[1] == i_value_shape[1]  TODO
                is_axis_2_ok = value_shape[2] == i_value_shape[2]
                if not is_axis_2_ok:
                    raise ValueError(
                        "The shape of 'groups's '{0}' is not compatible. They "
                        "must be identical on axis=2.".format(key)
                    )

            # Start by padding first entry in list.
            timestep_pad = max_timestep - trials_list[0].max_timestep
            pad_width = ((0, 0), (0, timestep_pad), (0, 0))
            groups = np.pad(
                trials_list[0].groups[key],
                pad_width, mode='constant', constant_values=0
            )

            # Loop over remaining list.
            for i_trials in trials_list[1:]:
                timestep_pad = max_timestep - i_trials.max_timestep
                pad_width = ((0, 0), (0, timestep_pad), (0, 0))
                curr_groups = np.pad(
                    i_trials.groups[key],
                    pad_width, mode='constant', constant_values=0
                )

                groups = np.concatenate(
                    (groups, curr_groups), axis=0
                )
            groups_stacked[key] = groups

        return groups_stacked

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
