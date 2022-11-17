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
"""Module for data.

Classes:
    Dataset: Generic composite class for data.

"""

import tensorflow as tf

from psiz.data.trial_component import TrialComponent
from psiz.data.contents.content import Content
from psiz.data.groups.group import Group
from psiz.data.outcomes.outcome import Outcome


class Dataset(object):
    """Generic composite class for data."""

    def __init__(self, trial_components):
        """Initialize.

        Args:
            trial_components: List of TrialComponent objects. List
                should include at least one `psiz.data.Content` object.
                Other valid objects include `psiz.data.Outcome` objects
                and `psiz.data.Group` objects.

        """
        n_sequence, sequence_length = self._validate_trial_components(
            trial_components
        )
        self.n_sequence = n_sequence
        self.sequence_length = sequence_length

        content_list, group_list, outcome_list = self._sort_trial_components(
            trial_components
        )
        self.content_list = content_list
        self.group_list = group_list
        self.outcome_list = outcome_list

    def _validate_trial_components(self, trial_components):
        """Validate all trial components."""
        # Anchor on first TrialComponent.
        n_sequence = trial_components[0].n_sequence
        sequence_length = trial_components[0].sequence_length

        for component_idx, trial_component in enumerate(trial_components[1:]):
            if not isinstance(trial_component, TrialComponent):
                raise ValueError(
                    "The object in position {0} is not a "
                    "`TrialComponent`.".format(component_idx + 1)
                )

            # Check shape of TrialComponent.
            if trial_component.n_sequence != n_sequence:
                raise ValueError(
                    "All user-provided 'TrialComponent' objects must have the "
                    "same `n_sequence`. The 'TrialComponent' in position {0} "
                    "does not match the previous components.".format(
                        component_idx + 1
                    )
                )

            if trial_component.sequence_length != sequence_length:
                raise ValueError(
                    "All user-provided 'TrialComponent' objects must have the "
                    "same `sequence_length`. The 'TrialComponent' in position "
                    "{0} does not match the previous components.".format(
                        component_idx + 1
                    )
                )

        return n_sequence, sequence_length

    def _sort_trial_components(self, trial_components):
        """Sort trial components."""
        content_list = []
        group_list = []
        outcome_list = []

        for component_idx, trial_component in enumerate(trial_components):
            if isinstance(trial_component, Content):
                content_list.append(trial_component)
            elif isinstance(trial_component, Outcome):
                outcome_list.append(trial_component)
            elif isinstance(trial_component, Group):
                group_list.append(trial_component)
            else:
                raise ValueError(
                    "The `TrialComponent` in position {0} must be an  "
                    "instance of `psiz.data.Content`, `psiz.data.Outcome`, or "
                    "`psiz.data.Group` to use `Dataset`.".format(
                        component_idx
                    )
                )

        return content_list, group_list, outcome_list

    @property
    def trial_components(self):
        """Return all trial components."""
        trial_components = []
        for content in self.content_list:
            trial_components.append(content)
        for group in self.group_list:
            trial_components.append(group)
        for outcome in self.outcome_list:
            trial_components.append(outcome)
        return trial_components

    def export(
        self, with_timestep_axis=True, export_format='tfds', inputs_only=False
    ):
        """Export trial data as model-consumable object.

        Args:
            with_timestep_axis (optional): Boolean indicating if data
                should be returned with a timestep axis. If `False`,
                data is reshaped.
            export_format (optional): The output format of the dataset.
                By default the dataset is formatted as a
                    tf.data.Dataset object.
            inputs_only (optional): Boolean indicating if only the input
                should be returned.
        Returns:
            ds: A dataset that can be consumed by a model.

        """
        # Assemble model input.
        x = {}
        for content in self.content_list:
            x_i = content.export(
                export_format=export_format,
                with_timestep_axis=with_timestep_axis
            )
            x.update(x_i)
        # Add groups (if present).
        if len(self.group_list) > 0:
            for group in self.group_list:
                x_i = group.export(
                    export_format=export_format,
                    with_timestep_axis=with_timestep_axis
                )
                x.update(x_i)

        # Assemble outcomes (if present and not suppressed).
        if len(self.outcome_list) > 0 and not inputs_only:
            y = {}
            w = {}
            for outcome in self.outcome_list:
                y_i, w_i = outcome.export(
                    export_format=export_format,
                    with_timestep_axis=with_timestep_axis
                )
                y.update(y_i)
                w.update(w_i)

        if export_format == 'tfds':
            try:
                y = self._prepare_for_tf_dataset(y)
                w = self._prepare_for_tf_dataset(w)
                ds = tf.data.Dataset.from_tensor_slices((x, y, w))
            except NameError:
                ds = tf.data.Dataset.from_tensor_slices((x))
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return ds

    def _prepare_for_tf_dataset(self, d):
        """Prepare `y` and `w` for TensorFlow Dataset.

        If only one key in dictionary, abandon dictionary structure and
        just use the Tensor since TensorFlow/Keras does not need it. If
        there is more than one key, we assume a multiple-output model
        that requires all outputs and sample weights to be labeled via
        dictionary keys.

        Args:
            d: A dictionary of TF Tensors.

        """
        if len(d) == 1:
            key, tensor = d.popitem()
            return tensor
        else:
            # Make sure all keys are defined.
            for k in d.keys():
                if k is None:
                    raise ValueError(
                        "When a `Dataset` has multiple outputs, all "
                        "outputs must be created with the `name` argument."
                    )
            return d
