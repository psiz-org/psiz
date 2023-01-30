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

from psiz.data.dataset_component import DatasetComponent
from psiz.data.contents.content import Content
from psiz.data.groups.group import Group
from psiz.data.outcomes.outcome import Outcome


class Dataset(object):
    """Generic composite class for data."""

    def __init__(self, components):
        """Initialize.

        Args:
            components: List of DatasetComponent objects. List
                should include at least one `psiz.data.Content` object.
                Other valid objects include `psiz.data.Outcome` objects
                and `psiz.data.Group` objects.

        """
        n_sample, sequence_length = self._validate_trial_components(components)
        self.n_sample = n_sample
        self.sequence_length = sequence_length

        content_list, group_list, outcome_list = self._sort_trial_components(components)
        self.content_list = content_list
        self.group_list = group_list
        self.outcome_list = outcome_list

    def _validate_trial_components(self, components):
        """Validate all trial components."""
        # Anchor on first DatasetComponent.
        n_sample = components[0].n_sample
        sequence_length = components[0].sequence_length

        for component_idx, component in enumerate(components[1:]):
            if not isinstance(component, DatasetComponent):
                raise ValueError(
                    "The object in position {0} is not a "
                    "`DatasetComponent`.".format(component_idx + 1)
                )

            # Check shape of DatasetComponent.
            if component.n_sample != n_sample:
                raise ValueError(
                    "All user-provided 'DatasetComponent' objects must have "
                    "the same `n_sample`. The 'DatasetComponent' in "
                    "position {0} does not match the previous "
                    "components.".format(component_idx + 1)
                )

            if component.sequence_length != sequence_length:
                raise ValueError(
                    "All user-provided 'DatasetComponent' objects must have "
                    "the same `sequence_length`. The 'DatasetComponent' in "
                    "position {0} does not match the previous "
                    "components.".format(component_idx + 1)
                )

        return n_sample, sequence_length

    def _sort_trial_components(self, components):
        """Sort trial components."""
        content_list = []
        group_list = []
        outcome_list = []

        for component_idx, component in enumerate(components):
            if isinstance(component, Content):
                content_list.append(component)
            elif isinstance(component, Outcome):
                outcome_list.append(component)
            elif isinstance(component, Group):
                group_list.append(component)
            else:
                raise ValueError(
                    "The `DatasetComponent` in position {0} must be an  "
                    "instance of `psiz.data.Content`, `psiz.data.Outcome`, or "
                    "`psiz.data.Group` to use `Dataset`.".format(component_idx)
                )

        return content_list, group_list, outcome_list

    @property
    def components(self):
        """Return all trial components."""
        components = []
        for content in self.content_list:
            components.append(content)
        for group in self.group_list:
            components.append(group)
        for outcome in self.outcome_list:
            components.append(outcome)
        return components

    def export(self, export_format="tfds", with_timestep_axis=None):
        """Export trial data as model-consumable object.

        Args:
            export_format (optional): The output format of the dataset.
                By default the dataset is formatted as a
                `tf.data.Dataset` object.
            with_timestep_axis (optional): Boolean indicating if data
                should be returned with a timestep axis. By default,
                dataset is exported with a timestep axis if any of the
                provided `DataComponents` were initialized with a
                timestep axis. Callers can overide default behavior
                by setting this argument.

        Returns:
            ds: A dataset that can be consumed by a model.

        """
        if with_timestep_axis is None:
            with_timestep_axis = False
            for component in self.components:
                with_timestep_axis = (
                    with_timestep_axis or component._export_with_timestep_axis
                )

        # Assemble model input.
        x = {}
        for content in self.content_list:
            x_i = content.export(
                export_format=export_format, with_timestep_axis=with_timestep_axis
            )
            x.update(x_i)
        # Add groups (if present).
        if len(self.group_list) > 0:
            for group in self.group_list:
                x_i = group.export(
                    export_format=export_format, with_timestep_axis=with_timestep_axis
                )
                x.update(x_i)

        # Assemble outcomes (if present and not suppressed).
        if len(self.outcome_list) > 0:
            y = {}
            w = {}
            for outcome in self.outcome_list:
                y_i, w_i = outcome.export(
                    export_format=export_format, with_timestep_axis=with_timestep_axis
                )
                y.update(y_i)
                w.update(w_i)

        if export_format == "tfds":
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
