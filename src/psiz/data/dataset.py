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
    Dataset: Generic composite class for data.

"""

import warnings

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
        if export_format != "tfds":
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )

        warnings.warn(
            "`Dataset.export(export_format='tfds')` is deprecated and will be "
            "removed in a future release. Use Tier A adapters such as "
            "`Dataset.tensorflow()`, `Dataset.torch()`, `Dataset.numpy()`, "
            "or `Dataset.arrow()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._export_tf_dataset(with_timestep_axis=with_timestep_axis)

    def _to_runtime_dataset(self, *, with_timestep_axis=None):
        """Materialize trial components as a backend-neutral runtime dataset."""
        from psiz.data.runtime.pydataset import PsizPyDataset

        x, y, w = self._materialize_xyw(with_timestep_axis=with_timestep_axis)
        return PsizPyDataset(x, y=y, w=w)

    def tensorflow(self, *, with_timestep_axis=None):
        """Return minimally processed, unbatched tf.data.Dataset rows."""
        return self._to_runtime_dataset(
            with_timestep_axis=with_timestep_axis
        ).tensorflow()

    def torch(self, *, with_timestep_axis=None):
        """Return minimally processed torch.utils.data.Dataset rows."""
        return self._to_runtime_dataset(with_timestep_axis=with_timestep_axis).torch()

    def numpy(self, *, with_timestep_axis=None):
        """Return minimally processed NumPy x/y/w payloads."""
        return self._to_runtime_dataset(with_timestep_axis=with_timestep_axis).numpy()

    def arrow(self, *, with_timestep_axis=None):
        """Return minimally processed pyarrow.Table rows."""
        return self._to_runtime_dataset(with_timestep_axis=with_timestep_axis).arrow()

    def save(
        self,
        output_dir,
        *,
        dataset_id,
        split_set_id="split_set_v1",
        split_label="train",
        split_version=1,
        license_name="Apache-2.0",
        with_timestep_axis=None,
    ):
        """Persist dataset as a PsiZ artifact directory.

        Args:
            output_dir: Target artifact directory path.
            dataset_id: Dataset identifier written to manifest.
            split_set_id (optional): Split set identifier.
            split_label (optional): Split label assigned to all rows.
            split_version (optional): Split assignment version.
            license_name (optional): License recorded in manifest.
            with_timestep_axis (optional): Override timestep-axis materialization.

        Returns:
            dict: Validated artifact manifest.

        """
        from psiz.data.io import write_dataset_artifact_from_samples

        x, y, w = self._materialize_xyw(with_timestep_axis=with_timestep_axis)
        samples = self._xyw_to_samples(x, y, w)
        return write_dataset_artifact_from_samples(
            samples,
            output_dir,
            dataset_id=dataset_id,
            split_set_id=split_set_id,
            split_label=split_label,
            split_version=split_version,
            license_name=license_name,
        )

    @staticmethod
    def load(
        dataset_root,
        *,
        split_set_id=None,
        split_labels=None,
    ):
        """Load a PsiZ artifact as a backend-neutral runtime dataset."""
        from psiz.data.runtime import load_dataset

        return load_dataset(
            dataset_root,
            split_set_id=split_set_id,
            split_labels=split_labels,
        )

    def _resolve_with_timestep_axis(self, with_timestep_axis):
        if with_timestep_axis is None:
            with_timestep_axis = False
            for component in self.components:
                with_timestep_axis = (
                    with_timestep_axis or component._export_with_timestep_axis
                )
        return with_timestep_axis

    def _materialize_xyw(self, with_timestep_axis=None):
        """Materialize dataset components to x/y/w mappings."""
        with_timestep_axis = self._resolve_with_timestep_axis(with_timestep_axis)

        x = {}
        for content in self.content_list:
            x_i = content.numpy(with_timestep_axis=with_timestep_axis)
            x.update(x_i)

        for group in self.group_list:
            x_i = group.numpy(with_timestep_axis=with_timestep_axis)
            x.update(x_i)

        y = {}
        w = {}
        for outcome in self.outcome_list:
            y_i, w_i = outcome.numpy(with_timestep_axis=with_timestep_axis)
            y.update(y_i)
            w.update(w_i)

        self._validate_output_keys(y)
        self._validate_output_keys(w)
        return x, y, w

    def _export_tf_dataset(self, with_timestep_axis=None):
        """Internal helper for creating TensorFlow datasets."""
        try:
            import tensorflow as tf
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "TensorFlow is required to materialize a tf.data.Dataset. "
                "Install TensorFlow or use `Dataset.torch()`, "
                "`Dataset.numpy()`, or `Dataset.arrow()` instead."
            ) from e

        x, y, w = self._materialize_xyw(with_timestep_axis=with_timestep_axis)
        if len(y) == 0:
            return tf.data.Dataset.from_tensor_slices((x))

        y = self._prepare_for_tf_dataset(y)
        w = self._prepare_for_tf_dataset(w)
        return tf.data.Dataset.from_tensor_slices((x, y, w))

    def _validate_output_keys(self, d):
        """Validate named output keys when multiple outputs are present."""
        if len(d) <= 1:
            return

        for k in d.keys():
            if k is None:
                raise ValueError(
                    "When a `Dataset` has multiple outputs, all "
                    "outputs must be created with the `name` argument."
                )

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
            self._validate_output_keys(d)
            return d

    def _xyw_to_samples(self, x, y, w):
        """Convert batched x/y/w mappings into per-sample payloads."""
        n_sample = self._infer_n_sample_from_blocks(x, y, w)
        samples = []
        for i_sample in range(n_sample):
            x_i = {k: v[i_sample] for k, v in x.items()}
            if len(y) == 0:
                samples.append(x_i)
                continue

            y_i = {k: v[i_sample] for k, v in y.items()}
            w_i = {k: v[i_sample] for k, v in w.items()}
            samples.append((x_i, y_i, w_i))
        return samples

    def _infer_n_sample_from_blocks(self, *blocks):
        for block in blocks:
            if len(block) == 0:
                continue
            key = next(iter(block.keys()))
            value = block[key]
            return int(value.shape[0])
        return 0
