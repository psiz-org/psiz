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
"""Module for defining and loading catalogs.

Classes:
    Catalog: A catalog to keep track of stimuli.

"""

import copy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


class Catalog():
    """Class to keep track of stimuli information.

    Attributes:
        n_stimuli:  The number of unique stimuli.
        stimuli: Pandas dataframe containing information about the
            stimuli:
            id: A unique stimulus id.
            filepath: The filepath for the corresponding stimulus.

    Methods:
        class_id:
        file_path:
        id:
        save:
        subset:

    """

    def __init__(
            self, stimulus_id, stimulus_filepath, class_id=None,
            class_label=None):
        """Initialize.

        Arguments:
            stimulus_id: A 1D integer array.
                shape=(n_stimuli,)
            stimulus_filepath: A 1D list of strings.
                shape=(n_stimuli,)
            class_id (optional): A 1D integer array.
            class_label (optional): A dictionary mapping between
                `class_id` and a string class label.

        """
        # Basic stimulus information.
        self.n_stimuli = len(stimulus_id)
        stimulus_id = self._check_id(stimulus_id)
        # self.common_path = os.path.commonpath(stimulus_filepath)
        # TODO modify check, move away from numpy array.
        stimulus_filepath = self._check_filepath(stimulus_filepath)

        if class_id is None:
            class_id = np.zeros((self.n_stimuli), dtype=int)

        stimuli = pd.DataFrame(
            data={
                'id': stimulus_id,
                'filepath': stimulus_filepath,
                'class_id': class_id
            }
        )
        stimuli = stimuli.sort_values('id')
        self.stimuli = stimuli

        # Optional information.
        self.common_path = ''
        self.class_label = class_label
        # Optional class information. MAYBE
        # self.leaf_class_id
        # self.class_id_label
        # self.class_class

    def _check_id(self, stimulus_id):
        """Check `stimulus_id` argument.

        Returns:
            stimulus_id

        Raises:
            ValueError

        """
        if len(stimulus_id.shape) != 1:
            raise ValueError((
                "The argument `stimulus_id` must be a 1D array of "
                "integers."))

        if not issubclass(stimulus_id.dtype.type, np.integer):
            raise ValueError((
                "The argument `stimulus_id` must be a 1D array of "
                "integers."))

        return stimulus_id

    def _check_filepath(self, stimulus_filepath):
        """Check `stimulus_filepath` argument.

        Returns:
            stimulus_filepath

        Raises:
            ValueError

        """
        stimulus_filepath = np.asarray(stimulus_filepath, dtype=object)

        if len(stimulus_filepath.shape) != 1:
            raise ValueError((
                'The argument `stimulus_filepath` must have the same shape as '
                '`stimulus_id`.'))

        if stimulus_filepath.shape[0] != self.n_stimuli:
            raise ValueError((
                'The argument `stimulus_filepath` must have the same shape as '
                '`stimulus_id`.'))

        return stimulus_filepath

    def class_id(self):
        """Return class ID."""
        return self.stimuli.class_id.values

    def file_path(self):
        """Return filepaths."""
        file_path_list = self.stimuli.filepath.values.tolist()
        file_path_list = [
            Path(self.common_path) / Path(i_file) for i_file in file_path_list
        ]
        return file_path_list

    def filepath(self):
        """Return filepaths."""
        return self.file_path()

    def id(self):
        """Return stimulus id."""
        return self.stimuli.id.values

    def save(self, filepath):
        """Save the Catalog object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        max_filepath_length = len(max(self.stimuli.filepath.values, key=len))

        h5_file = h5py.File(filepath, "w")
        h5_file.create_dataset("stimulus_id", data=self.stimuli.id.values)
        h5_file.create_dataset(
            "stimulus_filepath",
            data=self.stimuli.filepath.values.astype(
                dtype="S{0}".format(max_filepath_length)
            )
        )
        h5_file.create_dataset("class_id", data=self.stimuli.class_id.values)

        if self.class_label is not None:
            max_label_length = len(max(self.class_label.values(), key=len))

            n_class = len(self.class_label)
            class_map_class_id = np.empty(n_class, dtype=np.int)
            class_map_label = np.empty(n_class, dtype="S{0}".format(
                max_label_length
            ))
            idx = 0
            for key, value in self.class_label.items():
                class_map_class_id[idx] = key
                class_map_label[idx] = value
                idx = idx + 1

            h5_file.create_dataset(
                "class_map_class_id",
                data=class_map_class_id
            )
            h5_file.create_dataset(
                "class_map_label",
                data=class_map_label
            )

        h5_file.close()

    def subset(self, idx, squeeze=False):
        """Return a subset of the catalog.

        Arguments:
            idx: An integer of boolean array indicating which stimuli
                to retain.
            squeeze (optional): A boolean indicating if IDs should be
                reassigned in order to create a contiguous set of IDs
                from [0, N[.

        Returns:
            catalog: A new catalog containing the requested subset.

        """
        catalog = copy.deepcopy(self)
        catalog.stimuli = catalog.stimuli.iloc[idx]
        catalog.n_stimuli = len(catalog.stimuli)
        if squeeze:
            catalog.stimuli.at[:, "id"] = np.arange(0, catalog.n_stimuli)
        return catalog
