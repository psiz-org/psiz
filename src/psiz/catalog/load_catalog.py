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

Functions:
    load_catalog: Load a catalog.

"""

import h5py
import numpy as np

from psiz.catalog.catalog import Catalog


def load_catalog(filepath, verbose=0):
    """Load data saved via the save method.

    The loaded data is instantiated as a Catalog object.

    Arguments:
        filepath: The location of the hdf5 file to load.
        verbose (optional): Controls the verbosity of printed summary.

    Returns:
        Loaded catalog.

    """
    h5_file = h5py.File(filepath, "r")
    stimulus_id = h5_file["stimulus_id"][()]
    # pylint: disable=no-member
    stimulus_filepath = h5_file["stimulus_filepath"].asstr()[:]
    class_id = h5_file["class_id"][()]

    try:
        class_map_class_id = h5_file["class_map_class_id"][()]
        class_map_label = h5_file["class_map_label"][()]
        class_label_dict = {}
        for idx in np.arange(len(class_map_class_id)):
            class_label_dict[class_map_class_id[idx]] = (
                class_map_label[idx].decode('ascii')
            )
    except KeyError:
        class_label_dict = None

    catalog = Catalog(
        stimulus_id, stimulus_filepath, class_id, class_label_dict)
    h5_file.close()

    if verbose > 0:
        print("Catalog Summary")
        print('  n_stimuli: {0}'.format(catalog.n_stimuli))
        print('')
    return catalog
