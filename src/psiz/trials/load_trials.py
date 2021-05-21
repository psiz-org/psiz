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
"""Module of core `trials` functionality.

Functions:
    load_trials: Load a hdf5 file that was saved using the `save` class
        method.

"""
import h5py

from psiz.trials.similarity.rank.rank_docket import RankDocket
from psiz.trials.similarity.rank.rank_observations import RankObservations
from psiz.trials.similarity.rate.rate_docket import RateDocket
from psiz.trials.similarity.rate.rate_observations import RateObservations
from psiz.trials.experimental.trial_dataset import TrialDataset


def load_trials(filepath, verbose=0):
    """Load data saved via the save method.

    The loaded data is instantiated as a concrete class of
    psiz.trials.SimilarityTrials.

    Arguments:
        filepath: The location of the hdf5 file to load.
        verbose (optional): Controls the verbosity of printed summary.

    Returns:
        Loaded trials.

    Raises:
        ValueError

    """
    f = h5py.File(filepath, "r")
    # Retrieve trial class name. Fall back to "trial_type" field for legacy
    # implementations.
    try:
        # Encoding/read rules changed in h5py 3.0, requiring asstr() call.
        try:
            class_name = f["class_name"].asstr()[()]
        except AttributeError:
            class_name = f["class_name"][()]
    except KeyError:
        try:
            class_name = f["trial_type"].asstr()[()]
        except AttributeError:
            class_name = f["trial_type"][()]
    f.close()

    # Handle legacy class names.
    if class_name == "Docket":
        class_name = "RankDocket"
    elif class_name == "Observations":
        class_name = "RankObservations"

    # Route to appropriate class.
    custom_objects = {
        'RankDocket': RankDocket,
        'RankObservations': RankObservations,
        'RateDocket': RateDocket,
        'RateObservations': RateObservations,
        'TrialDataset': TrialDataset,
    }
    if class_name in custom_objects:
        trial_class = custom_objects[class_name]
    else:
        print(
            'NotImplementedError: class_name={0} is not implemented.'.format(
                class_name
            )
        )
        raise NotImplementedError

    trials = trial_class.load(filepath)
    return trials
