# -*- coding: utf-8 -*-
# Copyright 2019 The PsiZ Authors. All Rights Reserved.
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
# ==============================================================================

"""Module for loading datasets.

Functions:
    load_dataset: Load observations for the requested dataset.

Todo:
    - create .psiz directory on installation (if it doesn' exist)
    - migrate to scalable solution described in roadmap.
    - Give credit to keras library.
    - Query public (read-only) database.

"""

import os
import copy
import shutil
import sys
import tarfile
import time
import zipfile
import collections
from urllib import request

import numpy as np
import pandas as pd
import six
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen
from six.moves.urllib.request import urlretrieve
import h5py

from psiz.trials import load_trials

HOST_URL = "https://www.psiz.org/datasets/"


class Catalog(object):
    """Class to keep track of stimuli information.

    Attributes:
        n_stimuli:  The number of unique stimuli.
        stimuli: Pandas dataframe containing information about the
            stimuli:
            id: A unique stimulus id.
            filepath: The filepath for the corresponding stimulus.

    """

    def __init__(
            self, stimulus_id, stimulus_filepath, class_id=None,
            class_label=None):
        """Initialize.

        Arguments:
            stimulus_id: A 1D integer array.
                shape=(n_stimuli,)
            stimulus_filepath: A 1D list of strings.
                len=n_stimuli
            class_id (optional): A 1D integer array.
            class_label (optional): A dictionary mapping between class_id and a
                string label.
        """
        # Basic stimulus information.
        self.n_stimuli = len(stimulus_id)
        stimulus_id = self._check_stimulus_id(stimulus_id)
        stimulus_filepath = self._check_stimulus_path(stimulus_filepath)
        if class_id is None:
            class_id = np.zeros((self.n_stimuli))
        stimuli = pd.DataFrame(
            data={
                'id': stimulus_id,
                'filepath': stimulus_filepath,
                'class_id': class_id
            }
        )
        stimuli = stimuli.sort_values('id')
        self.stimuli = stimuli
        self.class_label = class_label
        # Optional class information. TODO MAYBE
        # self.leaf_class_id
        # self.class_id_label
        # self.class_class

    def _check_stimulus_id(self, stimulus_id):
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

        n_stimuli = len(stimulus_id)

        is_contiguous = False
        if np.array_equal(np.unique(stimulus_id), np.arange(0, n_stimuli)):
            is_contiguous = True
        if not is_contiguous:
            raise ValueError((
                'The argument `stimulus_id` must contain a contiguous set of '
                'integers [0, n_stimuli[.'))
        return stimulus_id

    def _check_stimulus_path(self, stimulus_filepath):
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

    def save(self, filepath):
        """Save the Catalog object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        max_filepath_length = len(max(self.stimuli.filepath.values, key=len))

        f = h5py.File(filepath, "w")
        f.create_dataset("stimulus_id", data=self.stimuli.id.values)
        f.create_dataset(
            "stimulus_filepath",
            data=self.stimuli.filepath.values.astype(
                dtype="S{0}".format(max_filepath_length)
            )
        )
        f.create_dataset("class_id", data=self.stimuli.class_id.values)

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

            f.create_dataset(
                "class_map_class_id",
                data=class_map_class_id
            )
            f.create_dataset(
                "class_map_label",
                data=class_map_label
            )

        f.close()

    def subset(self, idx, squeeze=False):
        """Return a subset of catalog with new stimulus IDs."""
        catalog = copy.deepcopy(self)
        catalog.stimuli = catalog.stimuli.iloc[idx]
        catalog.n_stimuli = len(catalog.stimuli)
        if squeeze:
            catalog.stimuli.at[:, "id"] = np.arange(0, catalog.n_stimuli)
        return catalog


def load_catalog(filepath, verbose=0):
    """Load data saved via the save method.

    The loaded data is instantiated as a Catalog object.

    Arguments:
        filepath: The location of the hdf5 file to load.
        verbose (optional): Controls the verbosity of printed summary.

    Returns:
        Loaded catalog.

    """
    f = h5py.File(filepath, "r")
    stimulus_id = f["stimulus_id"][()]
    stimulus_filepath = f["stimulus_filepath"][()].astype('U')
    class_id = f["class_id"][()]

    try:
        class_map_class_id = f["class_map_class_id"][()]
        class_map_label = f["class_map_label"][()]
        class_label_dict = {}
        for idx in np.arange(len(class_map_class_id)):
            class_label_dict[class_map_class_id[idx]] = (
                class_map_label[idx].decode('ascii')
            )
    except KeyError:
        class_label_dict = None

    catalog = Catalog(
        stimulus_id, stimulus_filepath, class_id, class_label_dict)
    f.close()

    if verbose > 0:
        print("Catalog Summary")
        print('  n_stimuli: {0}'.format(catalog.n_stimuli))
        print('')
    return catalog


def _fetch_catalog(dataset_name, cache_subdir='datasets', cache_dir=None):
    """Fetch catalog for the requested dataset.

    Arguments:
        dataset_name: The name of the dataset to load.

    Returns:
        catalog: An Catalog object.

    """
    fname = "catalog.hdf5"

    dataset_exists = True
    if dataset_name == "birds-12":
        origin = HOST_URL + "birds-12/" + fname
    elif dataset_name == "birds-16":
        origin = HOST_URL + "birds-16/" + fname
    elif dataset_name == "lesions":
        origin = HOST_URL + "lesions/" + fname
    elif dataset_name == "rocks_Nosofsky_etal_2016":
        origin = HOST_URL + "rocks_Nosofsky_etal_2016/" + fname
    else:
        dataset_exists = False

    if dataset_exists:
        path = get_file(
            os.path.join(dataset_name, fname), origin,
            cache_subdir=cache_subdir, extract=True,
            cache_dir=cache_dir
        )
        catalog = load_catalog(path)
    else:
        raise ValueError(
            'The requested dataset `{0}` may not exist since the '
            'corresponding catalog.hdf5 file does not '
            'exist.'.format(dataset_name)
        )

    return catalog


def _fetch_obs(dataset_name, cache_subdir='datasets', cache_dir=None):
    """Fetch observations for the requested dataset.

    Arguments:
        dataset_name: The name of the dataset to load.

    Returns:
        obs: An Observations object.

    """
    fname = 'obs.hdf5'

    dataset_exists = True
    if dataset_name == "birds-12":
        origin = HOST_URL + "birds-12/" + fname
    elif dataset_name == "birds-16":
        origin = HOST_URL + "birds-16/" + fname
    elif dataset_name == "lesions":
        origin = HOST_URL + "lesions/" + fname
    elif dataset_name == "rocks_Nosofsky_etal_2016":
        origin = HOST_URL + "rocks_Nosofsky_etal_2016/" + fname
    else:
        dataset_exists = False

    if dataset_exists:
        path = get_file(
            os.path.join(dataset_name, fname), origin,
            cache_subdir=cache_subdir, extract=True, cache_dir=cache_dir
        )
        obs = load_trials(path)
    else:
        raise ValueError(
            'The requested dataset `{0}` may not exist since the '
            'corresponding obs.hdf5 file does not '
            'exist.'.format(dataset_name)
        )

    return obs


def load_dataset(
        fp_dataset, cache_subdir='datasets', cache_dir=None,
        verbose=0):
    """Load observations and catalog for the requested hosted dataset.

    Arguments:
        fp_dataset: The filepath to the dataset. If loading a hosted
            dataset, just provide the name of the dataset.
        cache_subdir (optional): The subdirectory where downloaded
            datasets are cached.
        cache_dir (optional): The cache directory for PsiZ.
        verbose (optional): Controls the verbosity of printed dataset summary.

    Returns:
        obs: An Observations object.
        catalog: A catalog object containing information regarding the
            stimuli used to collect observations.

    """
    # Load from download cache.
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.psiz')
    dataset_path = os.path.join(cache_dir, cache_subdir, fp_dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    obs = _fetch_obs(fp_dataset, cache_subdir, cache_dir)
    catalog = _fetch_catalog(fp_dataset, cache_subdir, cache_dir)

    if verbose > 0:
        print("Dataset Summary")
        print('  n_stimuli: {0}'.format(catalog.n_stimuli))
        print('  n_trial: {0}'.format(obs.n_trial))

    return (obs, catalog)


def _extract_archive(file_path, path='.', archive_format='auto'):
    """Extract an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    Arguments:
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.

    """
    if archive_format is None:
        return False
    if archive_format is 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type is 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type is 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError,
                        KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def get_file(
        fname, origin, untar=False, cache_subdir='datasets', extract=False,
        archive_format='auto', cache_dir=None):
    """Download a file from a URL if it not already in the cache."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.psiz')

    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.psiz')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # TODO
        download = False
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size is -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(
            self, target, width=30, verbose=1, interval=0.05,
            stateful_metrics=None):
        """Initialize."""
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Update the progress bar.

        Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        """Add progress."""
        self.update(self._seen_so_far + n, values)
