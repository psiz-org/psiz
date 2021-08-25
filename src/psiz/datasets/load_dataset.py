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
"""Module for loading internally pre-defined datasets.

Functions:
    load: Load observations for the requested dataset.

Notes:
    The dataset will only be downloaded from the server if it does not
    exist locally. If it it already exists locally, no download will
    take place. If you would like to force a download, delete the
    existing local copy.

"""

import collections
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import zipfile

import numpy as np
import six
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve

import psiz.catalog
import psiz.trials


def load(
        dataset_name, cache_subdir='datasets', cache_dir=None,
        verbose=0):
    """Load observations and catalog for the requested hosted dataset.

    Arguments:
        dataset_name: The name of the hosted dataset.
        cache_subdir (optional): The subdirectory where downloaded
            datasets are cached.
        cache_dir (optional): The cache directory for PsiZ.
        verbose (optional): Controls the verbosity of printed dataset summary.

    Returns:
        obs: An RankObservations object.
        catalog: A catalog object containing information regarding the
            stimuli used to collect observations.

    """
    # Load from download cache.
    if cache_dir is None:
        cache_dir = Path.home() / Path('.psiz')
    else:
        cache_dir = Path(cache_dir)
    dataset_path = cache_dir / Path(cache_subdir, dataset_name)
    if not dataset_path.exists():
        dataset_path.mkdir(parents=True)

    obs = _fetch_obs(dataset_name, cache_dir, cache_subdir)
    catalog = _fetch_catalog(dataset_name, cache_dir, cache_subdir)

    if verbose > 0:
        print("Dataset Summary")
        print('  n_stimuli: {0}'.format(catalog.n_stimuli))
        print('  n_trial: {0}'.format(obs.n_trial))

    return (obs, catalog)


def _fetch_catalog(dataset_name, cache_dir, cache_subdir):
    """Fetch catalog for the requested dataset.

    Arguments:
        dataset_name: The name of the dataset to load.
        cache_dir: The cache directory for PsiZ.
        cache_subdir: The subdirectory where downloaded datasets are
            cached.

    Returns:
        catalog: A Catalog object.

    """
    fname = "catalog.hdf5"

    dataset_exists = True
    if dataset_name == "birds-12":
        origin = "https://osf.io/xek89/download"
    elif dataset_name == "birds-16":
        origin = "https://osf.io/473vh/download"
    elif dataset_name == "skin_lesions":
        origin = "https://osf.io/5grsp/download"
    elif dataset_name == "rocks_Nosofsky_etal_2016":
        origin = "https://osf.io/vw28u/download"
    elif dataset_name == "ilsvrc_val_v0_1":
        origin = "https://osf.io/bf3e2/download"
    elif dataset_name == "ilsvrc_val_v0_2":
        origin = "https://osf.io/bf3e2/download"
    else:
        dataset_exists = False

    if dataset_exists:
        path = _get_file(
            os.path.join(dataset_name, fname), origin,
            cache_subdir=cache_subdir, extract=True,
            cache_dir=cache_dir
        )
        catalog = psiz.catalog.load_catalog(path)
    else:
        raise ValueError(
            'The requested dataset `{0}` may not exist since the '
            'corresponding catalog.hdf5 file does not '
            'exist.'.format(dataset_name)
        )

    return catalog


def _fetch_obs(dataset_name, cache_dir, cache_subdir):
    """Fetch observations for the requested dataset.

    Arguments:
        dataset_name: The name of the dataset to load.
        cache_dir: The cache directory for PsiZ.
        cache_subdir: The subdirectory where downloaded datasets are
            cached.

    Returns:
        obs: An RankObservations object.

    """
    fname = 'obs.hdf5'

    dataset_exists = True
    if dataset_name == "birds-12":
        origin = "https://osf.io/apd3g/download"
    elif dataset_name == "birds-16":
        origin = "https://osf.io/nz4gy/download"
    elif dataset_name == "skin_lesions":
        origin = "https://osf.io/nbps4/download"
    elif dataset_name == "rocks_Nosofsky_etal_2016":
        origin = "https://osf.io/jauvh/download"
    elif dataset_name == "ilsvrc_val_v0_1":
        origin = "https://osf.io/ej6sz/download"
    elif dataset_name == "ilsvrc_val_v0_2":
        origin = "https://osf.io/x6dht/download"
    else:
        dataset_exists = False

    if dataset_exists:
        path = _get_file(
            os.path.join(dataset_name, fname), origin,
            cache_subdir=cache_subdir, extract=True, cache_dir=cache_dir
        )
        obs = psiz.trials.load_trials(path)
    else:
        raise ValueError(
            'The requested dataset `{0}` may not exist since the '
            'corresponding obs.hdf5 file does not '
            'exist.'.format(dataset_name)
        )

    return obs


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
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
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


def _get_file(
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
        download = False
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        class ProgressTracker():
            """Download progress bar."""
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg)) from e
            except URLError as e:
                raise Exception(
                    error_msg.format(origin, e.errno, e.reason)
                ) from e
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


class Progbar():
    """Displays a progress bar."""

    def __init__(
            self, target, width=30, verbose=1, interval=0.05,
            stateful_metrics=None):
        """Initialize.

        Arguments:
            target: Total number of steps expected, None if unknown.
            width: Progress bar width on screen.
            verbose: Degree of verbosity.
            stateful_metrics: Iterable of string names of metrics that
                should *not* be averaged over time. Metrics in this
                list will be displayed as-is. All others will be
                averaged by the progbar before display.
            interval: Minimum visual progress update interval (in
                seconds).

        """
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = (
            (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty())
            or 'ipykernel' in sys.modules
        )
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
            if (
                (now - self._last_update < self.interval)
                and (self.target is not None)
                and (current < self.target)
            ):
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
                displayed_bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    displayed_bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        displayed_bar += '>'
                    else:
                        displayed_bar += '='
                displayed_bar += ('.' * (self.width - prog_width))
                displayed_bar += ']'
            else:
                displayed_bar = '%7d/Unknown' % current

            self._total_width = len(displayed_bar)
            sys.stdout.write(displayed_bar)

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
