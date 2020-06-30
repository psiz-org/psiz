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
"""Module for handling model restarts.

Classes:
    Restarter: A class for performing restarts.
    FitTracker: A class for keeping track of the best performing
        restart(s).

Functions:
    set_from_record: Set the weights of a model using the weights
        stored in a record.

"""

import copy
import os
import time

import numpy as np
import tensorflow as tf

import psiz.utils


class Restarter(object):
    """Object for handling TensorFlow model restarts.

    Attributes:
        n_restart: An integer specifying the number of
            restarts to use for the inference procedure. Since the
            embedding procedure can get stuck in local optima,
            multiple restarts help find the global optimum.
        n_record: An integer indicating how many best-
            performing models to track.

    Methods:
        fit: Fit the provided model to the observations.

    """

    def __init__(
            self, model, compile_kwargs={}, monitor='loss', n_restart=10,
            n_record=1, do_init=False, custom_objects={}, weight_dir=None):
        """Initialize.

        Arguments:
            model: A compiled TensorFlow model.
            compile_kwargs (optional): Key-word arguments for compile
                method.
            monitor (optional): The value to monitor and use as a basis
                for selecting the best restarts.
            n_restart (optional): An integer indicating the number of
                independent restarts to perform.
            n_record (optional): An integer indicating the number of
                best performing restarts to record.
            do_init (optional): A Boolean variable indicating whether
                the initial model and it's corresponding performance
                should be included as a candidate in the set of
                restarts.
            custom_objects (optional): Custom objects for model
                creation.

        """
        # Make sure n_record is not greater than n_restart.
        n_record = np.minimum(n_record, n_restart)

        self.model = model
        self.optimizer = compile_kwargs.pop(
            'optimizer', tf.keras.optimizers.RMSprop()
        )
        self.stateless_compile_kwargs = compile_kwargs
        self.monitor = monitor
        self.n_restart = n_restart
        self.n_record = n_record
        self.do_init = do_init
        self.custom_objects = custom_objects
        if weight_dir is None:
            weight_dir = '/tmp/psiz/restarts'
        self.weight_dir = weight_dir

    def fit(
            self, x=None, validation_data=None, callbacks=[], verbose=0,
            **kwargs):
        """Fit the embedding model to the observations using restarts.

        Arguments:
            x: A a tf.data.Dataset object.
            callbacks: A list of callbacks.
            verbose: Verbosity of output.
            **kwargs: Any additional keyword arguments.

        Returns:
            tracker: A record of the best restarts.

        """
        start_time_s = time.time()
        tracker = FitTracker(self.n_record, self.monitor)

        # Initial evaluation.
        if self.do_init:
            # Create new optimizer.
            optimizer_re = _new_optimizer(self.optimizer)

            # Compile model.
            self.model.compile(
                optimizer=optimizer_re, **self.stateless_compile_kwargs
            )

            logs = {}
            n_epoch = 0
            logs['epoch'] = 0
            logs['total_duration_s'] = 0
            logs['ms_per_epoch'] = 0
            train_metrics = model_re.evaluate(
                x=x, verbose=0, return_dict=True
            )
            logs.update(train_metrics)
            if validation_data is not None:
                val_metrics = model_re.evaluate(
                    x=validation_data, verbose=0, return_dict=True
                )
                val_metrics = _append_prefix(val_metrics, 'val_')
                logs.update(val_metrics)

            fp_weights = os.path.join(self.weight_dir, 'restart_init')
            model_re.save_weights(fp_weights, overwrite=True)

            # Update fit record with latest restart.
            tracker.update_state(logs, fp_weights, is_init=True)

        if verbose == 1:
            progbar = psiz.utils.ProgressBarRe(
                self.n_restart, prefix='Progress:', length=50
            )
            progbar.update(0)

        # Run multiple restarts of embedding algorithm.
        for i_restart in range(self.n_restart):
            if verbose > 1:
                print('        Restart {0}'.format(i_restart))

            # Create new model.
            model_re = _new_model(
                self.model, custom_objects=self.custom_objects
            )

            # Create new optimizer.
            optimizer_re = _new_optimizer(self.optimizer)

            # Compile model.
            model_re.compile(
                optimizer=optimizer_re, **self.stateless_compile_kwargs
            )

            # Reset callbacks.
            for callback in callbacks:
                callback.reset(i_restart)

            fit_start_time_s = time.time()
            history = model_re.fit(
                x=x, validation_data=validation_data, callbacks=callbacks,
                verbose=verbose-1, **kwargs
            )
            total_duration = time.time() - fit_start_time_s
            logs = {}
            n_epoch = np.max(history.epoch) - np.min(history.epoch) + 1
            logs['epoch'] = n_epoch
            logs['total_duration_s'] = int(total_duration)
            logs['ms_per_epoch'] = int(1000 * total_duration / n_epoch)
            train_metrics = model_re.evaluate(
                x=x, verbose=0, return_dict=True
            )
            logs.update(train_metrics)
            if validation_data is not None:
                val_metrics = model_re.evaluate(
                    x=validation_data, verbose=0, return_dict=True
                )
                val_metrics = _append_prefix(val_metrics, 'val_')
                logs.update(val_metrics)
            fp_weights = os.path.join(
                self.weight_dir, 'restart_{0}'.format(i_restart)
            )
            model_re.save_weights(fp_weights, overwrite=True)

            # Update fit record with latest restart.
            tracker.update_state(logs, fp_weights, is_init=False)

            if verbose == 1:
                progbar.update(i_restart + 1)

        # Sort records from best to worst and grab best.
        tracker.sort()
        monitor_best = tracker.record[self.monitor][0]
        epoch_best = tracker.record['epoch'][0]
        model_re.load_weights(tracker.record['weights'][0])
        self.model = model_re

        fit_duration = time.time() - start_time_s
        summary_mean = tracker.result(np.mean)
        summary_std = tracker.result(np.std)

        if (verbose > 0):
            print('    Restart Summary')
            if not tracker.beat_init:
                print('    Did not beat initial model.')
            print(
                '    n_valid_restart {0:.0f} | '
                'total_duration: {1:.0f} s'.format(
                    tracker.count, fit_duration
                )
            )
            print(
                '    best | n_epoch: {0:.0f} | '
                '{1}: {2:.4f}'.format(
                    epoch_best, self.monitor, monitor_best
                )
            )
            print(
                '    mean \u00B1stddev | n_epoch: {0:.0f} \u00B1{1:.0f} | '
                '{2}: {3:.4f} \u00B1{4:.4f} | {5:.0f} \u00B1{6:.0f} s | '
                '{7:.0f} \u00B1{8:.0f} ms/epoch'.format(
                    summary_mean['epoch'], summary_std['epoch'],
                    self.monitor, summary_mean[self.monitor],
                    summary_std[self.monitor],
                    summary_mean['total_duration_s'],
                    summary_std['total_duration_s'],
                    summary_mean['ms_per_epoch'], summary_std['ms_per_epoch']
                )
            )

        return tracker


class FitTracker(object):
    """Class for keeping track of best restarts.

    Methods:
        update_state: Update the records with the provided restart.
        sort: Sort the records from best to worst.

    """

    def __init__(self, n_record, monitor):
        """Initialize.

        Arguments:
            n_record: Integer indicating the number of top restarts to
                record.
            monitor: String indicating the value to use in order to
                select the best performing restarts.

        """
        self.n_record = n_record
        self.monitor = monitor
        self.record = {
            monitor: np.inf * np.ones([n_record]),
            'weights': [None] * n_record
        }
        self.summary = {monitor: []}
        self.count = 0.
        self.beat_init = None
        self.init_loss = np.inf
        super().__init__()

    def update_state(self, logs, fp_weights, is_init=False):
        """Update record with incoming data.

        Arguments:
            logs: A dictionary of non-weight properties to track.
            fp_weights: A filepath to weights.
            is_init: Boolean indicating if the update is an initial
                evaluation.

        Notes:
            The update_state method does not worry about keeping the
            records sorted. If the records need to be sorted, use the
            sort method.

        """
        loss_monitor = logs[self.monitor]

        # Update aggregate summary if result is not nan.
        if not is_init and not np.isnan(loss_monitor):
            self.count = self.count + 1.
            for k, v in logs.items():
                if k not in self.summary:
                    self.summary[k] = []
                self.summary[k].append(v)

        dmy_idx = np.arange(self.n_record)
        locs_is_worse = np.greater(
            self.record[self.monitor], loss_monitor
        )

        if np.sum(locs_is_worse) > 0:
            # Identify worst restart in record.
            idx_eligable_as_worst = dmy_idx[locs_is_worse]
            idx_idx_worst = np.argmax(self.record[self.monitor][locs_is_worse])
            idx_worst = idx_eligable_as_worst[idx_idx_worst]

            # Replace worst restart with incoming restart.
            for k, v in logs.items():
                if k not in self.record:
                    self.record[k] = np.inf * np.ones([self.n_record])
                self.record[k][idx_worst] = v
            self.record['weights'][idx_worst] = fp_weights

        if is_init:
            self.init_loss = loss_monitor
            self.beat_init = False
        else:
            if loss_monitor < self.init_loss:
                self.beat_init = True

    def sort(self):
        """Sort the records from best to worst."""
        idx_sort = np.argsort(self.record[self.monitor])
        for k, v in self.record.items():
            if k == 'weights':
                self.record['weights'] = [
                    self.record['weights'][i] for i in idx_sort
                ]
            else:
                self.record[k] = self.record[k][idx_sort]

    def result(self, fnc):
        """Return function of all summary fields.

        Arguments:
            fnc: Function that takes a list of items and returns a
                scalar.

        """
        d = {}
        for k, v in self.summary.items():
            d[k] = fnc(v)
        return d


def set_from_record(model, tracker, idx):
    """Set embedding parameters using a record.

    Arguments:
        model: An appropriate TensorFlow model.
        tracker: An appropriate psiz.models.FitTracker object.
        idx: An integer indicating which record to use.

    """
    model.load_weights(tracker.record['weights'][idx])


def _new_model(model, custom_objects={}):
    """Create new model."""
    with tf.keras.utils.custom_object_scope(custom_objects):
        new_model = model.from_config(model.get_config())
    return new_model


def _new_optimizer(optimizer, custom_objects={}):
    """Create new optimizer."""
    config = tf.keras.optimizers.serialize(optimizer)
    with tf.keras.utils.custom_object_scope(custom_objects):
        new_optimizer = tf.keras.optimizers.deserialize(config)
    return new_optimizer


def _append_prefix(val_metrics, prefix):
    """Append prefix to dictionary keys."""
    d = {}
    for k, v in val_metrics.items():
        d[prefix + k] = v
    return d
