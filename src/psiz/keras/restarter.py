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
    Restarter: A class for performing restarts with Keras models.

"""

import os
import time

import numpy as np
import tensorflow as tf

from psiz.utils.fit_tracker import FitTracker
from psiz.utils.progress_bar_re import ProgressBarRe


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

    Note:
        To minimize memory consumption, model weights for each restart
        are stored on disk in a tempory file.

    """

    def __init__(
            self, model, compile_kwargs=None, monitor='loss', n_restart=10,
            n_record=None, do_init=False, custom_objects=None,
            weight_dir='/tmp/psiz/restarts'):
        """Initialize.

        Arguments:
            model: A compiled TensorFlow model.
            compile_kwargs (optional): Key-word arguments for compile
                method.
            monitor (optional): The value to monitor and use as a basis
                for selecting the best restarts.
            n_restart (optional): An integer indicating the number of
                independent restarts to perform.
            n_record (optional): A positive integer indicating the
                number of best performing restarts to record. By
                default, all restarts will be recorded. In general,
                the default value is the best.
            do_init (optional): A Boolean variable indicating whether
                the initial model state  should be included as a
                candidate in the set of restarts. This is useful if
                you are performing online training and want to
                "lock-in" past progress.
            custom_objects (optional): Custom objects for model
                creation.

        """
        if n_record is None:
            n_record = n_restart
        else:
            # Make sure `n_record` is not greater than `n_restart`.
            n_record = np.minimum(n_record, n_restart)

        self.model = model
        compile_kwargs = compile_kwargs or {}
        self.optimizer = compile_kwargs.pop(
            'optimizer', tf.keras.optimizers.RMSprop()
        )
        self.stateless_compile_kwargs = compile_kwargs
        self.monitor = monitor
        self.n_restart = n_restart
        self.n_record = n_record
        self.do_init = do_init
        self.custom_objects = custom_objects or {}
        self.weight_dir = weight_dir

    def fit(
            self, x=None, validation_data=None, callbacks=None, verbose=0,
            **kwargs):
        """Fit the embedding model to the observations using restarts.

        Arguments:
            x: A a tf.data.Dataset object.
            callbacks: A list of callbacks.
            verbose: Verbosity of output.
            **kwargs: Any additional keyword arguments.

        Returns:
            tracker: A psiz.utils.FitTracker object that contains a
                record of the best restarts.

        """
        callbacks = callbacks or []
        start_time_s = time.time()
        tracker = FitTracker(self.n_record, self.monitor)

        # Initial evaluation.
        if self.do_init:
            # Set restart model to the model that was passed in.
            model_re = self.model

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
            progbar = ProgressBarRe(
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
                verbose=np.maximum(0, verbose - 1), **kwargs
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


def _new_model(model, custom_objects=None):
    """Create new model."""
    with tf.keras.utils.custom_object_scope(custom_objects or {}):
        new_model = model.from_config(model.get_config())
    return new_model


def _new_optimizer(optimizer, custom_objects=None):
    """Create new optimizer."""
    config = tf.keras.optimizers.serialize(optimizer)
    with tf.keras.utils.custom_object_scope(custom_objects or {}):
        new_optimizer = tf.keras.optimizers.deserialize(config)
    return new_optimizer


def _append_prefix(val_metrics, prefix):
    """Append prefix to dictionary keys."""
    d = {}
    for k, v in val_metrics.items():
        d[prefix + k] = v
    return d
