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
# ==============================================================================

"""Module for handling model restarts.

Classes:
    Restarter: A class for performing restarts.
    FitRecord: A class for keeping track of the best performing
        restart(s).

Functions:
    set_from_record: Set the weights of a model using the weights
        stored in a record.

"""

import copy

import numpy as np

import psiz.utils


class Restarter(object):
    """Object for handling restarts.

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

    def __init__(self, emb, monitor, n_restart=10, n_record=1, do_init=False):
        """Initialize.

        Arguments:
            emb: A compiled model.
            monitor: The value to monitor and use as a basis for
                selecting the best restarts.
            n_restart (optional): An integer indicating the number of
                independent restarts to perform.
            n_record (optional): An integer indicating the number of
                best performing restarts to record.
            do_init (optional): A Boolean variable indicating whether
                the initial model and it's corresponding performance
                should be included as a candidate in the set of
                restarts.

        """
        # Make sure n_record is not greater than n_restart.
        n_record = np.minimum(n_record, n_restart)

        self.emb = emb
        self.monitor = monitor
        self.n_restart = n_restart
        self.n_record = n_record
        self.do_init = do_init
        self.log_dir = copy.copy(emb.log_dir)

    def fit(self, obs_train, verbose=0, **kwargs):
        """Fit the embedding model to the observations using restarts.

        Arguments: TODO
            obs: A psiz.trials.Observations object.
            verbose: Verbosity of output.
            **kwargs: Any additional keyword arguments.

        Returns:
            fit_record: A record of the best restarts.

        """
        fit_record = FitRecord(self.n_record, self.monitor)

        # Grab optimizer configuration.
        self.optimizer_config = self.emb.optimizer.get_config()

        # Initial evaluation. TODO
        if self.do_init:
            # Update record with initialization values.
            history = None
            weights = None
            fit_record.update(history, weights, is_init=True)
            if (verbose > 2):
                print('        Initialization')
                print(
                    '        '
                    '     --     | '
                    'loss_train: {0: .6f} | loss_val: {1: .6f}'.format(
                        loss_train_init, loss_val_init)
                )
                print('')

        if verbose > 0 and verbose < 3:
            progbar = psiz.utils.ProgressBar(
                self.n_restart, prefix='Progress:', length=50
            )
            progbar.update(0)

        # Run multiple restarts of embedding algorithm.
        for i_restart in range(self.n_restart):
            if (verbose > 2):
                print('        Restart {0}'.format(i_restart))
            if verbose > 0 and verbose < 3:
                progbar.update(i_restart + 1)

            # Reset trainable weights.
            self.emb.reset_weights()
            # Reset optimizer.
            self.emb.optimizer = type(
                self.emb.optimizer
            ).from_config(self.optimizer_config)

            # Distinguish between restart by setting log_dir for TensorBoard.
            self.emb.log_dir = '{0}/{1}'.format(self.log_dir, i_restart)
            history = self.emb.fit(obs_train, verbose=verbose, **kwargs)
            weights = self.emb.weights

            # Update fit record with latest restart.
            fit_record.update(history, weights)

        # Sort records from best to worst and grab best.
        fit_record.sort()
        loss_train_best = fit_record.record['train_loss'][0]
        loss_val_best = fit_record.record['val_loss'][0]
        epoch_best = fit_record.record['epoch'][0]
        self.emb.set_weights(fit_record.record['weights'][0])

        # TODO handle time stats
        # emb.fit_duration = time.time() - start_time_s  # TODO
        # emb.fit_record = fit_record

        if (verbose > 1):
            if fit_record.beat_init:
                print(
                    '    Best Restart\n        n_epoch: {0} | '
                    'loss: {1: .6f} | loss_val: {2: .6f}'.format(
                        epoch_best, loss_train_best, loss_val_best
                    )
                )
            else:
                print('    Did not beat initialization.')

        # Clean up.
        self.emb.log_dir = self.log_dir

        return fit_record


class FitRecord(object):
    """Class for keeping track of best restarts.

    Methods:
        update: Update the records with the provided restart.
        sort: Sort the records from best to worst.

    """

    def __init__(self, n_record, monitor):
        """Initialize.

        Arguments:
            n_record: Integer indicating the number of top restarts
                to record.
            monitor: String indicating the value to use in order to
                select the best performing restarts.

        """
        self.n_record = n_record
        self.monitor = monitor
        self.record = {
            monitor: np.inf * np.ones([n_record]),
            'weights': [None] * n_record
        }
        self.beat_init = None
        self.init_loss = np.inf
        super().__init__()

    def update(self, history, weights, is_init=False):
        """Update record with incoming data.

        Arguments:
            loss_train: TODO
            loss_val: TODO
            weights: TODO
            is_init: TODO

        Notes:
            The update method does not worry about keeping the
            records sorted. If the records need to be sorted, use the
            sort method.

        """
        loss_monitor = history.final[self.monitor]
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
            for k, v in history.final.items():
                if k not in self.record:
                    self.record[k] = np.inf * np.ones([self.n_record])
                self.record[k][idx_worst] = v
            self.record['weights'][idx_worst] = weights

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


def set_from_record(emb, fit_record, idx):
    """Set embedding parameters using a record.

    Arguments:
        emb: TODO
        fit_record: An appropriate psiz.models.FitRecord object.
        idx: An integer indicating which record to use.

    """
    emb.set_weights(fit_record.record['weights'][idx])
