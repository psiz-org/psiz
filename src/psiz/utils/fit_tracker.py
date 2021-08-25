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
    FitTracker: A class for keeping track of the best performing
        restart(s).

"""

import numpy as np


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

    def sort(self, ascending=True):
        """Sort the records from best to worst."""
        if ascending:
            idx_sort = np.argsort(self.record[self.monitor])
        else:
            idx_sort = np.argsort(-self.record[self.monitor])

        for k, _ in self.record.items():
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
