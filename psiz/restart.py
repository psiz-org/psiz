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

"""Module for handling (pararallelized) model restarts.

Classes:
    Restarter: A class for performing restarts.
    FitRecord: A class for keeping track of the best performing
        restart(s).

"""


class Restarter(object):
    """Object for handling restarts.

    Attributes:
        n_restart (optional): An integer specifying the number of
            restarts to use for the inference procedure. Since the
            embedding procedure can get stuck in local optima,
            multiple restarts help find the global optimum.
        n_record (optional): An integer indicating how many best-
            performing models to track.
        n_worker (optional): An integer indicating the number of models
            to train in parallel.

    Methods:
        fit: Fit the provided model to the observations.

    """

    def __init__(self, n_restart=10, n_record=1, n_worker=1):
        """Initialize."""
        # Make sure n_record is not greater than n_restart.
        n_record = np.minimum(n_record, n_restart)

        self.n_restart = n_restart
        self.n_record = n_record
        self.n_worker = n_worker

    def fit(self, emb, obs, max_epoch=5000, patience=10):
        """Fit the embedding model to the observations using restarts.

        Arguments:
            emb: A compiled model.
            obs: A psiz.trials.Observations object.
            max_epoch (optional): The maximum number of epochs for each
                restart.
            patience (optional): The patience for each restart.

        Returns:
            emb_best: The best embedding model.

        """
        fit_record = FitRecord(self, n_stimuli, n_dim, n_group, n_record)

        # Create train/val split
        # =============================================================
        # Grab configuration information. Need too grab here because
        # configuration mapping may change when grabbing train and
        # validation subset.
        (obs_config_list, obs_config_idx) = self._grab_config_info(obs)
        tf_config = self._prepare_config(obs_config_list)

        # Partition observations into train and validation set to
        # control early stopping of embedding algorithm.
        skf = StratifiedKFold(n_splits=10)
        (train_idx, val_idx) = list(
            skf.split(obs.stimulus_set, obs_config_idx)
        )[0]
        obs_train = obs.subset(train_idx)
        config_idx_train = obs_config_idx[train_idx]
        obs_val = obs.subset(val_idx)
        config_idx_val = obs_config_idx[val_idx]
        # =============================================================


        # Initial evaluation.
        if do_init:
            model = self._build_model(tf_config, init_mode='warm')
            prob_train_init = model(tf_inputs_train)
            loss_train_init = (
                self.loss(prob_train_init, tf_inputs_train[3]) +
                self.regularizer(model)
            ).numpy()
            prob_val_init = model(tf_inputs_val)
            loss_val_init = (
                self.loss(prob_val_init, tf_inputs_val[3]) +
                self.regularizer(model)
            ).numpy()
            # NOTE: The combined loss is based on the fact that there are
            # 10 splits.
            loss_combined_init = .9 * loss_train_init + .1 * loss_val_init

            # Update record with initialization values.
            fit_record.update(
                loss_train_init, loss_val_init, loss_combined_init,
                self._z["value"], self._phi['w']["value"], self._theta,
                is_init=True
            )
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
            progbar = ProgressBar(
                n_restart, prefix='Progress:', length=50
            )
            progbar.update(0)

        # Run multiple restarts of embedding algorithm.
        for i_restart in range(self.n_restart):
            if (verbose > 2):
                print('        Restart {0}'.format(i_restart))
            if verbose > 0 and verbose < 3:
                progbar.update(i_restart + 1)

            emb.reset()  # TODO
            # TODO inject restart by setting log_dir in RestartBoard
            # emb.log_dir = '{0}/{1}'.format(self.log_dir. i_restart)

            emb.fit(
                obs, max_epoch=max_epoch, patience=patience,
                init_mode=init_mode
            )

            # Update fit record with latest restart.
            fit_record.update(
                loss_train, loss_val, loss_combined, z, attention, theta
            )

        # Sort records from best to worst and grab best.
        fit_record.sort()
        loss_train_best = fit_record.record['loss_train'][0]
        loss_val_best = fit_record.record['loss_val'][0]
        emb._z["value"] = fit_record.record['z'][0]
        emb._phi['w']["value"] = fit_record.record['attention'][0]
        emb._set_theta(fit_record.record['theta'][0])
        emb.fit_duration = time.time() - start_time_s  # TODO
        emb.fit_record = fit_record

        if (verbose > 1):
            if fit_record.beat_init:
                print(
                    '    Best Restart\n        n_epoch: {0} | '
                    'loss: {1: .6f} | loss_val: {2: .6f}'.format(
                        epoch, loss_train_best, loss_val_best
                    )
                )
            else:
                print('    Did not beat initialization.')
        return best_emb

    def from_record(self, fit_record, idx):
        """Set embedding parameters using a record.

        Arguments:
            fit_record: An appropriate psiz.models.FitRecord object.
            idx: An integer indicating which record to use.

        """
        self._z["value"] = fit_record.record['z'][idx]
        self._phi['w']["value"] = fit_record.record['attention'][idx]
        self._set_theta(fit_record.record['theta'][idx])


class FitRecord(object):
    """Class for keeping track of multiple restarts."""

    def __init__(self, n_stimuli, n_dim, n_group, n_record):
        """Initialize.

        Arguments:
            n_restart: TODO
            n_keep: TODO

        """
        self.n_record = n_record
        self.record = {
            'loss_train': np.inf * np.ones([n_record]),
            'loss_val': np.inf * np.ones([n_record]),
            'loss_combined': np.inf * np.ones([n_record]),
            'z': np.zeros([n_record, n_stimuli, n_dim]),
            'attention': np.zeros([n_record, n_group, n_dim]),
            'theta': [None] * n_record
        }
        self.beat_init = None
        self._init_loss_combined = np.inf
        super().__init__()

    def update(
            self, loss_train, loss_val, loss_combined, z, attention, theta,
            is_init=False):
        """Update record with incoming data.

        Arguments:
            loss_train: TODO

        Notes:
            The update method does not worry about keeping the
            records sorted. If the records need to be sorted, use the
            sort method.

        """
        dmy_idx = np.arange(self.n_record)
        locs_is_worse = np.greater(self.record['loss_combined'], loss_combined)

        if np.sum(locs_is_worse) > 0:
            # Identify worst restart in record.
            idx_eligable_as_worst = dmy_idx[locs_is_worse]
            idx_idx_worst = np.argmax(self.record['loss_combined'][locs_is_worse])
            idx_worst = idx_eligable_as_worst[idx_idx_worst]

            # Replace worst restart with incoming restart.
            self.record['loss_train'][idx_worst] = loss_train
            self.record['loss_val'][idx_worst] = loss_val
            self.record['loss_combined'][idx_worst] = loss_combined
            self.record['z'][idx_worst] = z
            self.record['attention'][idx_worst] = attention
            self.record['theta'][idx_worst] = theta

        if is_init:
            self._init_loss_combined = loss_combined
            self.beat_init = False
        else:
            if loss_combined < self._init_loss_combined:
                self.beat_init = True

    def sort(self):
        """Sort the records from best to worst."""
        idx_sort = np.argsort(self.record['loss_combined'])

        self.record['loss_train'] = self.record['loss_train'][idx_sort]
        self.record['loss_val'] = self.record['loss_val'][idx_sort]
        self.record['loss_combined'] = self.record['loss_combined'][idx_sort]
        self.record['z'] = self.record['z'][idx_sort]
        self.record['attention'] = self.record['attention'][idx_sort]
        self.record['theta'] = [self.record['theta'][i] for i in idx_sort]
