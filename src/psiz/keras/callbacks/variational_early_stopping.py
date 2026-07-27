# -*- coding: utf-8 -*-
# Copyright 2026 The PsiZ Authors. All Rights Reserved.
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


import keras
import numpy as np


class VariationalEarlyStopping(keras.callbacks.Callback):
    """Stop training when both CCE and ELBO have stopped improving."""

    def __init__(
        self,
        monitor_cce="val_cce",
        monitor_elbo="val_loss",
        patience_cce=8,
        patience_elbo=12,
        verbose=0,
        epoch_start=0,
        min_delta=0.0,
        restore_best_weights=True,
    ):
        """Initialize early stopping callback.

        Args:
            monitor_cce: Name of the metric to monitor for CCE.
            monitor_elbo: Name of the metric to monitor for ELBO.
            patience_cce: Number of epochs with no improvement after which
                training will be stopped for CCE.
            patience_elbo: Number of epochs with no improvement after which
                training will be stopped for ELBO.
            verbose: Verbosity level.
            epoch_start: Monitoring only starts after this epoch.
            min_delta: Minimum change in the monitored quantity to qualify as
                an improvement.
            restore_best_weights: Whether to restore model weights from the
                epoch with the best value of the monitored quantity.

        """
        super().__init__()
        self.monitor_cce = monitor_cce
        self.monitor_elbo = monitor_elbo
        self.patience_cce = patience_cce
        self.patience_elbo = patience_elbo
        self.verbose = verbose
        self.epoch_start = epoch_start
        self.min_delta = min_delta

        self.best_cce = np.inf
        self.best_elbo = np.inf
        self.wait_cce = 0
        self.wait_elbo = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait_cce = 0
        self.wait_elbo = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_cce = logs.get(self.monitor_cce)
        current_elbo = logs.get(self.monitor_elbo)

        if epoch >= self.epoch_start:
            # --- Check validation CCE ---
            if current_cce is None:
                print(f"Warning: {self.monitor_cce} not found in logs")
                return

            if current_cce < self.best_cce - self.min_delta:
                self.best_cce = current_cce
                self.wait_cce = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
                self.best_epoch = epoch
            else:
                self.wait_cce += 1

            # --- Check validation ELBO ---
            if current_elbo is not None:
                if current_elbo < self.best_elbo - self.min_delta:
                    self.best_elbo = current_elbo
                    self.wait_elbo = 0
                else:
                    self.wait_elbo += 1

        # --- Verbose output ---
        if self.verbose > 0:
            msg = f"\nEpoch {epoch+1}: CCE wait={self.wait_cce}/{self.patience_cce}"
            if epoch >= self.epoch_start and current_elbo is not None:
                msg += f", ELBO wait={self.wait_elbo}/{self.patience_elbo}"
            print(msg)

        # --- Early stopping condition ---
        stop = False
        if epoch >= self.epoch_start:
            if (self.wait_cce >= self.patience_cce) and (
                self.wait_elbo >= self.patience_elbo
            ):
                stop = True

        if stop:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                print(
                    "Restoring model weights from the end of best epoch (by CCE): "
                    f"{self.best_epoch + 1}."
                )
            self.model.set_weights(self.best_weights)
