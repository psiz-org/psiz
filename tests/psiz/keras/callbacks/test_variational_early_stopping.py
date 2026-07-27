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
"""Tests for variational early stopping callback."""

import numpy as np

from psiz.keras.callbacks import VariationalEarlyStopping


class DummyModel:
    def __init__(self, weights):
        self.weights = [np.array(weight, dtype="float32") for weight in weights]
        self.stop_training = False

    def get_weights(self):
        return [weight.copy() for weight in self.weights]

    def set_weights(self, weights):
        self.weights = [weight.copy() for weight in weights]


def test_stops_when_cce_and_elbo_patience_are_reached_and_restores_best_weights():
    model = DummyModel(weights=[[1.0], [2.0]])
    callback = VariationalEarlyStopping(
        patience_cce=1,
        patience_elbo=1,
        restore_best_weights=True,
    )
    callback.set_model(model)

    callback.on_train_begin()

    callback.on_epoch_end(0, logs={"val_cce": 0.5, "val_loss": 0.9})
    model.weights = [np.array([3.0], dtype="float32"), np.array([4.0], dtype="float32")]

    callback.on_epoch_end(1, logs={"val_cce": 0.6, "val_loss": 1.0})
    callback.on_train_end()

    assert model.stop_training is True
    assert callback.stopped_epoch == 1
    assert callback.best_epoch == 0
    assert np.array_equal(model.weights[0], np.array([1.0], dtype="float32"))
    assert np.array_equal(model.weights[1], np.array([2.0], dtype="float32"))


def test_warns_and_skips_when_cce_metric_is_missing(capsys):
    model = DummyModel(weights=[[1.0], [2.0]])
    callback = VariationalEarlyStopping(patience_cce=1, patience_elbo=1)
    callback.set_model(model)

    callback.on_train_begin()
    callback.on_epoch_end(0, logs={"val_loss": 0.5})

    captured = capsys.readouterr()
    assert "Warning: val_cce not found in logs" in captured.out
    assert model.stop_training is False
