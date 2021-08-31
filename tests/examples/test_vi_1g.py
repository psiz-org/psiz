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
"""Module for testing models.py."""

import pytest

import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf

import psiz

# TODO
# model_inferred_best = {}
# val_loss_best = np.inf
# for i_restart in range(n_restart):
#     model_inferred = build_model(n_stimuli, n_dim)
#     model_inferred.compile(**compile_kwargs)

#     model_inferred.fit(
#         ds_obs_train, validation_data=ds_obs_val, epochs=epochs,
#         callbacks=callbacks, verbose=0
#     )

#     # d_train = model_inferred.evaluate(ds_obs_train, return_dict=True)
#     d_val = model_inferred.evaluate(ds_obs_val, return_dict=True)
#     # d_test = model_inferred.evaluate(ds_obs_test, return_dict=True)

#     if d_val['loss'] < val_loss_best:
#         val_loss_best = d_val['loss']
#         model_inferred_best = model_inferred
