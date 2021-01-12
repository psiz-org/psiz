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
"""Module of TensorFlow embedding layers.

Classes:
    EmbeddingNormalDiag: A normal distribution embedding layer.
    EmbeddingLaplaceDiag: A Laplace distribution embedding layer.
    EmbeddingLogNormalDiag: A log-normal distribution embedding layer.
    EmbeddingLogitNormalDiag: A logit-normal distribution embedding
        layer.
    EmbeddingTruncatedNormalDiag: A truncated normal distribution
        embedding layer.
    EmbeddingGammaDiag: A Gamma distribution embedding layer.
    EmbeddingVariational: A variational embedding layer.
    EmbeddingShared: An embedding layer that shares weights across
        stimuli and dimensions.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
import tensorflow_probability as tfp

from psiz.keras.layers.variational import Variational
import psiz.keras.constraints
