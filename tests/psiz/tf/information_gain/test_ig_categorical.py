# -*- coding: utf-8 -*-
# Copyright 2021 The PsiZ Authors. All Rights Reserved.
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
"""Test trials module."""

import tensorflow as tf

from psiz.tf.information_theory import ig_categorical


def test_0():
    """Test 0."""
    y_pred = tf.constant(
        [
            [
                [0.6323555, 0.29509985, 0.07254463],
                [0.6445243, 0.26066366, 0.09481203],
                [0.64741296, 0.24667357, 0.10591353],
                [0.7007505, 0.22065769, 0.07859177],
                [0.6902091, 0.23418467, 0.07560635],
                [0.6938999, 0.20956476, 0.0965353],
                [0.6230698, 0.26290143, 0.11402874],
                [0.67936206, 0.23431861, 0.08631929],
                [0.6828533, 0.2176706, 0.0994761],
                [0.6518116, 0.26351637, 0.08467206]
            ],
            [
                [0.43983725, 0.12965858, 0.43050423],
                [0.49327767, 0.16250171, 0.34422064],
                [0.47315526, 0.15181471, 0.37503004],
                [0.5293639, 0.15241306, 0.31822306],
                [0.55987227, 0.12040693, 0.3197208],
                [0.47270614, 0.18659984, 0.340694],
                [0.36352947, 0.11434101, 0.52212954],
                [0.42090833, 0.2071818, 0.37190983],
                [0.42788872, 0.19405055, 0.37806076],
                [0.4315316, 0.14196, 0.42650843]
            ],
            [
                [0.4994854, 0.1790246, 0.32149008],
                [0.47609267, 0.1828847, 0.34102267],
                [0.5113885, 0.15853655, 0.33007497],
                [0.4922062, 0.17076965, 0.33702415],
                [0.50326526, 0.16307637, 0.33365834],
                [0.4586086, 0.18553732, 0.3558541],
                [0.45560402, 0.19848944, 0.3459066],
                [0.54689276, 0.17514864, 0.27795857],
                [0.4682883, 0.17019485, 0.36151686],
                [0.47671524, 0.1877261, 0.3355587]
            ]
        ], dtype=tf.float32
    )
    ig = ig_categorical(y_pred)
    ig_desired = tf.constant(
        [0.00270152, 0.01042497, 0.00180888], dtype=tf.float32
    )
    tf.debugging.assert_near(ig, ig_desired)
