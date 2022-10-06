# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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
"""Test RankSimilarity."""


from psiz.keras.layers import RateSimilarity


def test_serialization(kernel_v0):
    """Test serialization."""
    kernel = kernel_v0
    layer = RateSimilarity(kernel=kernel)
    config = layer.get_config()

    recon_layer = RateSimilarity.from_config(config)
    assert recon_layer.supports_gating is True
