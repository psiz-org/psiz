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
"""Test `datasets` module."""

import pytest

import numpy as np

import psiz.datasets
from psiz.trials import RankObservations


@pytest.mark.slow
@pytest.mark.osf
def test_birds_12(tmpdir):
    dataset_name = 'birds-12'
    obs, catalog = psiz.datasets.load(
        dataset_name, cache_subdir='datasets', cache_dir=tmpdir, verbose=1
    )

    assert type(obs) is RankObservations
    assert obs.n_trial == 12253
    assert obs.max_n_reference == 8
    # Spot check obs.
    np.testing.assert_array_equal(
        obs.stimulus_set[10],
        np.array(
            [111, 145, 137, 119, 123, 126, 139, 140, 154], dtype=np.int32
        )
    )
    assert obs.n_reference[10] == 8
    obs.n_select[10] == 2
    assert obs.rt_ms[10] == 21699.0

    assert type(catalog) is psiz.catalog.Catalog
    assert catalog.n_stimuli == 156
    # Spot check catalog.
    row_10 = catalog.stimuli.iloc[10]
    assert row_10.id == 10
    assert row_10.filepath == 'Bird/Icteridae/Bobolink/Bobolink_0040_9681.jpg'
    assert row_10.class_id == 2


@pytest.mark.slow
@pytest.mark.osf
def test_birds_16(tmpdir):
    dataset_name = 'birds-16'
    obs, catalog = psiz.datasets.load(
        dataset_name, cache_subdir='datasets', cache_dir=tmpdir, verbose=1
    )

    assert type(obs) is RankObservations
    assert obs.n_trial == 16292
    assert obs.max_n_reference == 8
    # Spot check obs.
    np.testing.assert_array_equal(
        obs.stimulus_set[10],
        np.array(
            [163, 197, 189, 171, 175, 178, 191, 192, 206], dtype=np.int32
        )
    )
    assert obs.n_reference[10] == 8
    obs.n_select[10] == 2
    assert obs.rt_ms[10] == 21699.0

    assert type(catalog) is psiz.catalog.Catalog
    assert catalog.n_stimuli == 208
    # Spot check catalog.
    row_10 = catalog.stimuli.iloc[10]
    assert row_10.id == 10
    assert (
        row_10.filepath ==
        'Bird/Cardinalidae/Blue_Grosbeak/Blue_Grosbeak_0043_37200.jpg'
    )
    assert row_10.class_id == 2


@pytest.mark.slow
@pytest.mark.osf
def test_skin_lesions(tmpdir):
    dataset_name = 'skin_lesions'
    obs, catalog = psiz.datasets.load(
        dataset_name, cache_subdir='datasets', cache_dir=tmpdir, verbose=1
    )

    assert type(obs) is RankObservations
    assert obs.n_trial == 6726
    assert obs.max_n_reference == 8
    # Spot check obs.
    np.testing.assert_array_equal(
        obs.stimulus_set[10],
        np.array(
            [40, 42, 52, 37, 47, 48, 49, 51, 54], dtype=np.int32
        )
    )
    assert obs.n_reference[10] == 8
    obs.n_select[10] == 2
    assert obs.rt_ms[10] == 4835.0

    assert type(catalog) is psiz.catalog.Catalog
    assert catalog.n_stimuli == 237
    # Spot check catalog.
    row_10 = catalog.stimuli.iloc[10]
    assert row_10.id == 10
    assert row_10.filepath == 'lesion/benign/blue_nevus/blue_nevus_21.jpg'
    assert row_10.class_id == 2


@pytest.mark.slow
@pytest.mark.osf
def test_rocks(tmpdir):
    dataset_name = 'rocks_Nosofsky_etal_2016'
    obs, catalog = psiz.datasets.load(
        dataset_name, cache_subdir='datasets', cache_dir=tmpdir, verbose=1
    )

    assert type(obs) is RankObservations
    assert obs.n_trial == 10798
    assert obs.max_n_reference == 8
    # Spot check obs.
    np.testing.assert_array_equal(
        obs.stimulus_set[10],
        np.array(
            [175, 345, 337,  63, 160, 168, 202, 316, 324], dtype=np.int32
        )
    )
    assert obs.n_reference[10] == 8
    obs.n_select[10] == 2
    assert obs.rt_ms[10] == 9765.0

    assert type(catalog) is psiz.catalog.Catalog
    assert catalog.n_stimuli == 360
    # Spot check catalog.
    row_10 = catalog.stimuli.iloc[10]
    assert row_10.id == 10
    assert row_10.filepath == 'Rock/Igneous/Andesite/I_Andesite_11.png'
    assert row_10.class_id == 2


@pytest.mark.slow
@pytest.mark.osf
def test_ilscrc_val_v1(tmpdir):
    dataset_name = 'ilsvrc_val_v0_1'
    obs, catalog = psiz.datasets.load(
        dataset_name, cache_subdir='datasets', cache_dir=tmpdir, verbose=1
    )

    assert type(obs) is RankObservations
    assert obs.n_trial == 25273
    assert obs.max_n_reference == 8
    # Spot check obs.
    np.testing.assert_array_equal(
        obs.stimulus_set[10],
        np.array(
            [126, 999, 641, 790, 551, 832, 882,  12, 592], dtype=np.int32
        )
    )
    assert obs.n_reference[10] == 8
    obs.n_select[10] == 2
    assert obs.rt_ms[10] == 3333.0

    assert type(catalog) is psiz.catalog.Catalog
    assert catalog.n_stimuli == 50000
    # Spot check catalog.
    row_10 = catalog.stimuli.iloc[10]
    assert row_10.id == 10
    assert row_10.filepath == 'n01530575/ILSVRC2012_val_00010999.JPEG'
    assert row_10.class_id == 10


@pytest.mark.slow
@pytest.mark.osf
def test_ilscrc_val_v2(tmpdir):
    dataset_name = 'ilsvrc_val_v0_2'
    obs, catalog = psiz.datasets.load(
        dataset_name, cache_subdir='datasets', cache_dir=tmpdir, verbose=1
    )

    assert type(obs) is RankObservations
    assert obs.n_trial == 384277
    assert obs.max_n_reference == 8
    # Spot check obs.
    np.testing.assert_array_equal(
        obs.stimulus_set[10],
        np.array(
            [126, 999, 641, 790, 551, 832, 882,  12, 592], dtype=np.int32
        )
    )
    assert obs.n_reference[10] == 8
    obs.n_select[10] == 2
    assert obs.rt_ms[10] == 3333.0

    assert type(catalog) is psiz.catalog.Catalog
    assert catalog.n_stimuli == 50000
    # Spot check catalog.
    row_10 = catalog.stimuli.iloc[10]
    assert row_10.id == 10
    assert row_10.filepath == 'n01530575/ILSVRC2012_val_00010999.JPEG'
    assert row_10.class_id == 10


def test_nonexistent(tmpdir):
    dataset_name = 'nonexistent_dataset'

    with pytest.raises(Exception) as e_info:
        obs, catalog = psiz.datasets.load(
            dataset_name, cache_subdir='datasets', cache_dir=tmpdir
        )
    assert e_info.type == ValueError
