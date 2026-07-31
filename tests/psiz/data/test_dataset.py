# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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
"""Test data module."""


import keras
import numpy as np
import pytest
import psiz


class _TensorFlowProxy:
    """Lazy TensorFlow import that skips tests when TF is unavailable."""

    _module = None

    def __getattr__(self, name):
        if self._module is None:
            self._module = pytest.importorskip("tensorflow")
        return getattr(self._module, name)


tf = _TensorFlowProxy()

from psiz.data.groups.group import Group
from psiz.data.outcomes.sparse_categorical import SparseCategorical
from psiz.data.dataset import Dataset
from psiz.data.dataset_component import DatasetComponent


class BadTrialComponent(DatasetComponent):
    """Abstract class for trial content data."""

    def __init__(self):
        """Initialize."""
        DatasetComponent.__init__(self)
        self.x = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        )
        self.n_sample = 4
        self.sequence_length = 1

    def numpy(self, with_timestep_axis=None):
        del with_timestep_axis
        return self.x


def test_init_0a(c_2rank1_a_4x1):
    """Test initialization.

    Bare minimum arguments.

    """
    pds = Dataset([c_2rank1_a_4x1])

    assert pds.n_sample == c_2rank1_a_4x1.n_sample
    assert pds.sequence_length == c_2rank1_a_4x1.sequence_length
    assert len(pds.content_list) == 1
    assert len(pds.group_list) == 0
    assert len(pds.outcome_list) == 0


def test_init_0b(c_2rank1_aa_4x1):
    """Test initialization.

    Bare minimum arguments.

    """
    pds = Dataset([c_2rank1_aa_4x1])

    assert pds.n_sample == c_2rank1_aa_4x1.n_sample
    assert pds.sequence_length == c_2rank1_aa_4x1.sequence_length
    assert len(pds.content_list) == 1
    assert len(pds.group_list) == 0
    assert len(pds.outcome_list) == 0


def test_init_1(c_2rank1_aa_4x1):
    """Test initialization.

    With outcome, no sample weights.

    """
    # TODO move rank_outcome to conftest to promote readability
    outcome_idx = np.zeros(
        [c_2rank1_aa_4x1.n_sample, c_2rank1_aa_4x1.sequence_length], dtype=np.int32
    )
    rank_outcome = SparseCategorical(
        outcome_idx, depth=c_2rank1_aa_4x1.n_outcome, name="rank_outcome"
    )

    pds = Dataset([c_2rank1_aa_4x1, rank_outcome])

    assert pds.n_sample == c_2rank1_aa_4x1.n_sample
    assert pds.sequence_length == c_2rank1_aa_4x1.sequence_length
    assert len(pds.content_list) == 1
    assert len(pds.group_list) == 0
    assert len(pds.outcome_list) == 1


def test_init_2(c_2rank1_aa_4x1, o_2rank1_aa_4x1):
    """Test initialization.

    With outcome, including sample_weight.
    With group, mixture format.

    """
    value = np.array(
        [
            [[0.1, 0.9]],
            [[0.5, 0.5]],
            [[1.0, 0.0]],
            [[0.9, 0.1]],
        ]
    )
    group_0 = Group(value, name="group_id")

    pds = Dataset([c_2rank1_aa_4x1, group_0, o_2rank1_aa_4x1])

    assert pds.n_sample == c_2rank1_aa_4x1.n_sample
    assert pds.sequence_length == c_2rank1_aa_4x1.sequence_length
    assert len(pds.content_list) == 1
    assert len(pds.group_list) == 1
    assert len(pds.outcome_list) == 1


def test_init_3(c_2rank1_aa_4x1):
    """Test initialization.

    With outcome, including sample_weight argument.
    With group, pass in sparse format.

    """
    # Create rank outcome.
    outcome_idx = np.zeros(
        [c_2rank1_aa_4x1.n_sample, c_2rank1_aa_4x1.sequence_length], dtype=np.int32
    )
    sample_weight = 0.9 * np.ones(
        [c_2rank1_aa_4x1.n_sample, c_2rank1_aa_4x1.sequence_length]
    )
    rank_outcome = SparseCategorical(
        outcome_idx,
        depth=c_2rank1_aa_4x1.n_outcome,
        sample_weight=sample_weight,
        name="rank_outcome",
    )

    value = np.array(
        [
            [[0]],
            [[1]],
            [[0]],
            [[0]],
        ]
    )
    group_0 = Group(value, name="condition_idx")

    pds = Dataset([c_2rank1_aa_4x1, group_0, rank_outcome])

    assert pds.n_sample == c_2rank1_aa_4x1.n_sample
    assert pds.sequence_length == c_2rank1_aa_4x1.sequence_length
    assert len(pds.content_list) == 1
    assert len(pds.group_list) == 1
    assert len(pds.outcome_list) == 1


def test_init_4(c_2rank1_d_3x2, o_2rank1_d_3x2, o_rt_a_3x2):
    """Test initialization.

    One content, two outcomes.

    """
    pds = Dataset([c_2rank1_d_3x2, o_2rank1_d_3x2, o_rt_a_3x2])

    assert pds.n_sample == 3
    assert pds.sequence_length == 2
    assert len(pds.content_list) == 1
    assert len(pds.group_list) == 0
    assert len(pds.outcome_list) == 2


def test_init_5(c_2rank1_d_3x2, o_2rank1_d_3x2, c_rate2_e_3x2, o_rate2_a_3x2):
    """Test initialization.

    * two contents
    * two outcomes

    """
    pds = Dataset([c_2rank1_d_3x2, o_2rank1_d_3x2, c_rate2_e_3x2, o_rate2_a_3x2])

    assert pds.n_sample == 3
    assert pds.sequence_length == 2
    assert len(pds.content_list) == 2
    assert len(pds.group_list) == 0
    assert len(pds.outcome_list) == 2


def test_invalid_init_0(c_2rank1_aa_4x1, o_2rank1_d_3x2, o_4rank2_c_4x3):
    """Test invalid initialization.

    * Number of sequences disagrees.
    * Sequence length disagrees.

    """
    with pytest.raises(Exception) as e_info:
        Dataset([c_2rank1_aa_4x1, o_2rank1_d_3x2])
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "All user-provided 'DatasetComponent' objects must have the same "
        "`n_sample`. The 'DatasetComponent' in position 1 does not match "
        "the previous components."
    )

    with pytest.raises(Exception) as e_info:
        Dataset([c_2rank1_aa_4x1, o_4rank2_c_4x3])
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "All user-provided 'DatasetComponent' objects must have the same "
        "`sequence_length`. The 'DatasetComponent' in position 1 does not "
        "match the previous components."
    )

    bad_component_4x1 = BadTrialComponent()
    with pytest.raises(Exception) as e_info:
        Dataset([c_2rank1_aa_4x1, bad_component_4x1])
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The `DatasetComponent` in position 1 must be an  instance of "
        "`psiz.data.Content`, `psiz.data.Outcome`, or `psiz.data.Group` to "
        "use `Dataset`."
    )


@pytest.mark.backend_tensorflow
def test_export_0(c_2rank1_d_3x2, g_condition_idx_3x2):
    """Test export.

    * Include content and group only.

    """
    pds = Dataset([c_2rank1_d_3x2, g_condition_idx_3x2])

    desired_x_stimulus_set = tf.constant(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [0, 0, 0],
            ],
            [
                [10, 11, 12],
                [14, 15, 16],
            ],
        ],
        dtype=tf.int32,
    )
    desired_condition_id = tf.constant(
        [[[0], [0]], [[1], [1]], [[0], [0]]], dtype=tf.int32
    )

    tfds = pds.export().batch(4, drop_remainder=False)
    ds_list = list(tfds)
    x = ds_list[0]

    assert len(ds_list) == 1
    tf.debugging.assert_equal(desired_x_stimulus_set, x["given2rank1_stimulus_set"])
    tf.debugging.assert_equal(desired_condition_id, x["condition_idx"])


@pytest.mark.backend_tensorflow
def test_export_1(c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2):
    """Test export.

    Return dataset using override `with_timestep_axis=False`.

    """
    pds = Dataset([c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2])

    desired_x_stimulus_set = tf.constant(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [0, 0, 0],
            [10, 11, 12],
            [14, 15, 16],
        ],
        dtype=tf.int32,
    )
    desired_condition_id = tf.constant([[0], [0], [1], [1], [0], [0]], dtype=tf.int32)
    desired_y = tf.constant(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=tf.float32,
    )
    desired_w = tf.constant([0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=tf.float32)

    tfds = pds.export(with_timestep_axis=False).batch(6, drop_remainder=False)
    ds_list = list(tfds)
    x = ds_list[0][0]
    y = ds_list[0][1]
    w = ds_list[0][2]

    assert len(ds_list[0]) == 3
    tf.debugging.assert_equal(desired_x_stimulus_set, x["given2rank1_stimulus_set"])
    tf.debugging.assert_equal(desired_condition_id, x["condition_idx"])
    tf.debugging.assert_equal(desired_y, y)
    tf.debugging.assert_equal(desired_w, w)


@pytest.mark.backend_tensorflow
def test_export_2a(c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2, o_rt_a_3x2):
    """Test export.

    * Multi-output model, therefore keep dictionary keys for `y` and
    `w`.

    """
    pds = Dataset([c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2, o_rt_a_3x2])

    desired_x_stimulus_set = tf.constant(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [0, 0, 0],
            ],
            [
                [10, 11, 12],
                [14, 15, 16],
            ],
        ],
        dtype=tf.int32,
    )
    desired_condition_id = tf.constant(
        [[[0], [0]], [[1], [1]], [[0], [0]]], dtype=tf.int32
    )
    desired_y_prob = tf.constant(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
        ],
        dtype=tf.float32,
    )
    desired_w_prob = tf.constant(
        [
            [0.9, 0.9],
            [0.9, 0.9],
            [0.9, 0.9],
        ],
        dtype=tf.float32,
    )
    desired_y_rt = tf.constant(
        [
            [[4.1], [4.2]],
            [[5.1], [5.2]],
            [[6.1], [6.2]],
        ],
        dtype=tf.float32,
    )
    desired_w_rt = tf.constant(
        [
            [0.8, 0.8],
            [0.8, 0.8],
            [0.8, 0.8],
        ],
        dtype=tf.float32,
    )

    tfds = pds.export().batch(4, drop_remainder=False)
    ds_list = list(tfds)
    x = ds_list[0][0]
    y = ds_list[0][1]
    w = ds_list[0][2]

    assert len(ds_list[0]) == 3
    tf.debugging.assert_equal(desired_x_stimulus_set, x["given2rank1_stimulus_set"])
    tf.debugging.assert_equal(desired_condition_id, x["condition_idx"])
    tf.debugging.assert_equal(desired_y_prob, y["rank_prob"])
    tf.debugging.assert_equal(desired_w_prob, w["rank_prob"])
    tf.debugging.assert_equal(desired_y_rt, y["rt"])
    tf.debugging.assert_equal(desired_w_rt, w["rt"])


@pytest.mark.backend_tensorflow
def test_export_3(c_rate2_a_4x1, g_condition_label_4x1, o_continuous_a_4x1):
    """Test export with `StringLookup`."""
    pds = Dataset([c_rate2_a_4x1, g_condition_label_4x1, o_continuous_a_4x1])
    tfds = pds.export(export_format="tfds")

    # Map strings to indices.
    condition_lookup_layer = keras.layers.StringLookup(
        vocabulary=["block", "interleave"], num_oov_indices=0
    )

    def parse_inputs(x):
        condition_label = x.pop("condition_label")
        condition_idx = condition_lookup_layer(condition_label)
        x["condition_idx"] = condition_idx
        return x

    ds2 = tfds.map(lambda x, y, w: (parse_inputs(x), y, w))
    ds2 = ds2.batch(4)
    ds2_list = list(ds2)

    desired_condition_idx = tf.constant(
        [
            [[0]],
            [[1]],
            [[0]],
            [[0]],
        ],
        dtype=tf.int64,
    )
    tf.debugging.assert_equal(ds2_list[0][0]["condition_idx"], desired_condition_idx)


@pytest.mark.backend_tensorflow
def test_export_4(c_2rank1_a_4x1):
    """Test export."""
    pds = Dataset([c_2rank1_a_4x1])

    desired_x_stimulus_set = tf.constant(
        [[3, 1, 2], [9, 12, 7], [5, 6, 7], [13, 14, 15]], dtype=np.int32
    )

    tfds = pds.export().batch(4, drop_remainder=False)
    ds_list = list(tfds)
    x = ds_list[0]

    assert len(ds_list) == 1
    tf.debugging.assert_equal(desired_x_stimulus_set, x["given2rank1_stimulus_set"])


@pytest.mark.backend_tensorflow
def test_export_5(c_2rank1_aa_4x1):
    """Test export."""
    pds = Dataset([c_2rank1_aa_4x1])

    desired_x_stimulus_set = tf.constant(
        [[[3, 1, 2]], [[9, 12, 7]], [[5, 6, 7]], [[13, 14, 15]]], dtype=np.int32
    )

    tfds = pds.export().batch(4, drop_remainder=False)
    ds_list = list(tfds)
    x = ds_list[0]

    assert len(ds_list) == 1
    tf.debugging.assert_equal(desired_x_stimulus_set, x["given2rank1_stimulus_set"])


@pytest.mark.backend_tensorflow
def test_invalid_export_0(c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2):
    """Test export.

    Using incorrect `export_format`.

    """
    pds = Dataset([c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2])

    with pytest.raises(Exception) as e_info:
        pds.export(export_format="garbage")
    assert e_info.type == ValueError
    assert str(e_info.value) == "Unrecognized `export_format` 'garbage'."


@pytest.mark.backend_tensorflow
def test_invalid_export_1(c_2rank1_d_3x2, o_2rank1_d_3x2, o_rt_a_3x2_noname):
    """Test export.

    Using incorrect `export_format`.

    """
    pds = Dataset([c_2rank1_d_3x2, o_2rank1_d_3x2, o_rt_a_3x2_noname])

    with pytest.raises(Exception) as e_info:
        pds.export(export_format="tfds")
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "When a `Dataset` has multiple outputs, all "
        "outputs must be created with the `name` argument."
    )


@pytest.mark.backend_tensorflow
def test_tf_ds_concatenate(c_2rank1_d_3x2, c_2rank1_e_3x2):
    """Test concatenating two datasets"""
    td_0 = Dataset([c_2rank1_d_3x2])
    td_1 = Dataset([c_2rank1_e_3x2])

    ds_0 = td_0.export(export_format="tfds")
    ds_1 = td_1.export(export_format="tfds")

    tfds = ds_0.concatenate(ds_1).batch(6)
    ds_list = list(tfds)
    _ = ds_list[0]


@pytest.mark.backend_tensorflow
def test_export_tfds_deprecation_warning(c_2rank1_a_4x1):
    """Legacy tfds export should emit a deprecation warning."""
    pds = Dataset([c_2rank1_a_4x1])
    with pytest.warns(DeprecationWarning, match="Dataset.export"):
        _ = pds.export(export_format="tfds")


def test_tier_a_numpy_content_only(c_2rank1_a_4x1):
    """Dataset.numpy should materialize content-only arrays."""
    pds = Dataset([c_2rank1_a_4x1])
    x = pds.numpy()
    assert x["given2rank1_stimulus_set"].shape == (4, 3)


def test_tier_a_tensorflow_with_outcome(c_2rank1_aa_4x1, o_2rank1_aa_4x1):
    """Dataset.tensorflow should produce unbatched (x, y, w) tuples."""
    pds = Dataset([c_2rank1_aa_4x1, o_2rank1_aa_4x1])
    tfds = pds.tensorflow()
    first = next(iter(tfds))
    assert len(first) == 3


def _assert_mapping_allclose(actual, expected):
    assert set(actual.keys()) == set(expected.keys())
    for key in actual.keys():
        actual_value = _normalize_text_array(actual[key])
        expected_value = _normalize_text_array(expected[key])
        if np.issubdtype(actual_value.dtype, np.number) and np.issubdtype(
            expected_value.dtype, np.number
        ):
            np.testing.assert_allclose(actual_value, expected_value)
        else:
            np.testing.assert_array_equal(actual_value, expected_value)


def _normalize_text_array(value):
    """Normalize bytes payloads to unicode for deterministic comparisons."""
    arr = np.asarray(value)
    if arr.dtype.kind == "S":
        return arr.astype("U")
    if arr.dtype == object:
        decode = np.vectorize(
            lambda x: x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x,
            otypes=[object],
        )
        return decode(arr)
    return arr


def _assert_roundtrip_matches_materialized(
    pds,
    artifact_dir,
    *,
    with_timestep_axis=None,
):
    expected = pds.numpy(with_timestep_axis=with_timestep_axis)

    pds.save(
        artifact_dir,
        dataset_id="dataset_roundtrip_test",
        with_timestep_axis=with_timestep_axis,
    )
    actual = psiz.data.load(artifact_dir).numpy()

    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        _assert_mapping_allclose(actual, expected)
        return

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    assert len(actual) == len(expected)

    _assert_mapping_allclose(actual[0], expected[0])
    if isinstance(expected[1], dict):
        _assert_mapping_allclose(actual[1], expected[1])
    else:
        np.testing.assert_allclose(actual[1], expected[1])

    if isinstance(expected[2], dict):
        _assert_mapping_allclose(actual[2], expected[2])
    else:
        np.testing.assert_allclose(actual[2], expected[2])


def test_roundtrip_save_load_0(c_2rank1_d_3x2, g_condition_idx_3x2, tmp_path):
    """Round-trip save/load for export_0-style dataset."""
    pds = Dataset([c_2rank1_d_3x2, g_condition_idx_3x2])
    artifact_dir = tmp_path / "dataset_export_0.psiz"
    _assert_roundtrip_matches_materialized(pds, artifact_dir)


def test_roundtrip_save_load_1(
    c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2, tmp_path
):
    """Round-trip save/load for export_1-style dataset with flattened timesteps."""
    pds = Dataset([c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2])
    artifact_dir = tmp_path / "dataset_export_1.psiz"
    _assert_roundtrip_matches_materialized(pds, artifact_dir, with_timestep_axis=False)


def test_roundtrip_save_load_2a(
    c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2, o_rt_a_3x2, tmp_path
):
    """Round-trip save/load for export_2a-style multi-output dataset."""
    pds = Dataset([c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2, o_rt_a_3x2])
    artifact_dir = tmp_path / "dataset_export_2a.psiz"
    _assert_roundtrip_matches_materialized(pds, artifact_dir)


def test_roundtrip_save_load_3(
    c_rate2_a_4x1, g_condition_label_4x1, o_continuous_a_4x1, tmp_path
):
    """Round-trip save/load for export_3-style dataset."""
    pds = Dataset([c_rate2_a_4x1, g_condition_label_4x1, o_continuous_a_4x1])
    artifact_dir = tmp_path / "dataset_export_3.psiz"
    _assert_roundtrip_matches_materialized(pds, artifact_dir)


def test_roundtrip_save_load_4(c_2rank1_a_4x1, tmp_path):
    """Round-trip save/load for export_4-style content-only dataset."""
    pds = Dataset([c_2rank1_a_4x1])
    artifact_dir = tmp_path / "dataset_export_4.psiz"
    _assert_roundtrip_matches_materialized(pds, artifact_dir)


def test_roundtrip_save_load_5(c_2rank1_aa_4x1, tmp_path):
    """Round-trip save/load for export_5-style content-only dataset."""
    pds = Dataset([c_2rank1_aa_4x1])
    artifact_dir = tmp_path / "dataset_export_5.psiz"
    _assert_roundtrip_matches_materialized(pds, artifact_dir)
