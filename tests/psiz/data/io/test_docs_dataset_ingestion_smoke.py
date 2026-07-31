# -*- coding: utf-8 -*-
"""Smoke test for dataset artifact docs workflow."""

from __future__ import annotations

import numpy as np

import psiz


def test_docs_example_dataset_ingestion_smoke(tmp_path):
    stimulus_set = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    content = psiz.data.Rank(stimulus_set, n_select=1, name="stimulus_set")

    outcome_idx = np.array([0, 1], dtype=np.int32)
    sample_weight = np.array([[1.0], [0.5]], dtype=np.float32)
    outcome = psiz.data.SparseCategorical(
        outcome_idx,
        depth=2,
        sample_weight=sample_weight,
        name="outcome",
    )

    dataset = psiz.data.Dataset([content, outcome])

    artifact_dir = tmp_path / "docs_dataset.psiz"
    _ = dataset.save(
        artifact_dir,
        dataset_id="docs_example",
        split_set_id="split_set_v1",
    )

    pyds = psiz.data.load(
        artifact_dir,
        split_set_id="split_set_v1",
        split_labels=["train"],
    )

    batch = pyds[0]
    x, y, w = batch
    x_key = next(iter(x.keys()))
    assert x[x_key].shape == (2, 3)
    assert y.shape == (2, 2)
    assert w.shape[0] == 2

    tf = __import__("pytest").importorskip("tensorflow")
    tf_dataset = pyds.tensorflow()
    first = next(iter(tf_dataset))
    tf_x_key = next(iter(first[0].keys()))
    assert isinstance(first[0][tf_x_key], tf.Tensor)
