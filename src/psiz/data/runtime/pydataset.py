# -*- coding: utf-8 -*-
"""Backend-neutral runtime dataset for PsiZ artifacts."""

from __future__ import annotations

from typing import Any

import keras
import numpy as np

from psiz.data.io import decode_observations_to_xyw
from psiz.data.io import read_dataset_artifact


class PsizPyDataset(keras.utils.PyDataset):
    """Backend-neutral Keras PyDataset wrapper around x/y/w arrays."""

    def __init__(
        self,
        x: dict[str, np.ndarray],
        y: dict[str, np.ndarray] | None = None,
        w: dict[str, np.ndarray] | None = None,
        *,
        batch_size: int | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        self.x = x
        self.y = y or {}
        self.w = w or {}

        n_sample = _infer_n_sample(self.x, self.y, self.w)
        self._n_sample = n_sample

        if batch_size is None:
            batch_size = n_sample
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = seed

        self._indices = np.arange(n_sample)
        super().__init__()

    def __len__(self) -> int:
        return int(np.ceil(self._n_sample / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self._indices)

    def __getitem__(self, index: int) -> Any:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self._n_sample)
        idx = self._indices[start:end]

        xb = {k: v[idx] for k, v in self.x.items()}
        if not self.y:
            return xb

        yb = {k: v[idx] for k, v in self.y.items()}
        wb = {k: v[idx] for k, v in self.w.items()}

        if len(yb) == 1:
            yb = next(iter(yb.values()))
        if len(wb) == 1:
            wb = next(iter(wb.values()))

        return xb, yb, wb

    def numpy(self):
        """Return minimally processed NumPy x/y/w payloads."""
        x = {k: np.asarray(v) for k, v in self.x.items()}
        if not self.y:
            return x

        y = {k: np.asarray(v) for k, v in self.y.items()}
        w = {k: np.asarray(v) for k, v in self.w.items()}
        return x, _collapse_singleton_block(y), _collapse_singleton_block(w)

    def tensorflow(self):
        """Return minimally processed, unbatched tf.data.Dataset rows."""
        import tensorflow as tf

        return tf.data.Dataset.from_tensor_slices(self.numpy())

    def torch(self):
        """Return minimally processed torch.utils.data.Dataset rows."""
        import torch
        from torch.utils.data import Dataset as TorchDataset

        parent = self

        class _TorchDataset(TorchDataset):
            def __len__(self):
                return parent._n_sample

            def __getitem__(self, idx):
                x = {k: torch.as_tensor(v[idx]) for k, v in parent.x.items()}
                if not parent.y:
                    return x

                y = {k: torch.as_tensor(v[idx]) for k, v in parent.y.items()}
                w = {k: torch.as_tensor(v[idx]) for k, v in parent.w.items()}
                if len(y) == 1:
                    y = next(iter(y.values()))
                if len(w) == 1:
                    w = next(iter(w.values()))
                return x, y, w

        return _TorchDataset()

    def arrow(self):
        """Return minimally processed rows as a pyarrow.Table."""
        import pyarrow as pa

        columns = {}

        def _add_columns(prefix, block):
            for name, value in block.items():
                arr = np.asarray(value)
                col_name = f"{prefix}{name}"
                if arr.ndim <= 1:
                    columns[col_name] = pa.array(arr)
                else:
                    columns[col_name] = pa.array([np.asarray(v).tolist() for v in arr])

        _add_columns("x::", self.x)
        _add_columns("y::", self.y)
        _add_columns("w::", self.w)
        return pa.table(columns)



def load_dataset(
    dataset_root,
    *,
    split_set_id: str | None = None,
    split_labels: list[str] | None = None,
):
    """Load a PsiZ dataset artifact into a backend-neutral PyDataset."""
    payload = read_dataset_artifact(
        dataset_root,
        split_set_id=split_set_id,
        split_labels=split_labels,
    )
    runtime = payload["manifest"]["runtime_contract"]
    x, y, w = decode_observations_to_xyw(payload["observations"], runtime)
    return PsizPyDataset(x, y, w)


def _infer_n_sample(*blocks):
    for block in blocks:
        if not block:
            continue
        key = next(iter(block.keys()))
        value = np.asarray(block[key])
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    return 0


def _collapse_singleton_block(block):
    if len(block) == 1:
        return next(iter(block.values()))
    return block
