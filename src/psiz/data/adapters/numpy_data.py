# -*- coding: utf-8 -*-
"""NumPy adapter helpers for PsiZ runtime datasets."""


def numpy(dataset):
    """Convert a PsiZ dataset-like object to NumPy payloads."""
    if not hasattr(dataset, "numpy"):
        raise TypeError("Expected dataset with `numpy` method.")
    return dataset.numpy()
