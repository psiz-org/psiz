# -*- coding: utf-8 -*-
"""Arrow adapter helpers for PsiZ runtime datasets."""


def arrow(dataset):
    """Convert a PsiZ dataset-like object to pyarrow.Table."""
    if not hasattr(dataset, "arrow"):
        raise TypeError("Expected dataset with `arrow` method.")
    return dataset.arrow()
