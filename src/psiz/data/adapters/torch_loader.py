# -*- coding: utf-8 -*-
"""PyTorch adapter helpers for PsiZ runtime datasets."""

def torch(dataset):
    """Convert a PsiZ dataset-like object to torch.utils.data.Dataset."""
    if not hasattr(dataset, "torch"):
        raise TypeError("Expected dataset with `torch` method.")
    return dataset.torch()
