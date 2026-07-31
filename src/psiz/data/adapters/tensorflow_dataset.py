# -*- coding: utf-8 -*-
"""TensorFlow adapter helpers for PsiZ runtime datasets."""

def tensorflow(dataset):
    """Convert a PsiZ dataset-like object to tf.data.Dataset."""
    if not hasattr(dataset, "tensorflow"):
        raise TypeError("Expected dataset with `tensorflow` method.")
    return dataset.tensorflow()
