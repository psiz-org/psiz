# -*- coding: utf-8 -*-
# Copyright 2026 The PsiZ Authors. All Rights Reserved.
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
"""Utilities for compacting and restoring model_config payloads."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

EXTERNAL_BLOB_MARKER_KEY = "__psiz_external_blob__"
EXTERNAL_BLOB_MARKER_VERSION = 1
DEFAULT_BLOB_FILE = "model_config_blobs.safetensors"
DEFAULT_MIN_EXTERNALIZED_BYTES = 64 * 1024
_FORCED_EXTERNALIZED_LIST_KEYS = {"membership_current", "membership_parent"}


def compact_model_config(
    model_config: dict[str, Any],
    *,
    min_externalized_bytes: int = DEFAULT_MIN_EXTERNALIZED_BYTES,
    blob_file: str = DEFAULT_BLOB_FILE,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any] | None]:
    """Externalize large numeric list payloads from model_config.

    Returns:
        compact_model_config: Model config with marker objects in place of
            externalized numeric payloads.
        blob_tensors: Mapping of blob key to ndarray for safetensors storage.
        manifest: None if no payloads were externalized, otherwise metadata for
            config.json.
    """
    if min_externalized_bytes <= 0:
        raise ValueError("min_externalized_bytes must be > 0.")

    blob_tensors: dict[str, np.ndarray] = {}
    total_json_externalized_bytes = 0
    counter = 0

    def _walk(
        node: Any,
        parent_key: str | None = None,
        parent_dict: dict[str, Any] | None = None,
    ) -> Any:
        nonlocal counter, total_json_externalized_bytes

        if isinstance(node, dict):
            return {k: _walk(v, parent_key=k, parent_dict=node) for k, v in node.items()}

        if isinstance(node, list):
            marker = _externalize_list_if_needed(
                node,
                parent_key=parent_key,
                parent_dict=parent_dict,
                min_externalized_bytes=min_externalized_bytes,
                counter=counter,
            )
            if marker is not None:
                counter += 1
                total_json_externalized_bytes += int(marker.pop("_json_bytes"))
                marker_metadata = marker[EXTERNAL_BLOB_MARKER_KEY]
                blob_tensors[marker_metadata["key"]] = marker.pop("_tensor")
                return marker

            return [_walk(v, parent_key=None, parent_dict=None) for v in node]

        return node

    compact_config = _walk(model_config)
    if not blob_tensors:
        return compact_config, blob_tensors, None

    manifest = {
        "blob_file": blob_file,
        "blob_count": len(blob_tensors),
        "marker_schema_version": EXTERNAL_BLOB_MARKER_VERSION,
        "min_externalized_bytes": int(min_externalized_bytes),
        "externalized_tensor_bytes": int(
            sum(np.asarray(value).nbytes for value in blob_tensors.values())
        ),
        "externalized_json_estimate_bytes": int(total_json_externalized_bytes),
    }
    return compact_config, blob_tensors, manifest


def restore_externalized_model_config(
    model_config: dict[str, Any], blob_tensors: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Restore externalized model_config payloads from sidecar tensors."""

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            marker_metadata = _parse_marker_metadata(node)
            if marker_metadata is not None:
                marker_key = marker_metadata["key"]
                marker_dtype = np.dtype(marker_metadata["dtype"])
                marker_shape = tuple(int(v) for v in marker_metadata["shape"])

                if marker_key not in blob_tensors:
                    raise ValueError(
                        "Externalized model_config payload is missing blob key "
                        f"'{marker_key}'."
                    )

                tensor = np.asarray(blob_tensors[marker_key])
                if tuple(int(v) for v in tensor.shape) != marker_shape:
                    raise ValueError(
                        "Externalized model_config payload shape mismatch for blob "
                        f"'{marker_key}': expected {marker_shape}, "
                        f"found {tuple(int(v) for v in tensor.shape)}."
                    )

                if tensor.dtype != marker_dtype:
                    raise ValueError(
                        "Externalized model_config payload dtype mismatch for blob "
                        f"'{marker_key}': expected {marker_dtype}, found {tensor.dtype}."
                    )

                return tensor.tolist()

            return {k: _walk(v) for k, v in node.items()}

        if isinstance(node, list):
            return [_walk(v) for v in node]

        return node

    return _walk(model_config)


def _externalize_list_if_needed(
    value: list[Any],
    *,
    parent_key: str | None,
    parent_dict: dict[str, Any] | None,
    min_externalized_bytes: int,
    counter: int,
) -> dict[str, Any] | None:
    array_value = _list_to_numeric_ndarray(value)
    if array_value is None:
        return None

    target_dtype = _infer_target_dtype(array_value, parent_key=parent_key, parent_dict=parent_dict)
    if target_dtype is not None and array_value.dtype != target_dtype:
        array_value = array_value.astype(target_dtype, copy=False)

    force_externalize = parent_key in _FORCED_EXTERNALIZED_LIST_KEYS
    if (not force_externalize) and (_estimate_json_bytes(value) < min_externalized_bytes):
        return None

    key = f"config_blob_{counter:05d}"
    return {
        EXTERNAL_BLOB_MARKER_KEY: {
            "version": EXTERNAL_BLOB_MARKER_VERSION,
            "key": key,
            "dtype": str(array_value.dtype),
            "shape": [int(v) for v in array_value.shape],
        },
        "_json_bytes": int(_estimate_json_bytes(value)),
        "_tensor": array_value,
    }


def _parse_marker_metadata(node: dict[str, Any]) -> dict[str, Any] | None:
    marker_container = node.get(EXTERNAL_BLOB_MARKER_KEY)
    if marker_container is None:
        return None

    # Only treat this object as a marker if it matches the full marker schema.
    if len(node) != 1:
        return None
    if not isinstance(marker_container, dict):
        return None

    required = {"version", "key", "dtype", "shape"}
    if set(marker_container.keys()) != required:
        return None

    version = marker_container["version"]
    key = marker_container["key"]
    dtype = marker_container["dtype"]
    shape = marker_container["shape"]
    if version != EXTERNAL_BLOB_MARKER_VERSION:
        raise ValueError(
            "Externalized model_config payload marker version mismatch: "
            f"expected {EXTERNAL_BLOB_MARKER_VERSION}, found {version}."
        )
    if not isinstance(key, str) or not key:
        raise ValueError("Externalized model_config payload marker key is invalid.")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("Externalized model_config payload marker dtype is invalid.")
    if not isinstance(shape, list) or not all(isinstance(v, int) and v >= 0 for v in shape):
        raise ValueError("Externalized model_config payload marker shape is invalid.")

    return {
        "key": key,
        "dtype": dtype,
        "shape": shape,
    }


def _infer_target_dtype(
    array_value: np.ndarray,
    *,
    parent_key: str | None,
    parent_dict: dict[str, Any] | None,
) -> np.dtype | None:
    if parent_key in _FORCED_EXTERNALIZED_LIST_KEYS:
        return np.dtype("int32")

    if not isinstance(parent_dict, dict):
        return None

    dtype_value = parent_dict.get("dtype")
    if not isinstance(dtype_value, str):
        return None

    try:
        target = np.dtype(dtype_value)
    except TypeError:
        return None

    if target.kind not in {"b", "i", "u", "f"}:
        return None
    return target


def _list_to_numeric_ndarray(value: list[Any]) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return None

    if arr.dtype == np.dtype("O"):
        return None

    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        return None

    return arr


def _estimate_json_bytes(payload: Any) -> int:
    try:
        return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    except Exception:
        return 0
