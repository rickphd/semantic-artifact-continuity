"""Deterministic fingerprints for semantic matrices consumed by downstream models."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np


def sequence_sha256(values: Sequence[object]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def matrix_sha256(matrix: object) -> str:
    array = np.asarray(matrix)
    if array.dtype.kind != "f":
        array = array.astype(np.float64)
    dtype = np.dtype(f"<f{array.dtype.itemsize}")
    canonical = np.ascontiguousarray(array, dtype=dtype)
    header = json.dumps(
        {"shape": list(canonical.shape), "dtype": canonical.dtype.str},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\n" + canonical.tobytes(order="C")).hexdigest()


def matrix_fingerprint(
    post_ids: Sequence[object],
    features: Sequence[str],
    matrix: object,
    profile: str,
) -> dict[str, object]:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional matrix, received shape {array.shape}")
    if array.shape != (len(post_ids), len(features)):
        raise ValueError(
            "Matrix dimensions do not match the ordered post and feature sequences: "
            f"{array.shape} != {(len(post_ids), len(features))}"
        )
    return {
        "profile": profile,
        "rows": int(array.shape[0]),
        "columns": int(array.shape[1]),
        "dtype": str(array.dtype),
        "ordered_post_ids_sha256": sequence_sha256(post_ids),
        "ordered_features_sha256": sequence_sha256(features),
        "matrix_sha256": matrix_sha256(array),
    }

