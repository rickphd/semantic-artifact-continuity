#!/usr/bin/env python3
"""Persist and reconcile the selected semantic input matrix at cell level."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ke_artifact_utils import (
    FEATURE_SELECTION,
    ANOVA_DIR,
    GOLD_METADATA,
    GOLD_PATH,
    PROJECT_ROOT,
    MODEL_INPUT_DIR,
    SPLIT_PATH,
    TRACEABILITY_DIR,
    load_gold,
    rel,
    selected_features,
    sha256,
    stable_post_iri,
    write_csv,
    write_json,
)
from model_input_provenance import matrix_fingerprint, sequence_sha256


METRICS_DIR = ANOVA_DIR / "metrics"
PREDICTIONS_DIR = ANOVA_DIR / "predictions"
MODEL_DIR = ANOVA_DIR / "models"
TRACE_PATH = TRACEABILITY_DIR / "traceability_map.csv"

RAW_MATRIX_PATH = MODEL_INPUT_DIR / "selected_semantic_input_matrix.csv"
STANDARDIZED_MATRIX_PATH = MODEL_INPUT_DIR / "selected_semantic_input_standardized.csv"
CELL_TRACE_PATH = MODEL_INPUT_DIR / "selected_semantic_input_cell_trace.csv"
RUN_LINK_PATH = MODEL_INPUT_DIR / "downstream_semantic_input_fingerprints.csv"
REPORT_PATH = MODEL_INPUT_DIR / "selected_semantic_input_reconciliation_report.json"
REPORT_MD_PATH = MODEL_INPUT_DIR / "selected_semantic_input_reconciliation_report.md"

MODEL_PROFILE = {
    "LR": "train_standardized_float64",
    "RF": "raw_float64",
    "XGB": "raw_float64",
    "CNN1D": "train_standardized_float32",
}
MODEL_PARTITIONS = {
    "LR": ("train", "test"),
    "RF": ("train", "test"),
    "XGB": ("train", "test"),
    "CNN1D": ("train", "val", "test"),
}


def load_parts() -> tuple[dict[str, pd.DataFrame], list[str]]:
    gold = load_gold()
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    features = selected_features()
    indexed = gold.set_index("id", drop=False)
    parts = {}
    for name in ("train", "val", "test"):
        ids = split[name]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate identifiers in the {name} partition")
        missing = [post_id for post_id in ids if post_id not in indexed.index]
        if missing:
            raise ValueError(f"Missing Gold identifiers in {name}: {missing[:5]}")
        parts[name] = indexed.loc[ids].reset_index(drop=True)
    return parts, features


def write_matrix(
    path: Path,
    parts: dict[str, pd.DataFrame],
    features: list[str],
    matrices: dict[str, np.ndarray],
) -> None:
    rows = []
    for split_name in ("train", "val", "test"):
        ids = parts[split_name]["id"].astype(str).tolist()
        matrix = matrices[split_name]
        for row_index, post_id in enumerate(ids):
            row = {"post_id": post_id, "split": split_name}
            row.update(
                {feature: repr(float(matrix[row_index, col_index])) for col_index, feature in enumerate(features)}
            )
            rows.append(row)
    write_csv(path, rows, ["post_id", "split", *features])


def trace_lookup() -> tuple[dict[tuple[str, str], dict[str, str]], int]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    duplicates = 0
    with TRACE_PATH.open(encoding="utf-8", newline="") as handle:
        for trace_row_number, row in enumerate(csv.DictReader(handle), start=2):
            key = (row["post_id"], row["feature"])
            if key in lookup:
                duplicates += 1
            enriched = dict(row)
            enriched["trace_row_number"] = str(trace_row_number)
            lookup[key] = enriched
    return lookup, duplicates


def build_cell_trace(
    parts: dict[str, pd.DataFrame],
    features: list[str],
    trace: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    counts = {
        "expected_cells": 0,
        "matched_cells": 0,
        "nonzero_cells": 0,
        "matched_nonzero_trace_cells": 0,
        "missing_nonzero_trace_cells": 0,
        "value_mismatches": 0,
        "split_mismatches": 0,
        "post_iri_mismatches": 0,
        "unexpected_zero_trace_rows": 0,
    }
    for split_name in ("train", "val", "test"):
        part = parts[split_name]
        for _, gold_row in part.iterrows():
            post_id = str(gold_row["id"])
            post_iri = stable_post_iri(post_id)
            for feature in features:
                value = float(0 if pd.isna(gold_row[feature]) else gold_row[feature])
                trace_row = trace.get((post_id, feature))
                is_nonzero = value != 0.0
                trace_value = float(trace_row["value"]) if trace_row else None
                value_match = trace_value is not None and np.isclose(
                    value, trace_value, rtol=0.0, atol=1e-12
                )
                split_match = trace_row is not None and trace_row["split"] == split_name
                iri_match = trace_row is not None and trace_row["post_iri"] == post_iri
                cell_match = (
                    value_match and split_match and iri_match
                    if is_nonzero
                    else trace_row is None
                )

                counts["expected_cells"] += 1
                counts["nonzero_cells"] += int(is_nonzero)
                counts["matched_cells"] += int(cell_match)
                if is_nonzero:
                    counts["missing_nonzero_trace_cells"] += int(trace_row is None)
                    counts["matched_nonzero_trace_cells"] += int(
                        trace_row is not None and value_match and split_match and iri_match
                    )
                    counts["value_mismatches"] += int(trace_row is not None and not value_match)
                    counts["split_mismatches"] += int(trace_row is not None and not split_match)
                    counts["post_iri_mismatches"] += int(trace_row is not None and not iri_match)
                else:
                    counts["unexpected_zero_trace_rows"] += int(trace_row is not None)

                rows.append(
                    {
                        "post_id": post_id,
                        "split": split_name,
                        "feature": feature,
                        "input_value": repr(value),
                        "post_iri": post_iri,
                        "trace_row_number": trace_row["trace_row_number"] if trace_row else "",
                        "trace_value": trace_row["value"] if trace_row else "",
                        "trace_expected": is_nonzero,
                        "cell_match": cell_match,
                    }
                )
    return rows, counts


def profile_matrices(
    parts: dict[str, pd.DataFrame], features: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], StandardScaler]:
    raw = {
        name: part[features].fillna(0).astype(float).to_numpy()
        for name, part in parts.items()
    }
    scaler = StandardScaler()
    standardized = {"train": scaler.fit_transform(raw["train"])}
    for name in ("val", "test"):
        standardized[name] = scaler.transform(raw[name])
    return raw, standardized, scaler


def profile_for(
    model: str,
    split_name: str,
    raw: dict[str, np.ndarray],
    standardized: dict[str, np.ndarray],
) -> np.ndarray:
    if model in {"LR", "CNN1D"}:
        matrix = standardized[split_name]
        return matrix.astype(np.float32) if model == "CNN1D" else matrix
    return raw[split_name]


def link_runs(
    parts: dict[str, pd.DataFrame],
    features: list[str],
    raw: dict[str, np.ndarray],
    standardized: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    counts = {
        "declared_enriched_runs": 0,
        "linked_enriched_runs": 0,
        "runtime_captured_runs": 0,
        "reconstructed_partition_links": 0,
        "prediction_order_mismatches": 0,
        "dataset_hash_mismatches": 0,
        "feature_order_mismatches": 0,
    }
    release_dataset_hash = sha256(GOLD_PATH)
    metadata = json.loads(GOLD_METADATA.read_text(encoding="utf-8"))
    source_dataset_hash = metadata.get("source_sha256")
    accepted_dataset_hashes = {release_dataset_hash, source_dataset_hash}
    for metric_path in sorted(METRICS_DIR.glob("*_ENR_seed*_test_metrics_v2.json")):
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        model = metric.get("model")
        if model not in MODEL_PROFILE:
            continue
        counts["declared_enriched_runs"] += 1
        dataset_match = metric.get("dataset_sha256") in accepted_dataset_hashes
        feature_match = metric.get("ont_features") == features
        counts["dataset_hash_mismatches"] += int(not dataset_match)
        counts["feature_order_mismatches"] += int(not feature_match)

        prediction_path = PREDICTIONS_DIR / metric_path.name.replace(
            "_test_metrics_v2.json", "_test_predictions_v2.csv"
        )
        prediction_order_match = False
        if prediction_path.exists():
            prediction_ids = pd.read_csv(prediction_path, dtype={"id": str})["id"].tolist()
            expected_ids = parts["test"]["id"].astype(str).tolist()
            prediction_order_match = prediction_ids == expected_ids
        counts["prediction_order_mismatches"] += int(not prediction_order_match)

        runtime_capture = metric.get("semantic_input_provenance")
        counts["runtime_captured_runs"] += int(bool(runtime_capture))
        run_linked = dataset_match and feature_match and prediction_order_match
        counts["linked_enriched_runs"] += int(run_linked)

        for split_name in MODEL_PARTITIONS[model]:
            matrix = profile_for(model, split_name, raw, standardized)
            fingerprint = matrix_fingerprint(
                parts[split_name]["id"].astype(str).tolist(),
                features,
                matrix,
                MODEL_PROFILE[model],
            )
            runtime_partition = runtime_capture.get(split_name) if runtime_capture else None
            runtime_hash_match = (
                runtime_partition.get("matrix_sha256") == fingerprint["matrix_sha256"]
                if runtime_partition
                else None
            )
            rows.append(
                {
                    "run_id": metric_path.stem.replace("_test_metrics_v2", ""),
                    "model": model,
                    "condition": metric.get("condition"),
                    "seed": metric.get("seed"),
                    "split": split_name,
                    **fingerprint,
                    "release_dataset_sha256": release_dataset_hash,
                    "source_dataset_sha256": source_dataset_hash,
                    "metric_dataset_match": dataset_match,
                    "metric_feature_order_match": feature_match,
                    "test_prediction_order_match": prediction_order_match,
                    "runtime_hash_available": runtime_partition is not None,
                    "runtime_hash_match": "" if runtime_hash_match is None else runtime_hash_match,
                    "evidence_scope": (
                        "runtime_captured"
                        if runtime_partition is not None
                        else "reconstructed_from_declared_sources"
                    ),
                    "metric_path": rel(metric_path),
                }
            )
            counts["reconstructed_partition_links"] += int(run_linked)
    return rows, counts


def validate_saved_lr_scaler(scaler: StandardScaler) -> dict[str, object]:
    path = MODEL_DIR / "LR_ENR_scaler_v2.joblib"
    if not path.exists():
        return {"available": False, "parameters_match": False}
    saved = joblib.load(path)
    mean_match = np.array_equal(saved.mean_, scaler.mean_)
    scale_match = np.array_equal(saved.scale_, scaler.scale_)
    return {
        "available": True,
        "path": rel(path),
        "sha256": sha256(path),
        "mean_match": bool(mean_match),
        "scale_match": bool(scale_match),
        "parameters_match": bool(mean_match and scale_match),
    }


def main() -> None:
    parts, features = load_parts()
    raw, standardized, scaler = profile_matrices(parts, features)
    write_matrix(RAW_MATRIX_PATH, parts, features, raw)
    write_matrix(STANDARDIZED_MATRIX_PATH, parts, features, standardized)

    trace, duplicate_trace_keys = trace_lookup()
    cell_rows, cell_counts = build_cell_trace(parts, features, trace)
    write_csv(CELL_TRACE_PATH, cell_rows)

    run_rows, run_counts = link_runs(parts, features, raw, standardized)
    write_csv(RUN_LINK_PATH, run_rows)

    expected_cells = sum(len(part) for part in parts.values()) * len(features)
    cell_passed = (
        cell_counts["expected_cells"] == expected_cells
        and cell_counts["matched_cells"] == expected_cells
        and cell_counts["missing_nonzero_trace_cells"] == 0
        and cell_counts["value_mismatches"] == 0
        and cell_counts["split_mismatches"] == 0
        and cell_counts["post_iri_mismatches"] == 0
        and cell_counts["unexpected_zero_trace_rows"] == 0
        and duplicate_trace_keys == 0
    )
    runs_passed = (
        run_counts["declared_enriched_runs"] == 12
        and run_counts["linked_enriched_runs"] == 12
        and run_counts["prediction_order_mismatches"] == 0
        and run_counts["dataset_hash_mismatches"] == 0
        and run_counts["feature_order_mismatches"] == 0
    )
    scaler_check = validate_saved_lr_scaler(scaler)
    status = "passed" if cell_passed and runs_passed and scaler_check["parameters_match"] else "attention_required"

    profile_fingerprints = {}
    for profile_name, matrices in (
        ("raw_float64", raw),
        ("train_standardized_float64", standardized),
        (
            "train_standardized_float32",
            {name: matrix.astype(np.float32) for name, matrix in standardized.items()},
        ),
    ):
        profile_fingerprints[profile_name] = {
            split_name: matrix_fingerprint(
                parts[split_name]["id"].astype(str).tolist(),
                features,
                matrices[split_name],
                profile_name,
            )
            for split_name in ("train", "val", "test")
        }

    payload = {
        "status": status,
        "evidence_scope": {
            "cell_reconciliation": "persisted",
            "stored_run_linkage": "reconstructed_from_declared_sources",
            "runtime_capture_available": run_counts["runtime_captured_runs"] == 12,
            "interpretation": (
                "Cell values, row order, feature order, and deterministic transformation profiles "
                "are reconciled. Existing runs did not record runtime matrix hashes."
            ),
        },
        "source": {
            "gold_path": rel(GOLD_PATH),
            "release_gold_sha256": sha256(GOLD_PATH),
            "source_gold_sha256": json.loads(
                GOLD_METADATA.read_text(encoding="utf-8")
            ).get("source_sha256"),
            "split_path": rel(SPLIT_PATH),
            "split_sha256": sha256(SPLIT_PATH),
            "feature_selection_path": rel(FEATURE_SELECTION),
            "feature_selection_sha256": sha256(FEATURE_SELECTION),
            "traceability_path": rel(TRACE_PATH),
            "traceability_sha256": sha256(TRACE_PATH),
        },
        "selected_features": features,
        "selected_feature_count": len(features),
        "ordered_features_sha256": sequence_sha256(features),
        "partition_sizes": {name: len(part) for name, part in parts.items()},
        "cell_reconciliation": {**cell_counts, "duplicate_trace_keys": duplicate_trace_keys},
        "run_reconciliation": run_counts,
        "saved_lr_scaler_check": scaler_check,
        "profile_fingerprints": profile_fingerprints,
        "outputs": {
            "raw_matrix": rel(RAW_MATRIX_PATH),
            "standardized_matrix": rel(STANDARDIZED_MATRIX_PATH),
            "cell_trace": rel(CELL_TRACE_PATH),
            "run_fingerprints": rel(RUN_LINK_PATH),
        },
    }
    write_json(REPORT_PATH, payload)

    md = [
        "# Selected Semantic Input Reconciliation",
        "",
        f"Status: `{status}`",
        f"Expected cells: `{cell_counts['expected_cells']}`",
        f"Matched cells: `{cell_counts['matched_cells']}`",
        f"Nonzero cells with matching trace: `{cell_counts['matched_nonzero_trace_cells']}`",
        f"Missing nonzero traces: `{cell_counts['missing_nonzero_trace_cells']}`",
        f"Value mismatches: `{cell_counts['value_mismatches']}`",
        f"Linked enriched runs: `{run_counts['linked_enriched_runs']}` of `{run_counts['declared_enriched_runs']}`",
        f"Runtime-captured stored runs: `{run_counts['runtime_captured_runs']}`",
        "",
        "Existing runs are linked by dataset hash, ordered selected features, deterministic split order, "
        "test-prediction order, and model-specific transformation profile. They did not persist runtime "
        "matrix hashes; those hashes are captured only by future executions of the instrumented trainer.",
        "",
    ]
    REPORT_MD_PATH.write_text("\n".join(md), encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
