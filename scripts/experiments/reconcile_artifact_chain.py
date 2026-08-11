#!/usr/bin/env python3
"""Reconcile the active v04.2 evidence chain at record level."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ke_artifact_utils import (
    ANOVA_DIR,
    FEATURE_SELECTION,
    GOLD_PATH,
    KNOWLEDGE_GRAPH_DIR,
    MODEL_INPUT_DIR,
    SHACL_DIR,
    SPLIT_PATH,
    TRACEABILITY_DIR,
    load_json,
    selected_features,
    stable_post_iri,
    write_json,
)


PREDICTIONS = ANOVA_DIR / "predictions"
MODEL_INPUT_TRACE = MODEL_INPUT_DIR / "model_input_trace.csv"
TRACEABILITY_MAP = TRACEABILITY_DIR / "traceability_map.csv"
RDF_REPORT = KNOWLEDGE_GRAPH_DIR / "materialization_report.json"
TRACEABILITY_REPORT = TRACEABILITY_DIR / "traceability_audit_report.json"
CELL_INPUT_REPORT = MODEL_INPUT_DIR / "selected_semantic_input_reconciliation_report.json"
SHACL_REPORT = SHACL_DIR / "baseline_report.json"


def row(
    relation: str,
    unit: str,
    expected: int,
    verified: int,
    exceptions: int,
    interpretation: str,
) -> dict[str, object]:
    return {
        "relation": relation,
        "unit": unit,
        "expected": expected,
        "verified": verified,
        "exceptions": exceptions,
        "status": "passed" if expected == verified and exceptions == 0 else "attention_required",
        "interpretation": interpretation,
    }


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    split = load_json(SPLIT_PATH)
    selected = selected_features()
    trace = pd.read_csv(TRACEABILITY_MAP)
    model_inputs = pd.read_csv(MODEL_INPUT_TRACE)
    rdf = load_json(RDF_REPORT)
    trace_report = load_json(TRACEABILITY_REPORT)
    cell_input_report = load_json(CELL_INPUT_REPORT)
    shacl = load_json(SHACL_REPORT)

    gold_ids = set(gold["id"].astype(str))
    expected_iris = {post_id: stable_post_iri(post_id) for post_id in gold_ids}
    trace_posts = set(trace["post_id"].astype(str))
    traced_iris = dict(zip(trace["post_id"].astype(str), trace["post_iri"].astype(str)))
    resource_exceptions = len(gold_ids - trace_posts) + sum(
        traced_iris.get(post_id) != iri for post_id, iri in expected_iris.items()
    )

    selected_trace = trace[(trace["feature"].isin(selected)) & (trace["value"] != 0)].copy()
    activation_exceptions = int(
        selected_trace["post_id"].isna().sum()
        + selected_trace["module"].isna().sum()
        + selected_trace["post_iri"].isna().sum()
    )

    input_features = set(model_inputs["feature"].astype(str))
    input_exceptions = len(set(selected) - input_features) + len(input_features - set(selected))
    input_exceptions += int((model_inputs["trace_present"] != True).sum())
    input_exceptions += int((model_inputs["selection_scope"] != "train_only").sum())
    cell_counts = cell_input_report["cell_reconciliation"]
    cell_exceptions = (
        int(cell_counts["expected_cells"]) - int(cell_counts["matched_cells"])
        + int(cell_counts["duplicate_trace_keys"])
    )

    gold_test = gold[gold["split"] == "test"].copy()
    gold_test_ids = set(gold_test["id"].astype(str))
    gold_test_labels = dict(zip(gold_test["id"].astype(str), gold_test["label"].astype(int)))
    split_test_ids = set(str(value) for value in split["test"])
    split_exceptions = len(gold_test_ids ^ split_test_ids)

    prediction_files = sorted(PREDICTIONS.glob("*_test_predictions_v2.csv"))
    prediction_exceptions = 0
    prediction_records = 0
    evaluation_units = 0
    for prediction_file in prediction_files:
        prediction = pd.read_csv(prediction_file)
        prediction["id"] = prediction["id"].astype(str)
        evaluation_units += 1
        prediction_records += len(prediction)
        prediction_exceptions += int((prediction["split"] != "test").sum())
        prediction_exceptions += int(prediction["id"].duplicated().sum())
        prediction_ids = set(prediction["id"])
        prediction_exceptions += len(prediction_ids ^ gold_test_ids)
        prediction_exceptions += sum(
            gold_test_labels.get(post_id) != int(label)
            for post_id, label in zip(prediction["id"], prediction["y_true"])
        )

    expected_prediction_records = len(gold_test_ids) * len(prediction_files)
    conformance_exceptions = int(not shacl["baseline"]["conforms"]) + int(
        shacl["baseline"]["validation_result_count"] != 0
    )

    checks = [
        row(
            "Canonical records to RDF resources",
            "canonical records",
            len(gold_ids),
            int(rdf["rdf_post_count"]),
            resource_exceptions,
            "Each canonical record is represented by one stable RDF resource.",
        ),
        row(
            "Selected semantic activations to source records",
            "nonzero selected activations",
            int(trace_report["nonzero_selected_activations"]),
            len(selected_trace),
            activation_exceptions,
            "Each selected activation retains a record, semantic-variable, module, and RDF-resource link; domain variables additionally retain their concept links.",
        ),
        row(
            "Selected semantic input cells to governed source values",
            "selected input cells",
            int(cell_counts["expected_cells"]),
            int(cell_counts["matched_cells"]),
            input_exceptions + cell_exceptions,
            "All cells in the seven-variable interface retain ordered record and variable identity; every nonzero value matches its traceability relation.",
        ),
        row(
            "RDF graph to declared structural constraints",
            "baseline graph",
            1,
            1 if shacl["baseline"]["conforms"] else 0,
            conformance_exceptions,
            "The baseline RDF graph conforms to the declared structural constraints.",
        ),
        row(
            "Test records to prediction records",
            "record assignments across evaluation units",
            expected_prediction_records,
            prediction_records,
            split_exceptions + prediction_exceptions,
            f"All {len(gold_test_ids)} test records are reconciled across {evaluation_units} evaluation units with matching reference labels.",
        ),
    ]
    status = "passed" if all(check["status"] == "passed" for check in checks) else "attention_required"
    payload = {
        "status": status,
        "active_evidence_lineage": "v04.2",
        "checks": checks,
        "summary": {
            "checked_relations": len(checks),
            "passed_relations": sum(check["status"] == "passed" for check in checks),
            "exceptions": sum(int(check["exceptions"]) for check in checks),
        },
    }
    write_json(TRACEABILITY_DIR / "chain_reconciliation_report.json", payload)

    lines = [
        "# Evidence Chain Reconciliation",
        "",
        f"Status: `{status}`",
        "",
        "This report reconciles the active evidence lineage at the level of scientific records and evidence relations.",
        "",
        "| Evidence relation | Unit | Expected | Verified | Exceptions | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check['relation']} | {check['unit']} | {check['expected']} | {check['verified']} | {check['exceptions']} | {check['status']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {check['interpretation']}" for check in checks)
    (TRACEABILITY_DIR / "chain_reconciliation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(status)
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
