#!/usr/bin/env python3
"""Audit source to activation to RDF traceability."""

from __future__ import annotations

import csv

from ke_artifact_utils import TRACEABILITY_DIR, load_gold, ontology_columns, selected_features, stable_post_iri, write_json


def main() -> None:
    df = load_gold()
    ont_cols = ontology_columns(df)
    selected = set(selected_features())
    trace_path = TRACEABILITY_DIR / "traceability_map.csv"
    with trace_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    post_ids = set(df["id"].astype(str))
    trace_post_ids = {r["post_id"] for r in rows}
    expected_iris = {stable_post_iri(pid) for pid in post_ids}
    trace_iris = {r["post_iri"] for r in rows}
    nonzero_selected = 0
    traced_selected = 0
    for _, row in df.iterrows():
        for feature in selected:
            if float(row[feature]) != 0.0:
                nonzero_selected += 1
    for r in rows:
        if r["feature"] in selected and float(r["value"]) != 0.0 and (r["concept_uri"] or r["module"]):
            traced_selected += 1

    payload = {
        "status": "passed"
        if post_ids == trace_post_ids and expected_iris.issubset(trace_iris) and nonzero_selected == traced_selected
        else "attention_required",
        "gold_post_count": len(post_ids),
        "trace_post_count": len(trace_post_ids),
        "missing_posts_in_trace": sorted(post_ids - trace_post_ids)[:20],
        "missing_iris_in_trace_count": len(expected_iris - trace_iris),
        "ontology_feature_count": len(ont_cols),
        "selected_feature_count": len(selected),
        "nonzero_selected_activations": nonzero_selected,
        "traced_selected_activations": traced_selected,
        "selected_trace_rate_pct": 100.0 if nonzero_selected == 0 else traced_selected / nonzero_selected * 100,
        "traceability_map": "results/traceability/traceability_map.csv",
    }
    write_json(TRACEABILITY_DIR / "traceability_audit_report.json", payload)
    print(payload["status"], payload["selected_trace_rate_pct"])


if __name__ == "__main__":
    main()
