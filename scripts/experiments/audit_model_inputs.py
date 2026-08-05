#!/usr/bin/env python3
"""Audit semantic variables used as downstream model inputs."""

from __future__ import annotations

import csv
import hashlib
import json

from ke_artifact_utils import (
    FEATURE_SELECTION,
    LEXICON_MANIFEST,
    MODEL_INPUT_DIR,
    TRACEABILITY_DIR,
    feature_to_module,
    load_gold,
    ontology_columns,
    rel,
    selected_features,
    write_csv,
    write_json,
)


def main() -> None:
    df = load_gold()
    ont_cols = set(ontology_columns(df))
    selected = selected_features()
    fs = json.loads(FEATURE_SELECTION.read_text(encoding="utf-8"))
    lexicon = json.loads(LEXICON_MANIFEST.read_text(encoding="utf-8"))
    trace_rows = list(csv.DictReader((TRACEABILITY_DIR / "traceability_map.csv").open(encoding="utf-8")))
    trace_features = {r["feature"] for r in trace_rows if r["feature"]}

    rows = []
    for feature in selected:
        exists = feature in ont_cols
        nonzero = int((df[feature].fillna(0).astype(float) != 0).sum()) if exists else 0
        rows.append(
            {
                "feature": feature,
                "module": feature_to_module(feature),
                "exists_in_gold": exists,
                "nonzero_count": nonzero,
                "trace_present": feature in trace_features,
                "selection_method": fs.get("method"),
                "selection_scope": "train_only" if fs.get("train_size") == 968 else "unknown",
                "lexicon_scope": lexicon.get("induction_scope", "unknown"),
            }
        )
    write_csv(MODEL_INPUT_DIR / "model_input_trace.csv", rows)
    missing = [r["feature"] for r in rows if not r["exists_in_gold"] or not r["trace_present"]]
    lexicon_clean = (
        lexicon.get("induction_scope") == "train_text_only_no_labels"
        and lexicon.get("induction_counts", {}).get("validation_ids_used") == 0
        and lexicon.get("induction_counts", {}).get("test_ids_used") == 0
    )
    payload = {
        "status": "passed" if not missing and fs.get("train_size") == 968 and lexicon_clean else "attention_required",
        "selected_feature_count": len(selected),
        "missing_or_untraced_features": missing,
        "train_only_selection": fs.get("train_size") == 968 and "train-only" in fs.get("method", "").lower(),
        "train_scoped_external_lexicon": lexicon_clean,
        "lexicon_manifest_sha256": hashlib.sha256(LEXICON_MANIFEST.read_bytes()).hexdigest(),
        "candidate_feature_count": fs.get("n_candidates"),
        "selected_feature_source": rel(FEATURE_SELECTION),
    }
    write_json(MODEL_INPUT_DIR / "semantic_variable_model_input_report.json", payload)
    md = [
        "# Semantic Variable Model Input Report",
        "",
        f"Status: `{payload['status']}`",
        f"Selected features: `{payload['selected_feature_count']}`",
        f"Train only selection: `{payload['train_only_selection']}`",
        f"Train scoped external lexicon: `{payload['train_scoped_external_lexicon']}`",
        "",
    ]
    if missing:
        md.append("Missing or untraced features:")
        md.extend(f"- `{x}`" for x in missing)
    else:
        md.append("All selected semantic variables exist in the gold dataset and are present in the traceability map.")
    (MODEL_INPUT_DIR / "semantic_variable_model_input_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(payload["status"])


if __name__ == "__main__":
    main()
