#!/usr/bin/env python3
"""Validate the released v04.2 experiment and artifact chain."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT
RESULTS = REPO_ROOT / "results"

DATASET = REPO_ROOT / "data" / "gold" / "gold_enriched_ontology.parquet"
DATASET_METADATA = REPO_ROOT / "data" / "gold" / "gold_enriched_ontology_metadata.json"
SPLIT = REPO_ROOT / "data" / "gold" / "GEN_split_gld_reddit_ids_v02.json"
LEXICON = REPO_ROOT / "results" / "lexicon" / "ontology_lexicon_v04_2_train_only.json"
FEATURE_SELECTION = REPO_ROOT / "results" / "feature_selection" / "ENR_selected_ont_features_anova_train_only.json"
DOWNSTREAM = RESULTS / "anova_revalidation"
METRICS = DOWNSTREAM / "metrics"
PREDICTIONS = DOWNSTREAM / "predictions"
CLASSICAL_ABLATION = RESULTS / "ablation" / "classical"
CNN_ABLATION = RESULTS / "ablation" / "cnn1d"
FIGURE_MANIFEST = RESULTS / "provenance" / "figure_generation_manifest.json"

EXPECTED_SOURCE_DATASET_SHA = "a74cebfa0db51e9d6e34a8b58b16f149db690d2d5bca0cd5d5945a9c2cc0e99f"
EXPECTED_LEXICON_SHA = "313af4e18f4d7b58f87e87a7ac5bbbb19c32d5e7c829816bdad9fa6515e51e77"
EXPECTED_FEATURE_SELECTION_SHA = "03d660c83fdb0a796556edec66736f008df4d1b99b947051c640cb9c1b7ff854"
EXPECTED_SUPPORT = [48, 119, 156]
EXPECTED_SPLITS = {"train": 968, "val": 323, "test": 323}
EXPECTED_SELECTED = [
    "ont_domain_density",
    "ont_InteligenciaArtificial_Negativo",
    "ont_total_negativo_mentions",
    "ont_Innovacion_Neutro",
    "ont_Etica_Negativo",
    "ont_InteligenciaArtificial_Neutro",
    "ont_Tecnologia_Positivo",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_true(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def validate() -> dict[str, Any]:
    errors: list[str] = []

    assert_true(DATASET.exists(), f"missing dataset: {DATASET}", errors)
    assert_true(DATASET_METADATA.exists(), f"missing dataset metadata: {DATASET_METADATA}", errors)
    assert_true(LEXICON.exists(), f"missing lexicon manifest: {LEXICON}", errors)
    assert_true(FEATURE_SELECTION.exists(), f"missing feature selection: {FEATURE_SELECTION}", errors)
    if errors:
        return {"status": "blocked", "errors": errors}

    dataset_sha = sha256(DATASET)
    dataset_metadata = load_json(DATASET_METADATA)
    lexicon_sha = sha256(LEXICON)
    fs_sha = sha256(FEATURE_SELECTION)
    assert_true(
        dataset_metadata.get("source_sha256") == EXPECTED_SOURCE_DATASET_SHA,
        "public dataset does not identify the canonical source dataset",
        errors,
    )
    assert_true(
        dataset_metadata.get("release_sha256") == dataset_sha,
        f"public dataset hash mismatch: {dataset_sha}",
        errors,
    )
    assert_true(lexicon_sha == EXPECTED_LEXICON_SHA, f"lexicon hash mismatch: {lexicon_sha}", errors)
    assert_true(fs_sha == EXPECTED_FEATURE_SELECTION_SHA, f"feature-selection hash mismatch: {fs_sha}", errors)

    df = pd.read_parquet(DATASET)
    ont_cols = [c for c in df.columns if c.startswith("ont_")]
    split_counts = df["split"].value_counts().to_dict()
    assert_true(tuple(df.shape) == (1614, 47), f"unexpected dataset shape: {df.shape}", errors)
    assert_true(split_counts == EXPECTED_SPLITS, f"unexpected split counts: {split_counts}", errors)
    assert_true(len(ont_cols) == 37, f"unexpected ontology feature count: {len(ont_cols)}", errors)
    assert_true("author_id" in df.columns, "public dataset lacks pseudonymous author_id", errors)
    assert_true("autor" not in df.columns, "public dataset exposes Reddit account names", errors)
    assert_true("raw_payload" not in df.columns, "public dataset exposes raw API payloads", errors)

    lexicon = load_json(LEXICON)
    assert_true(lexicon.get("version") == "v04.2_vader_train_scope", "wrong lexicon version", errors)
    assert_true(lexicon.get("induction_scope") == "train_text_only_no_labels", "wrong lexicon induction scope", errors)
    counts = lexicon.get("induction_counts", {})
    assert_true(counts.get("validation_ids_used") == 0, "lexicon used validation ids", errors)
    assert_true(counts.get("test_ids_used") == 0, "lexicon used test ids", errors)

    fs = load_json(FEATURE_SELECTION)
    assert_true(
        fs.get("dataset_sha256") == EXPECTED_SOURCE_DATASET_SHA,
        "feature selection points to wrong source dataset hash",
        errors,
    )
    assert_true(fs.get("lexicon_manifest_sha256") == EXPECTED_LEXICON_SHA, "feature selection points to wrong lexicon hash", errors)
    assert_true(fs.get("train_size") == 968, "feature selection is not train-only", errors)
    assert_true(fs.get("n_candidates") == 37, "feature selection candidate count is not 37", errors)
    assert_true(fs.get("selected_features_train_only") == EXPECTED_SELECTED, "unexpected selected semantic variables", errors)

    summary = pd.read_csv(METRICS / "multiseed_summary_v2.csv")
    raw = pd.read_csv(METRICS / "multiseed_raw_v2.csv")
    assert_true(tuple(summary.shape) == (8, 6), f"unexpected multiseed summary shape: {summary.shape}", errors)
    assert_true(tuple(raw.shape) == (24, 5), f"unexpected multiseed raw shape: {raw.shape}", errors)
    metric_files = sorted(METRICS.glob("*_test_metrics_v2.json"))
    prediction_files = sorted(PREDICTIONS.glob("*_test_predictions_v2.csv"))
    assert_true(len(metric_files) == 24, f"expected 24 metric json files, observed {len(metric_files)}", errors)
    assert_true(len(prediction_files) == 24, f"expected 24 prediction csv files, observed {len(prediction_files)}", errors)
    for path in metric_files:
        payload = load_json(path)
        assert_true(payload.get("pipeline_version") == "canonical_v04_2_vader_train_scope", f"wrong pipeline version in {path.name}", errors)
        assert_true(payload.get("dataset_sha256") == dataset_sha, f"wrong release dataset hash in {path.name}", errors)
        assert_true(payload.get("lexicon_manifest_sha256") == EXPECTED_LEXICON_SHA, f"wrong lexicon hash in {path.name}", errors)
        assert_true(payload.get("support") == EXPECTED_SUPPORT, f"wrong test support in {path.name}", errors)
    for path in prediction_files:
        assert_true(row_count(path) == 323, f"prediction row count is not 323 in {path.name}", errors)

    rdf = load_json(RESULTS / "knowledge_graph" / "materialization_report.json")
    trace = load_json(RESULTS / "traceability" / "traceability_audit_report.json")
    shacl = load_json(RESULTS / "validation" / "shacl" / "baseline_report.json")
    stress = load_json(RESULTS / "validation" / "shacl" / "stress_test_report.json")
    coverage = load_json(RESULTS / "validation" / "coverage" / "coverage_sparsity_report.json")
    model_inputs = load_json(RESULTS / "traceability" / "model_inputs" / "semantic_variable_model_input_report.json")
    reconciliation = load_json(RESULTS / "traceability" / "chain_reconciliation_report.json")
    figure_manifest = load_json(FIGURE_MANIFEST)
    assert_true(rdf.get("status") == "passed" and rdf.get("rdf_triple_count") == 15603, "RDF materialization did not pass expected triple count", errors)
    assert_true(trace.get("status") == "passed" and trace.get("selected_trace_rate_pct") == 100.0, "traceability audit did not pass at 100%", errors)
    assert_true(shacl.get("status") == "passed", "SHACL baseline report did not pass", errors)
    assert_true(stress.get("conforms") is False and stress.get("validation_result_count", 0) > 0, "SHACL stress fixture did not fail as expected", errors)
    assert_true(coverage.get("status") == "passed" and coverage.get("ontology_feature_count") == 37, "coverage report did not pass", errors)
    assert_true(model_inputs.get("status") == "passed" and model_inputs.get("selected_feature_count") == 7, "model-input audit did not pass", errors)
    assert_true(reconciliation.get("status") == "passed", "evidence-chain reconciliation did not pass", errors)
    expected_figures = {
        "fig04_chain_governance_lineage_ledger",
        "fig05_module_coverage",
        "fig06_lexical_sensitivity",
        "fig07_downstream_macro_f1",
        "fig08_downstream_confusion_delta",
        "fig09_prediction_distribution",
        "fig10_module_sensitivity",
        "fig11_feature_stability",
    }
    observed_figures = {item.get("figure_id") for item in figure_manifest.get("figures", [])}
    assert_true(figure_manifest.get("status") == "passed", "figure provenance manifest did not pass", errors)
    assert_true(observed_figures == expected_figures, "figure provenance manifest has an unexpected figure set", errors)
    for item in figure_manifest.get("figures", []):
        output = PROJECT_ROOT / item["included_path"]
        assert_true(output.exists(), f"missing included result figure: {item['included_path']}", errors)
        if output.exists():
            assert_true(sha256(output) == item.get("png_sha256"), f"figure hash mismatch: {item['figure_id']}", errors)
        for source in item.get("input_artifacts", []):
            source_path = PROJECT_ROOT / source["path"]
            assert_true(source_path.exists(), f"missing figure input: {source['path']}", errors)
            if source_path.exists():
                assert_true(sha256(source_path) == source.get("sha256"), f"figure input hash mismatch: {item['figure_id']}", errors)

    classical_manifest = load_json(CLASSICAL_ABLATION / "module_ablation_summary.json")
    classical_raw = pd.read_csv(CLASSICAL_ABLATION / "module_ablation_raw.csv")
    classical_pred = pd.read_csv(CLASSICAL_ABLATION / "module_ablation_predictions.csv")
    assert_true(classical_manifest.get("status") == "passed", "classical ablation manifest did not pass", errors)
    assert_true(tuple(classical_raw.shape)[0] == 126, f"classical raw rows not 126: {classical_raw.shape}", errors)
    assert_true(tuple(classical_pred.shape)[0] == 40698, f"classical prediction rows not 40698: {classical_pred.shape}", errors)

    cnn_manifest = load_json(CNN_ABLATION / "cnn1d_module_ablation_manifest.json")
    cnn_raw = pd.read_csv(CNN_ABLATION / "cnn1d_module_ablation_raw.csv")
    cnn_pred = pd.read_csv(CNN_ABLATION / "cnn1d_module_ablation_predictions.csv")
    assert_true(cnn_manifest.get("dataset_sha256") == dataset_sha, "CNN ablation points to wrong release dataset hash", errors)
    assert_true(cnn_manifest.get("lexicon_manifest_sha256") == EXPECTED_LEXICON_SHA, "CNN ablation points to wrong lexicon hash", errors)
    assert_true(cnn_manifest.get("feature_selection_sha256") == EXPECTED_FEATURE_SELECTION_SHA, "CNN ablation points to wrong feature-selection hash", errors)
    assert_true(cnn_manifest.get("split_sizes") == EXPECTED_SPLITS, "CNN ablation split sizes are wrong", errors)
    assert_true(cnn_manifest.get("test_support") == EXPECTED_SUPPORT, "CNN ablation support is wrong", errors)
    assert_true(tuple(cnn_raw.shape)[0] == 42, f"CNN raw rows not 42: {cnn_raw.shape}", errors)
    assert_true(tuple(cnn_pred.shape)[0] == 13566, f"CNN prediction rows not 13566: {cnn_pred.shape}", errors)

    payload = {
        "status": "passed" if not errors else "blocked",
        "errors": errors,
        "release_dataset_sha256": dataset_sha,
        "source_dataset_sha256": dataset_metadata.get("source_sha256"),
        "lexicon_manifest_sha256": lexicon_sha,
        "feature_selection_sha256": fs_sha,
        "dataset_shape": list(df.shape),
        "split_counts": split_counts,
        "ontology_feature_count": len(ont_cols),
        "selected_feature_count": len(EXPECTED_SELECTED),
        "chain_reconciliation_relations": len(reconciliation.get("checks", [])),
        "included_result_figure_count": len(observed_figures),
        "primary_metric_rows": int(raw.shape[0]),
        "primary_prediction_files": len(prediction_files),
        "classical_ablation_rows": int(classical_raw.shape[0]),
        "classical_prediction_rows": int(classical_pred.shape[0]),
        "cnn_ablation_rows": int(cnn_raw.shape[0]),
        "cnn_prediction_rows": int(cnn_pred.shape[0]),
        "historical_outputs_included": False,
    }
    return payload


def main() -> None:
    payload = validate()
    out = RESULTS / "validation" / "experiment_chain_validation.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(payload["status"])
    if payload["errors"]:
        for error in payload["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
