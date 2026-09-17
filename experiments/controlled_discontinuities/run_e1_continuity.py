#!/usr/bin/env python3
"""Run E1 controlled continuity faults on isolated package copies.

This harness executes the 4 controls and 22 non-SHACL fault cases listed in
../../E1_CASES.json. It does not run the independent SHACL fixture suite.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD
from sklearn.preprocessing import StandardScaler


from replay_support import PACKAGE, copy_baseline, scratch_root, safe_reset as reset_scratch

ROOT = PACKAGE.parents[1]
CASES_JSON = PACKAGE / "E1_CASES.json"
RUN_ROOT = None
FREEZE_PACKAGE = ROOT
CAMPAIGN_PACKAGE = ROOT
GITHUB_REPO = ROOT
DETECTOR_SCRIPTS = ROOT / "scripts/experiments"
PYTHON = Path(sys.executable)

RR = Namespace("http://wfrp.ia/ontologia/ia-sentimiento#")
RRSENT = Namespace("http://wfrp.ia/sentimiento/core#")
RRDOM = Namespace("http://wfrp.ia/dominio/ciencia-tecnologia#")
BASE = Namespace("http://wfrp.ia/resource/reddit/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


SCRIPT_NAMES = [
    "audit_traceability.py",
    "audit_model_inputs.py",
    "audit_model_input_cells.py",
    "run_shacl_conformance.py",
    "reconcile_artifact_chain.py",
]

SELECTED_RUN = "LR_ENR_seed42_test_predictions_v2.csv"


@dataclass
class DetectorRun:
    phase: str
    script: str
    returncode: int
    stdout: str
    stderr: str
    parsed_status: str | None
    detected: bool
    operational_error: bool
    details: dict[str, Any]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_reset(path: Path) -> None:
    reset_scratch(path, RUN_ROOT)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            hashes[str(path.relative_to(root))] = sha256(path)
    return hashes


def sequence_sha256(values: list[Any]) -> str:
    return hashlib.sha256("\n".join(str(v) for v in values).encode("utf-8")).hexdigest()


def matrix_sha256(matrix: Any) -> str:
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


def stable_post_iri(post_id: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_\\-]", "_", str(post_id))
    return f"http://wfrp.ia/resource/reddit/post/{cleaned or 'missing'}"


def concept_for_feature(feature: str) -> str | None:
    parts = feature.split("_")
    if len(parts) >= 3 and parts[1] not in {"has", "count", "total", "domain"}:
        return parts[1]
    return None


def prepare_template() -> Path:
    template = RUN_ROOT / "_prepared_baseline" / "package"
    safe_reset(template.parent)
    copy_baseline(FREEZE_PACKAGE, template, CAMPAIGN_PACKAGE, DETECTOR_SCRIPTS)

    # Generate the clean reports required by the chain consumer inside the template.
    for script in [
        "audit_traceability.py",
        "audit_model_inputs.py",
        "audit_model_input_cells.py",
        "run_shacl_conformance.py",
        "reconcile_artifact_chain.py",
    ]:
        result = run_script(template, script, RUN_ROOT / "_prepared_baseline" / "logs", "prepare")
        if result.returncode != 0:
            raise RuntimeError(f"Template preparation failed at {script}: {result.stderr}")

    write_json(RUN_ROOT / "baseline_hashes.json", file_hashes(template))
    return template


def run_script(package: Path, script_name: str, log_dir: Path, phase: str) -> DetectorRun:
    script = package / "scripts" / "experiments" / script_name
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), str(script)] if PYTHON.exists() else [sys.executable, str(script)]
    proc = subprocess.run(cmd, cwd=package, text=True, capture_output=True, env=env)
    stem = f"{phase}_{script_name.replace('.py', '')}"
    (log_dir / f"{stem}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (log_dir / f"{stem}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    parsed_status, details = parse_script_output(package, script_name, proc.returncode)
    detected = parsed_status in {"attention_required", "failed"}
    operational_error = proc.returncode != 0 and not detected
    return DetectorRun(phase, script_name, proc.returncode, proc.stdout, proc.stderr, parsed_status, detected, operational_error, details)


def parse_script_output(package: Path, script_name: str, returncode: int) -> tuple[str | None, dict[str, Any]]:
    details: dict[str, Any] = {}
    candidates = {
        "audit_traceability.py": package / "results" / "traceability" / "traceability_audit_report.json",
        "audit_model_inputs.py": package / "results" / "traceability" / "model_inputs" / "semantic_variable_model_input_report.json",
        "audit_model_input_cells.py": package / "results" / "traceability" / "model_inputs" / "selected_semantic_input_reconciliation_report.json",
        "run_shacl_conformance.py": package / "results" / "validation" / "shacl" / "baseline_report.json",
        "reconcile_artifact_chain.py": package / "results" / "traceability" / "chain_reconciliation_report.json",
    }
    path = candidates.get(script_name)
    if path and path.exists():
        try:
            payload = load_json(path)
            details = payload
            status = payload.get("status")
            if script_name == "run_shacl_conformance.py":
                baseline = payload.get("baseline", {})
                if baseline.get("conforms") is False:
                    status = "attention_required"
            return status, details
        except Exception as exc:  # noqa: BLE001
            return "error", {"parse_error": str(exc), "path": str(path)}
    if returncode != 0:
        return "error", {"missing_report": str(path) if path else script_name}
    return None, {}


def first_trace_row(package: Path, selected_only: bool = True) -> dict[str, str]:
    selected = set(load_json(package / "results" / "feature_selection" / "ENR_selected_ont_features_anova_train_only.json")["selected_features_train_only"])
    trace = pd.read_csv(package / "results" / "traceability" / "traceability_map.csv", keep_default_na=False)
    mask = trace["feature"].astype(str).ne("") & trace["value"].astype(float).ne(0.0)
    if selected_only:
        mask &= trace["feature"].isin(selected)
    row = trace[mask].iloc[0].to_dict()
    return {k: str(v) for k, v in row.items()}


def prediction_path(package: Path) -> Path:
    return package / "results" / "anova_revalidation" / "predictions" / SELECTED_RUN


def selected_features(package: Path) -> list[str]:
    return load_json(package / "results" / "feature_selection" / "ENR_selected_ont_features_anova_train_only.json")["selected_features_train_only"]


def mutation_details(case_id: str, package: Path) -> dict[str, Any]:
    details: dict[str, Any] = {}
    gold_path = package / "data" / "gold" / "gold_enriched_ontology.parquet"
    split_path = package / "data" / "gold" / "GEN_split_gld_reddit_ids_v02.json"
    trace_path = package / "results" / "traceability" / "traceability_map.csv"
    rdf_path = package / "results" / "knowledge_graph" / "posts.ttl"
    features_path = package / "results" / "feature_selection" / "ENR_selected_ont_features_anova_train_only.json"

    if case_id == "E1-C0-pristine-copy":
        return {"mutation": "none"}
    if case_id == "E1-C1-trace-row-permutation":
        trace = pd.read_csv(trace_path, keep_default_na=False)
        trace.iloc[::-1].to_csv(trace_path, index=False)
        return {"mutation": "trace rows reversed", "rows": len(trace)}
    if case_id == "E1-C2-rdf-triple-order-reserialization":
        g = Graph()
        g.parse(rdf_path, format="turtle")
        g.serialize(destination=rdf_path, format="turtle")
        return {"mutation": "rdf graph reserialized", "triples": len(g)}
    if case_id == "E1-C3-regenerated-reports-unchanged-inputs":
        return {"mutation": "none; detectors regenerate reports"}

    if case_id in {"E1-I1-remove-selected-trace-row", "E1-SR1-stale-traceability-success-report"}:
        trace = pd.read_csv(trace_path, keep_default_na=False)
        row = first_trace_row(package, selected_only=True)
        idx = trace[(trace["post_id"].astype(str) == row["post_id"]) & (trace["feature"].astype(str) == row["feature"])].index[0]
        trace.drop(index=idx).to_csv(trace_path, index=False)
        return {"removed_trace": row, "rows_after": int(len(trace) - 1)}

    if case_id == "E1-I2-remove-rdf-post-resource":
        row = first_trace_row(package, selected_only=True)
        post = URIRef(row["post_iri"])
        g = Graph()
        g.parse(rdf_path, format="turtle")
        removed = list(g.triples((post, None, None))) + list(g.triples((None, None, post)))
        for triple in removed:
            g.remove(triple)
        g.serialize(destination=rdf_path, format="turtle")
        return {"removed_post_id": row["post_id"], "removed_triples": len(removed)}

    if case_id == "E1-I3-add-extra-valid-rdf-post":
        g = Graph()
        g.parse(rdf_path, format="turtle")
        post = URIRef(f"{BASE}post/e1extra001")
        author = URIRef(f"{BASE}author/e1extra")
        subreddit = URIRef(f"{BASE}subreddit/e1extra")
        g.add((post, RDF.type, RR.Post))
        g.add((author, RDF.type, RR.Author))
        g.add((subreddit, RDF.type, RR.Subreddit))
        g.add((post, RR.identificador, Literal("e1extra001", datatype=XSD.string)))
        g.add((post, RR.fechaCreacion, Literal("2026-09-15T00:00:00+00:00", datatype=XSD.dateTime)))
        g.add((post, RR.publicadoPor, author))
        g.add((post, RR.enSubreddit, subreddit))
        g.add((post, RR.idiomaDetectado, Literal("en")))
        g.add((post, RR.longitudTexto, Literal(100, datatype=XSD.integer)))
        g.add((post, RR.etiquetaSentimiento, RRSENT.Positivo))
        g.serialize(destination=rdf_path, format="turtle")
        return {"extra_post_id": "e1extra001", "extra_post_iri": str(post)}

    if case_id == "E1-U1-duplicate-trace-key":
        trace = pd.read_csv(trace_path, keep_default_na=False)
        row = first_trace_row(package, selected_only=True)
        duplicate = trace[(trace["post_id"].astype(str) == row["post_id"]) & (trace["feature"].astype(str) == row["feature"])].iloc[[0]]
        pd.concat([trace, duplicate], ignore_index=True).to_csv(trace_path, index=False)
        return {"duplicated_trace": row}

    if case_id == "E1-U2-duplicate-split-id":
        split = load_json(split_path)
        duplicate_id = split["train"][0]
        displaced_id = split["train"][1]
        split["train"][1] = duplicate_id
        write_json(split_path, split)
        return {"split": "train", "duplicated_id": duplicate_id, "displaced_id": displaced_id}

    if case_id == "E1-U3-duplicate-prediction-id":
        path = prediction_path(package)
        pred = pd.read_csv(path, dtype={"id": str})
        duplicated_id = pred.loc[0, "id"]
        displaced_id = pred.loc[1, "id"]
        pred.loc[1, "id"] = duplicated_id
        pred.to_csv(path, index=False)
        return {"prediction_file": SELECTED_RUN, "duplicated_id": duplicated_id, "displaced_id": displaced_id}

    if case_id == "E1-R1-swap-selected-matrix-rows":
        path = package / "results" / "prepared_model_inputs" / "test_raw_float64.npy"
        arr = np.load(path)
        arr[[0, 1]] = arr[[1, 0]]
        np.save(path, arr)
        ids = load_json(package / "results" / "prepared_model_inputs" / "test_ids.json")
        return {"matrix": path.name, "swapped_rows": [0, 1], "post_ids": ids[:2]}

    if case_id == "E1-R2-reorder-predictions-against-contract":
        path = prediction_path(package)
        pred = pd.read_csv(path, dtype={"id": str})
        pred = pred.iloc[::-1].reset_index(drop=True)
        pred.to_csv(path, index=False)
        return {"prediction_file": SELECTED_RUN, "mutation": "rows reversed"}

    if case_id == "E1-COL1-swap-matrix-columns":
        path = package / "results" / "prepared_model_inputs" / "test_raw_float64.npy"
        arr = np.load(path)
        arr[:, [0, 1]] = arr[:, [1, 0]]
        np.save(path, arr)
        return {"matrix": path.name, "swapped_columns": [0, 1], "features": selected_features(package)[:2]}

    if case_id == "E1-COL2-unknown-selected-feature":
        payload = load_json(features_path)
        old = payload["selected_features_train_only"][0]
        payload["selected_features_train_only"][0] = "ont_E1_Injected_Unknown"
        write_json(features_path, payload)
        return {"replaced_feature": old, "injected_feature": "ont_E1_Injected_Unknown"}

    if case_id == "E1-V1-corrupt-selected-gold-cell":
        row = first_trace_row(package, selected_only=True)
        gold = pd.read_parquet(gold_path)
        idx = gold[gold["id"].astype(str) == row["post_id"]].index[0]
        old = float(gold.loc[idx, row["feature"]])
        gold.loc[idx, row["feature"]] = old + 0.125
        gold.to_parquet(gold_path, index=False)
        return {"post_id": row["post_id"], "feature": row["feature"], "old": old, "new": old + 0.125}

    if case_id == "E1-V2-corrupt-trace-value":
        trace = pd.read_csv(trace_path, keep_default_na=False)
        row = first_trace_row(package, selected_only=True)
        mask = (trace["post_id"].astype(str) == row["post_id"]) & (trace["feature"].astype(str) == row["feature"])
        old = float(trace.loc[mask, "value"].iloc[0])
        trace.loc[mask, "value"] = old + 0.125
        trace.to_csv(trace_path, index=False)
        return {"post_id": row["post_id"], "feature": row["feature"], "old": old, "new": old + 0.125}

    if case_id == "E1-V3-corrupt-standardized-matrix-cell":
        path = package / "results" / "prepared_model_inputs" / "test_train_standardized_float64.npy"
        arr = np.load(path)
        old = float(arr[0, 0])
        arr[0, 0] = old + 0.125
        np.save(path, arr)
        ids = load_json(package / "results" / "prepared_model_inputs" / "test_ids.json")
        return {"matrix": path.name, "row": 0, "feature": selected_features(package)[0], "post_id": ids[0], "old": old, "new": old + 0.125}

    if case_id == "E1-P1-move-id-between-splits":
        split = load_json(split_path)
        train_id = split["train"][0]
        test_id = split["test"][0]
        split["train"][0], split["test"][0] = test_id, train_id
        write_json(split_path, split)
        return {"train_to_test": train_id, "test_to_train": test_id}

    if case_id == "E1-P2-nontest-prediction-row":
        path = prediction_path(package)
        pred = pd.read_csv(path, dtype={"id": str})
        post_id = pred.loc[0, "id"]
        pred.loc[0, "split"] = "train"
        pred.to_csv(path, index=False)
        return {"prediction_file": SELECTED_RUN, "post_id": post_id, "new_split": "train"}

    if case_id == "E1-RDF1-break-trace-post-iri":
        trace = pd.read_csv(trace_path, keep_default_na=False)
        row = first_trace_row(package, selected_only=True)
        mask = (trace["post_id"].astype(str) == row["post_id"]) & (trace["feature"].astype(str) == row["feature"])
        old = str(trace.loc[mask, "post_iri"].iloc[0])
        new = old + "_broken"
        trace.loc[mask, "post_iri"] = new
        trace.to_csv(trace_path, index=False)
        return {"post_id": row["post_id"], "feature": row["feature"], "old_iri": old, "new_iri": new}

    if case_id == "E1-RDF2-remove-trataSobre-for-active-concept":
        row = first_trace_row(package, selected_only=False)
        g = Graph()
        g.parse(rdf_path, format="turtle")
        post = URIRef(row["post_iri"])
        triples = list(g.triples((post, RR.trataSobre, None)))
        if not triples:
            raise RuntimeError("No rr:trataSobre triples available for RDF2")
        g.remove(triples[0])
        g.serialize(destination=rdf_path, format="turtle")
        return {"post_id": row["post_id"], "removed_triple": [str(x) for x in triples[0]]}

    if case_id in {"E1-RDF3-invalid-domain-concept-iri", "E1-SR2-stale-shacl-success-report"}:
        g = Graph()
        g.parse(rdf_path, format="turtle")
        triple = next(g.triples((None, RR.trataSobre, None)))
        invalid = URIRef(f"{RRDOM}E1InvalidConcept")
        g.remove(triple)
        g.add((triple[0], triple[1], invalid))
        g.serialize(destination=rdf_path, format="turtle")
        return {"post_iri": str(triple[0]), "old_concept": str(triple[2]), "new_concept": str(invalid)}

    if case_id == "E1-PR1-wrong-y-true":
        path = prediction_path(package)
        pred = pd.read_csv(path, dtype={"id": str})
        old = int(pred.loc[0, "y_true"])
        pred.loc[0, "y_true"] = (old + 1) % 3
        pred.to_csv(path, index=False)
        return {"prediction_file": SELECTED_RUN, "post_id": pred.loc[0, "id"], "old_y_true": old, "new_y_true": int(pred.loc[0, "y_true"])}

    if case_id == "E1-PR2-permute-y-pred-only":
        path = prediction_path(package)
        pred = pd.read_csv(path, dtype={"id": str})
        old_hash = sequence_sha256(pred["y_pred"].tolist())
        pred["y_pred"] = pred["y_pred"].iloc[::-1].to_numpy()
        pred.to_csv(path, index=False)
        return {"prediction_file": SELECTED_RUN, "old_y_pred_order_sha256": old_hash, "new_y_pred_order_sha256": sequence_sha256(pred["y_pred"].tolist())}

    raise NotImplementedError(case_id)


def matrix_diagnostics(package: Path) -> dict[str, Any]:
    d = package / "results" / "prepared_model_inputs"
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        return {"available": False, "detected": False, "reason": "manifest missing"}
    manifest = load_json(manifest_path)
    features = manifest["features"]
    detected = False
    mismatches: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        ids = load_json(d / f"{split}_ids.json")
        profiles = {
            "raw_float64": np.load(d / f"{split}_raw_float64.npy"),
            "train_standardized_float64": np.load(d / f"{split}_train_standardized_float64.npy"),
            "train_standardized_float32": np.load(d / f"{split}_train_standardized_float32.npy"),
        }
        for profile, arr in profiles.items():
            expected = manifest["profiles"][split][profile]["matrix_sha256"]
            actual = matrix_sha256(arr)
            if actual != expected:
                detected = True
                mismatches.append(
                    {
                        "split": split,
                        "profile": profile,
                        "expected_matrix_sha256": expected,
                        "actual_matrix_sha256": actual,
                        "rows": int(arr.shape[0]),
                        "columns": int(arr.shape[1]),
                    }
                )
    # Value-level diagnostic against Gold/scaler for prepared matrices.
    try:
        gold = pd.read_parquet(package / "data" / "gold" / "gold_enriched_ontology.parquet").set_index("id")
        scaler: StandardScaler = joblib.load(d / "train_only_scaler.joblib")
        for split in ["train", "val", "test"]:
            ids = load_json(d / f"{split}_ids.json")
            raw = np.load(d / f"{split}_raw_float64.npy")
            expected_raw = gold.loc[ids, features].to_numpy(float)
            if not np.array_equal(raw, expected_raw):
                detected = True
                loc = first_array_difference(raw, expected_raw)
                mismatches.append({"split": split, "profile": "raw_float64", "value_mismatch_at": loc})
            std = np.load(d / f"{split}_train_standardized_float64.npy")
            expected_std = scaler.transform(expected_raw)
            if not np.array_equal(std, expected_std):
                detected = True
                loc = first_array_difference(std, expected_std)
                mismatches.append({"split": split, "profile": "train_standardized_float64", "value_mismatch_at": loc})
            std32 = np.load(d / f"{split}_train_standardized_float32.npy")
            if not np.array_equal(std32, expected_std.astype(np.float32)):
                detected = True
                loc = first_array_difference(std32, expected_std.astype(np.float32))
                mismatches.append({"split": split, "profile": "train_standardized_float32", "value_mismatch_at": loc})
    except Exception as exc:  # noqa: BLE001
        detected = True
        mismatches.append({"diagnostic_error": str(exc)})
    return {"available": True, "detected": detected, "mismatches": mismatches}


def first_array_difference(a: np.ndarray, b: np.ndarray) -> dict[str, Any] | None:
    diff = np.argwhere(a != b)
    if diff.size == 0:
        return None
    row, col = diff[0].tolist()
    return {"row": int(row), "column": int(col), "actual": repr(float(a[row, col])), "expected": repr(float(b[row, col]))}


def rdf_diagnostics(package: Path) -> dict[str, Any]:
    rdf_path = package / "results" / "knowledge_graph" / "posts.ttl"
    gold = pd.read_parquet(package / "data" / "gold" / "gold_enriched_ontology.parquet")
    features = [c for c in gold.columns if c.startswith("ont_")]
    g = Graph()
    try:
        g.parse(rdf_path, format="turtle")
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "detected": True, "parse_error": str(exc)}
    ontology_graph = Graph()
    for name in ["rr-core.ttl", "rr-domain.ttl", "rr-sentiment.ttl"]:
        ontology_graph.parse(package / "src" / "ontology" / "resources" / name, format="turtle")
    gold_ids = set(gold["id"].astype(str))
    rdf_ids: set[str] = set()
    iri_by_id: dict[str, str] = {}
    for post in g.subjects(RDF.type, RR.Post):
        vals = [str(v) for v in g.objects(post, RR.identificador)]
        for value in vals:
            rdf_ids.add(value)
            iri_by_id[value] = str(post)
    missing_ids = sorted(gold_ids - rdf_ids)
    extra_ids = sorted(rdf_ids - gold_ids)
    iri_mismatches = [
        {"post_id": post_id, "expected": stable_post_iri(post_id), "actual": iri_by_id.get(post_id)}
        for post_id in sorted(gold_ids & rdf_ids)
        if iri_by_id.get(post_id) != stable_post_iri(post_id)
    ]
    expected_concepts: set[tuple[str, str]] = set()
    for _, row in gold.iterrows():
        post_id = str(row["id"])
        for feature in features:
            concept = concept_for_feature(feature)
            if concept and pd.notna(row[feature]) and float(row[feature]) != 0.0:
                expected_concepts.add((post_id, str(RRDOM[concept])))
    actual_concepts: set[tuple[str, str]] = set()
    invalid_concepts: list[dict[str, str]] = []
    valid_concepts = {str(x) for x in ontology_graph.subjects(RDF.type, SKOS.Concept)}
    for post, _, concept in g.triples((None, RR.trataSobre, None)):
        ids = [str(v) for v in g.objects(post, RR.identificador)]
        if ids:
            actual_concepts.add((ids[0], str(concept)))
        if str(concept) not in valid_concepts:
            invalid_concepts.append({"post_iri": str(post), "concept_uri": str(concept)})
    missing_concepts = sorted(expected_concepts - actual_concepts)[:20]
    extra_concepts = sorted(actual_concepts - expected_concepts)[:20]
    detected = bool(missing_ids or extra_ids or iri_mismatches or missing_concepts or invalid_concepts)
    return {
        "available": True,
        "detected": detected,
        "rdf_post_count": len(rdf_ids),
        "missing_ids": missing_ids[:20],
        "extra_ids": extra_ids[:20],
        "iri_mismatches": iri_mismatches[:20],
        "missing_concepts": [{"post_id": a, "concept_uri": b} for a, b in missing_concepts],
        "extra_concepts": [{"post_id": a, "concept_uri": b} for a, b in extra_concepts],
        "invalid_concepts": invalid_concepts[:20],
    }


def prediction_diagnostics(package: Path, baseline: Path) -> dict[str, Any]:
    pred_dir = package / "results" / "anova_revalidation" / "predictions"
    base_dir = baseline / "results" / "anova_revalidation" / "predictions"
    detected = False
    mismatches: list[dict[str, Any]] = []
    for path in sorted(pred_dir.glob("*_test_predictions_v2.csv")):
        base_path = base_dir / path.name
        if not base_path.exists():
            detected = True
            mismatches.append({"prediction_file": path.name, "reason": "missing baseline"})
            continue
        cur = pd.read_csv(path, dtype={"id": str})
        base = pd.read_csv(base_path, dtype={"id": str})
        cols = ["id", "split", "y_true", "y_pred", "model", "condition", "seed"]
        for col in cols:
            if sequence_sha256(cur[col].tolist()) != sequence_sha256(base[col].tolist()):
                detected = True
                mismatches.append(
                    {
                        "prediction_file": path.name,
                        "column": col,
                        "baseline_sha256": sequence_sha256(base[col].tolist()),
                        "current_sha256": sequence_sha256(cur[col].tolist()),
                    }
                )
    return {"available": True, "detected": detected, "mismatches": mismatches}


def report_freshness_diagnostics(package: Path, baseline_hashes: dict[str, str]) -> dict[str, Any]:
    pairs = [
        ("results/traceability/traceability_map.csv", "results/traceability/traceability_audit_report.json"),
        ("results/knowledge_graph/posts.ttl", "results/validation/shacl/baseline_report.json"),
    ]
    stale: list[dict[str, str]] = []
    for input_rel, report_rel in pairs:
        input_path = package / input_rel
        report_path = package / report_rel
        if not input_path.exists() or not report_path.exists():
            continue
        input_changed = sha256(input_path) != baseline_hashes.get(input_rel)
        report_unchanged = sha256(report_path) == baseline_hashes.get(report_rel)
        if input_changed and report_unchanged:
            stale.append({"input": input_rel, "report": report_rel})
    return {"available": True, "detected": bool(stale), "stale_pairs": stale}


def extension_diagnostics(package: Path, baseline: Path, baseline_hashes: dict[str, str], case: dict[str, Any]) -> dict[str, Any]:
    if case["family"] == "stale_report":
        freshness = report_freshness_diagnostics(package, baseline_hashes)
    else:
        freshness = {
            "available": False,
            "detected": False,
            "reason": "Report freshness is evaluated only for stale-report cases to avoid treating valid order-only changes as stale evidence.",
        }
    checks = {
        "matrix_manifest_rechecker": matrix_diagnostics(package),
        "rdf_inventory_against_gold": rdf_diagnostics(package),
        "prediction_output_fingerprint_check": prediction_diagnostics(package, baseline),
        "report_freshness_check": freshness,
    }
    return {"detected": any(v.get("detected") for v in checks.values()), "checks": checks}


def classify(case: dict[str, Any], runs: list[DetectorRun], ext: dict[str, Any]) -> dict[str, Any]:
    should_alarm = bool(case["oracle_expected"]["should_alarm"])
    expected_exception_detected = any(expected_exception_matches(case["case_id"], run) for run in runs)
    original_detected = any(r.detected for r in runs) or expected_exception_detected
    unexpected_errors = [
        {
            "phase": r.phase,
            "script": r.script,
            "returncode": r.returncode,
            "stderr_head": r.stderr[:500],
        }
        for r in runs
        if r.operational_error and not expected_exception_matches(case["case_id"], r)
    ]
    extension_detected = bool(ext.get("detected"))
    if unexpected_errors and not original_detected:
        status = "evaluation_error"
    elif not should_alarm and not original_detected and not extension_detected:
        status = "control_passed"
    elif not should_alarm and (original_detected or extension_detected):
        status = "false_positive"
    elif should_alarm and original_detected:
        status = "detected_by_original"
    elif should_alarm and extension_detected:
        status = "detected_by_extension"
    else:
        status = "not_detected"
    return {
        "original_detected": original_detected,
        "extension_detected": extension_detected,
        "detection_status": status,
        "expected_exception_detected": expected_exception_detected,
        "evaluation_error": bool(unexpected_errors),
        "unexpected_errors": unexpected_errors,
        "false_positive": (not should_alarm) and (original_detected or extension_detected),
        "non_detection": should_alarm and not original_detected and not extension_detected,
        "partial_detection": should_alarm and (not original_detected) and extension_detected,
    }


def expected_exception_matches(case_id: str, run: DetectorRun) -> bool:
    text = f"{run.stdout}\n{run.stderr}"
    if case_id == "E1-U2-duplicate-split-id":
        return run.script == "audit_model_input_cells.py" and "Duplicate identifiers in the train partition" in text
    return False


def refresh_scripts_for(case: dict[str, Any]) -> tuple[list[str], list[str]]:
    case_id = case["case_id"]
    family = case["family"]
    scripts: list[str] = []
    skipped: list[str] = []

    if family == "valid_control":
        if case_id in {"E1-C0-pristine-copy", "E1-C3-regenerated-reports-unchanged-inputs"}:
            scripts = [
                "audit_traceability.py",
                "audit_model_inputs.py",
                "audit_model_input_cells.py",
                "run_shacl_conformance.py",
                "reconcile_artifact_chain.py",
            ]
        elif case_id == "E1-C1-trace-row-permutation":
            scripts = [
                "audit_traceability.py",
                "audit_model_inputs.py",
                "audit_model_input_cells.py",
                "reconcile_artifact_chain.py",
            ]
            skipped = ["run_shacl_conformance.py inherited pristine SHACL report; RDF/shapes unchanged"]
        elif case_id == "E1-C2-rdf-triple-order-reserialization":
            scripts = ["run_shacl_conformance.py", "reconcile_artifact_chain.py"]
            skipped = ["trace/model-input audits inherited; Gold/trace/splits/predictions unchanged"]
        return scripts, skipped

    if family in {"identity", "rdf_link"} and case_id in {
        "E1-I2-remove-rdf-post-resource",
        "E1-I3-add-extra-valid-rdf-post",
        "E1-RDF2-remove-trataSobre-for-active-concept",
        "E1-RDF3-invalid-domain-concept-iri",
    }:
        scripts = ["run_shacl_conformance.py", "reconcile_artifact_chain.py"]
        skipped = ["trace/model-input audits inherited; mutation targets RDF graph only"]
        return scripts, skipped

    if case_id == "E1-SR2-stale-shacl-success-report":
        scripts = ["run_shacl_conformance.py", "reconcile_artifact_chain.py"]
        skipped = ["trace/model-input audits inherited; stale object is SHACL report"]
        return scripts, skipped

    if family == "column_alignment" and case_id == "E1-COL2-unknown-selected-feature":
        return ["audit_model_inputs.py", "audit_model_input_cells.py", "reconcile_artifact_chain.py"], [
            "SHACL inherited; RDF graph unchanged"
        ]

    scripts = [
        "audit_traceability.py",
        "audit_model_inputs.py",
        "audit_model_input_cells.py",
        "reconcile_artifact_chain.py",
    ]
    skipped = ["run_shacl_conformance.py inherited pristine SHACL report; RDF/shapes unchanged"]
    return scripts, skipped


def execute_case(case: dict[str, Any], template: Path, baseline_hashes: dict[str, str]) -> dict[str, Any]:
    case_id = case["case_id"]
    case_dir = RUN_ROOT / "cases" / case_id
    safe_reset(case_dir)
    package = case_dir / "package"
    shutil.copytree(template, package)
    before = file_hashes(package)
    mutation = mutation_details(case_id, package)
    after_mutation = file_hashes(package)
    write_json(case_dir / "mutation_manifest.json", {"case_id": case_id, "case": case, "mutation_details": mutation})
    write_json(case_dir / "before_hashes.json", before)
    write_json(case_dir / "after_mutation_hashes.json", after_mutation)

    runs: list[DetectorRun] = []
    # Preserve stale-consumer behavior before refreshing producers.
    runs.append(run_script(package, "reconcile_artifact_chain.py", case_dir / "logs", "original_consumer_before_refresh"))

    refresh_scripts, skipped_scripts = refresh_scripts_for(case)
    for script in refresh_scripts:
        runs.append(run_script(package, script, case_dir / "logs", "original_after_refresh"))

    ext = extension_diagnostics(package, template, baseline_hashes, case)
    run_payloads = [r.__dict__ for r in runs]
    write_json(case_dir / "original_detector_result.json", {"runs": run_payloads, "skipped_after_refresh": skipped_scripts})
    write_json(case_dir / "extension_detector_result.json", ext)

    classification = classify(case, runs, ext)
    final_hashes = file_hashes(package)
    write_json(case_dir / "after_detector_hashes.json", final_hashes)
    result = {
        "case_id": case_id,
        "family": case["family"],
        "required": case.get("required", False),
        "reserved_for_after_extension": case.get("reserved_for_after_extension", False),
        "oracle_should_alarm": case["oracle_expected"]["should_alarm"],
        **classification,
        "original_detected_before_refresh": runs[0].detected,
        "original_detected_after_refresh": any(r.detected for r in runs[1:]),
        "extension_check_ids": ",".join(k for k, v in ext["checks"].items() if v.get("detected")),
        "mutation_summary": json.dumps(mutation, sort_keys=True, ensure_ascii=True)[:500],
        "notes": case.get("acceptance", ""),
    }
    write_json(case_dir / "oracle_comparison.json", result)
    return result


def report_markdown(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    controls = [r for r in rows if r["family"] == "valid_control"]
    faults = [r for r in rows if r["family"] != "valid_control"]
    eval_errors = [r for r in rows if r.get("evaluation_error")]
    by_status: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    for row in rows:
        by_status[row["detection_status"]] = by_status.get(row["detection_status"], 0) + 1
        fam = row["family"]
        by_family.setdefault(fam, {"cases": 0, "original": 0, "extension": 0, "miss": 0, "false_positive": 0})
        by_family[fam]["cases"] += 1
        by_family[fam]["original"] += int(row["original_detected"])
        by_family[fam]["extension"] += int(row["extension_detected"])
        by_family[fam]["miss"] += int(row["non_detection"])
        by_family[fam]["false_positive"] += int(row["false_positive"])

    lines = [
        "# E1 Continuity Fault Injection Results",
        "",
        "Status: executed continuity controls and non-SHACL fault cases from `E1_CASES.json`.",
        "",
        "Scope: this run excludes the independent SHACL fixture suite. It uses isolated per-case copies and does not train models or mutate the freeze, campaign, public release, root protocol files, or `.tex` files.",
        "",
        "## Denominators",
        "",
        f"- Total executed cases: {total}",
        f"- Valid controls: {len(controls)}",
        f"- Continuity fault cases: {len(faults)}",
        "- Independent SHACL fixture cases: 0 in this run; delegated separately.",
        f"- Cases with an additional operational error after a contract signal: {len(eval_errors)}",
        "",
        "## Detection Summary",
        "",
        "| Status | Cases |",
        "|---|---:|",
    ]
    for status, count in sorted(by_status.items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## Family Summary", "", "| Family | Cases | Original detected | Extension detected | Misses | False positives |", "|---|---:|---:|---:|---:|---:|"])
    for fam, data in sorted(by_family.items()):
        lines.append(f"| `{fam}` | {data['cases']} | {data['original']} | {data['extension']} | {data['miss']} | {data['false_positive']} |")
    lines.extend(["", "## Case Results", "", "| Case | Family | Oracle alarm | Original before refresh | Original after refresh | Extension | Status |", "|---|---|---:|---:|---:|---:|---|"])
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['family']}` | {row['oracle_should_alarm']} | {row['original_detected_before_refresh']} | {row['original_detected_after_refresh']} | {row['extension_detected']} | `{row['detection_status']}` |"
        )
    if eval_errors:
        lines.extend(["", "## Operational Errors Kept Separate", ""])
        for row in eval_errors:
            lines.append(
                f"- `{row['case_id']}`: detector status remains `{row['detection_status']}` because another original script signaled the contract fault, but an additional script produced an operational error. Inspect the case `original_detector_result.json` before using this case as evidence of clean localization."
            )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- `detected_by_original` means at least one original script returned a nonzero code or wrote an attention/failure status after the mutation was applied.",
            "- `original_detected_before_refresh` preserves the stale-consumer phase. Several cases only become visible after a producer is rerun against the mutated artefact.",
            "- Extension diagnostics are reported separately and should not be described as behavior of the submitted detector.",
            "- Matrix and prediction-output cases expose gaps in the original detector surface when it relies on metadata, IDs, or labels but not the mutated matrix/prediction assignment bytes.",
            "- No result here validates semantic correctness of ontology activations or SHACL as truth.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    global RUN_ROOT, FREEZE_PACKAGE, CAMPAIGN_PACKAGE, DETECTOR_SCRIPTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=ROOT)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--detector-scripts", type=Path, default=DETECTOR_SCRIPTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("case_ids", nargs="*")
    args = parser.parse_args()
    FREEZE_PACKAGE = args.baseline_root.resolve()
    CAMPAIGN_PACKAGE = (args.campaign_root or FREEZE_PACKAGE).resolve()
    DETECTOR_SCRIPTS = args.detector_scripts.resolve()
    RUN_ROOT = scratch_root(args.output_dir, FREEZE_PACKAGE)
    for protected in (CAMPAIGN_PACKAGE, DETECTOR_SCRIPTS):
        if RUN_ROOT == protected or RUN_ROOT in protected.parents or protected in RUN_ROOT.parents:
            raise ValueError("Scratch overlaps a source directory")
    spec = load_json(CASES_JSON)
    cases = spec["controls"] + spec["fault_cases"]
    if len(cases) != 26:
        raise RuntimeError(f"Expected 26 continuity cases, got {len(cases)}")
    summarize_only = False
    selected_case_ids = set(args.case_ids)
    cases_to_execute = [case for case in cases if not selected_case_ids or case["case_id"] in selected_case_ids]
    if selected_case_ids and len(cases_to_execute) != len(selected_case_ids):
        known = {case["case_id"] for case in cases}
        raise RuntimeError(f"Unknown case ids: {sorted(selected_case_ids - known)}")
    if not summarize_only:
        template = prepare_template()
        baseline_hashes = load_json(RUN_ROOT / "baseline_hashes.json")
        write_json(
            RUN_ROOT / "E1_run_manifest.json",
            {
                "status": "running",
                "scope": "continuity_controls_and_fault_cases_only",
                "case_count": len(cases),
                "executed_this_invocation": [case["case_id"] for case in cases_to_execute],
                "excluded_shacl_fixture_cases": len(spec.get("shacl_cases", [])),
                "python": str(PYTHON if PYTHON.exists() else sys.executable),
                "freeze_package": str(FREEZE_PACKAGE),
                "campaign_package": str(CAMPAIGN_PACKAGE),
                "template_package": str(template),
                "script_names": SCRIPT_NAMES,
            },
        )
        for case in cases_to_execute:
            result = execute_case(case, template, baseline_hashes)
            print(f"{case['case_id']}: {result['detection_status']}", flush=True)

    rows = []
    for case in cases:
        result_path = RUN_ROOT / "cases" / case["case_id"] / "oracle_comparison.json"
        if result_path.exists():
            rows.append(load_json(result_path))
        else:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "required": case.get("required", False),
                    "reserved_for_after_extension": case.get("reserved_for_after_extension", False),
                    "oracle_should_alarm": case["oracle_expected"]["should_alarm"],
                    "original_detected": False,
                    "extension_detected": False,
                    "detection_status": "not_executed",
                    "expected_exception_detected": False,
                    "evaluation_error": False,
                    "unexpected_errors": [],
                    "false_positive": False,
                    "non_detection": bool(case["oracle_expected"]["should_alarm"]),
                    "partial_detection": False,
                    "original_detected_before_refresh": False,
                    "original_detected_after_refresh": False,
                    "extension_check_ids": "",
                    "mutation_summary": "",
                    "notes": "Case not executed in current run directory.",
                }
            )
    write_csv(RUN_ROOT / "E1_RESULTS.csv", rows)
    write_json(RUN_ROOT / "E1_RESULTS.json", rows)
    (RUN_ROOT / "E1_REPORT.md").write_text(report_markdown(rows), encoding="utf-8")
    manifest_path = RUN_ROOT / "E1_run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    manifest["status"] = "executed"
    manifest["aggregate_includes_existing_case_results"] = True
    manifest["cases_with_existing_results"] = sum(
        (RUN_ROOT / "cases" / case["case_id"] / "oracle_comparison.json").exists() for case in cases
    )
    manifest["results"] = {
        "total": len(rows),
        "controls": sum(r["family"] == "valid_control" for r in rows),
        "faults": sum(r["family"] != "valid_control" for r in rows),
        "not_detected": sum(r["non_detection"] for r in rows),
        "false_positives": sum(r["false_positive"] for r in rows),
    }
    write_json(RUN_ROOT / "E1_run_manifest.json", manifest)


if __name__ == "__main__":
    main()
