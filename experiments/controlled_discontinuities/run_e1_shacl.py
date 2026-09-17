#!/usr/bin/env python3
"""Execute E1 SHACL controlled discontinuities.

This harness uses the F1+F2+F3 RDF graph and declared SHACL shapes as immutable
inputs, applies one controlled mutation per case, and records the pySHACL report
without treating SHACL conformance as semantic truth.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pyshacl
import rdflib
from pyshacl import validate
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, SH, SKOS, XSD


from replay_support import PACKAGE, scratch_root, safe_reset
import reclassify_shacl_reports as reclassification

RUN_DIR = None
REPO_ROOT = PACKAGE.parents[1]
PROTOCOL_DIR = PACKAGE
BASELINE = REPO_ROOT

POSTS = BASELINE / "results" / "knowledge_graph" / "posts.ttl"
SHAPES = BASELINE / "src" / "ontology" / "resources" / "rr-shapes.ttl"
ONTOLOGY_FILES = [
    BASELINE / "src" / "ontology" / "resources" / "rr-core.ttl",
    BASELINE / "src" / "ontology" / "resources" / "rr-domain.ttl",
    BASELINE / "src" / "ontology" / "resources" / "rr-sentiment.ttl",
]
CASES_JSON = PROTOCOL_DIR / "E1_CASES.json"

RR = Namespace("http://wfrp.ia/ontologia/ia-sentimiento#")
RRSENT = Namespace("http://wfrp.ia/sentimiento/core#")
RRDOM = Namespace("http://wfrp.ia/dominio/ciencia-tecnologia#")
BASE = Namespace("http://wfrp.ia/resource/reddit/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return os.path.relpath(path, RUN_DIR)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_case_specs() -> dict[str, dict[str, Any]]:
    with CASES_JSON.open(encoding="utf-8") as f:
        payload = json.load(f)
    return {case["case_id"]: case for case in payload["shacl_cases"]}


def ontology_graph() -> Graph:
    g = Graph()
    for path in ONTOLOGY_FILES:
        g.parse(path, format="turtle")
    return g


def normalized_shapes(case_dir: Path) -> Path:
    text = SHAPES.read_text(encoding="utf-8")
    text = text.replace("https://www.w3.org/ns/shacl#", "http://www.w3.org/ns/shacl#")
    out = case_dir / "canonical_shapes_normalized.ttl"
    out.write_text(text, encoding="utf-8")
    return out


def base_graph() -> Graph:
    g = Graph()
    g.parse(POSTS, format="turtle")
    return g


def ordered_posts(g: Graph) -> list[URIRef]:
    return sorted(set(g.subjects(RDF.type, RR.Post)), key=str)


def positive_post(g: Graph) -> URIRef:
    candidates = sorted(set(g.subjects(RR.etiquetaSentimiento, RRSENT.Positivo)), key=str)
    if not candidates:
        raise RuntimeError("No rr:Post with rrsent:Positivo label found.")
    return candidates[0]


def post_with_topic(g: Graph) -> URIRef:
    candidates = sorted(set(g.subjects(RR.trataSobre, None)), key=str)
    if not candidates:
        raise RuntimeError("No rr:Post with rr:trataSobre found.")
    return candidates[0]


def clone_required_post_facts(g: Graph, source: URIRef, target: URIRef) -> dict[str, str]:
    copied: dict[str, str] = {}
    g.add((target, RDF.type, RR.Post))
    for predicate in [
        RR.identificador,
        RR.fechaCreacion,
        RR.publicadoPor,
        RR.enSubreddit,
        RR.idiomaDetectado,
        RR.longitudTexto,
        RR.etiquetaSentimiento,
    ]:
        value = next(g.objects(source, predicate))
        g.add((target, predicate, value))
        copied[str(predicate)] = str(value)
    return copied


def mutate_missing_identifier(g: Graph) -> dict[str, Any]:
    post = ordered_posts(g)[0]
    removed = [str(o) for o in g.objects(post, RR.identificador)]
    g.remove((post, RR.identificador, None))
    return {"focus_node": str(post), "removed_values": removed}


def mutate_invalid_identifier_pattern(g: Graph) -> dict[str, Any]:
    post = ordered_posts(g)[0]
    old = [str(o) for o in g.objects(post, RR.identificador)]
    g.remove((post, RR.identificador, None))
    g.add((post, RR.identificador, Literal("BAD-ID!", datatype=XSD.string)))
    return {"focus_node": str(post), "old_values": old, "new_value": "BAD-ID!"}


def mutate_invalid_language(g: Graph) -> dict[str, Any]:
    post = ordered_posts(g)[0]
    old = [str(o) for o in g.objects(post, RR.idiomaDetectado)]
    g.remove((post, RR.idiomaDetectado, None))
    g.add((post, RR.idiomaDetectado, Literal("fr", datatype=XSD.string)))
    return {"focus_node": str(post), "old_values": old, "new_value": "fr"}


def mutate_missing_author(g: Graph) -> dict[str, Any]:
    post = ordered_posts(g)[0]
    removed = [str(o) for o in g.objects(post, RR.publicadoPor)]
    g.remove((post, RR.publicadoPor, None))
    return {"focus_node": str(post), "removed_values": removed}


def mutate_probability_out_of_range(g: Graph) -> dict[str, Any]:
    post = ordered_posts(g)[0]
    g.remove((post, RR.probPos, None))
    g.add((post, RR.probPos, Literal("1.5", datatype=XSD.decimal)))
    return {"focus_node": str(post), "property": str(RR.probPos), "new_value": "1.5"}


def mutate_probability_sum(g: Graph) -> dict[str, Any]:
    post = positive_post(g)
    for predicate in [RR.probPos, RR.probNeu, RR.probNeg]:
        g.remove((post, predicate, None))
    values = {RR.probPos: "0.8", RR.probNeu: "0.8", RR.probNeg: "0.2"}
    for predicate, value in values.items():
        g.add((post, predicate, Literal(value, datatype=XSD.decimal)))
    return {"focus_node": str(post), "probabilities": {str(k): v for k, v in values.items()}}


def mutate_label_probability_inconsistency(g: Graph) -> dict[str, Any]:
    post = positive_post(g)
    for predicate in [RR.probPos, RR.probNeu, RR.probNeg]:
        g.remove((post, predicate, None))
    values = {RR.probPos: "0.1", RR.probNeu: "0.8", RR.probNeg: "0.1"}
    for predicate, value in values.items():
        g.add((post, predicate, Literal(value, datatype=XSD.decimal)))
    return {
        "focus_node": str(post),
        "existing_label": str(next(g.objects(post, RR.etiquetaSentimiento))),
        "probabilities": {str(k): v for k, v in values.items()},
    }


def mutate_invalid_domain_concept(g: Graph) -> dict[str, Any]:
    post = post_with_topic(g)
    old = next(g.objects(post, RR.trataSobre))
    g.remove((post, RR.trataSobre, old))
    invalid = RRDOM.E1InvalidConcept
    g.add((post, RR.trataSobre, invalid))
    return {"focus_node": str(post), "old_concept": str(old), "new_concept": str(invalid)}


def mutate_duplicate_identifier(g: Graph) -> dict[str, Any]:
    source = ordered_posts(g)[0]
    duplicate = URIRef(f"{BASE}post/e1_duplicate_identifier_fixture")
    copied = clone_required_post_facts(g, source, duplicate)
    return {
        "source_focus_node": str(source),
        "duplicate_focus_node": str(duplicate),
        "duplicated_identifier": copied[str(RR.identificador)],
    }


def mutate_concept_without_scheme(g: Graph) -> dict[str, Any]:
    concept = RRDOM.E1ConceptWithoutScheme
    g.add((concept, RDF.type, SKOS.Concept))
    return {"focus_node": str(concept), "omitted_property": str(SKOS.inScheme)}


MUTATORS: dict[str, Callable[[Graph], dict[str, Any] | None]] = {
    "E1-C0-pristine-shacl-control": lambda g: None,
    "E1-SHACL1-missing-identifier": mutate_missing_identifier,
    "E1-SHACL2-invalid-identifier-pattern": mutate_invalid_identifier_pattern,
    "E1-SHACL3-invalid-language": mutate_invalid_language,
    "E1-SHACL4-missing-author": mutate_missing_author,
    "E1-SHACL5-probability-out-of-range": mutate_probability_out_of_range,
    "E1-SHACL6-probability-sum": mutate_probability_sum,
    "E1-SHACL7-label-probability-inconsistency": mutate_label_probability_inconsistency,
    "E1-SHACL8-invalid-domain-concept": mutate_invalid_domain_concept,
    "E1-SHACL9-duplicate-rdf-identifier": mutate_duplicate_identifier,
    "E1-SHACL10-concept-without-scheme": mutate_concept_without_scheme,
}


def result_terms(report_graph: Graph) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in sorted(set(report_graph.subjects(RDF.type, SH.ValidationResult)), key=str):
        rows.append(
            {
                "result_node": str(result),
                "focus_node": [str(o) for o in report_graph.objects(result, SH.focusNode)],
                "result_path": [str(o) for o in report_graph.objects(result, SH.resultPath)],
                "source_constraint_component": [
                    str(o) for o in report_graph.objects(result, SH.sourceConstraintComponent)
                ],
                "result_severity": [str(o) for o in report_graph.objects(result, SH.resultSeverity)],
                "source_shape": [str(o) for o in report_graph.objects(result, SH.sourceShape)],
                "message": [str(o) for o in report_graph.objects(result, SH.resultMessage)],
            }
        )
    return rows


def validate_case(data_graph: Graph, shapes_path: Path, ont_graph: Graph) -> tuple[bool, Graph, str]:
    conforms, report_graph, report_text = validate(
        data_graph,
        shacl_graph=str(shapes_path),
        ont_graph=ont_graph,
        inference="rdfs",
        advanced=True,
    )
    return bool(conforms), report_graph, str(report_text)


def expected_for(case_id: str, specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if case_id == "E1-C0-pristine-shacl-control":
        return {
            "should_alarm": False,
            "conforms": True,
            "expected_path": None,
            "expected_component": None,
        }
    spec = specs[case_id]
    return {
        "should_alarm": spec["oracle_expected"]["should_alarm"],
        "conforms": spec["oracle_expected"]["conforms"],
        "expected_path": spec["expected_path"],
        "expected_component": spec["expected_component"],
        "shape_family": spec["shape_family"],
        "reserved_for_after_extension": spec["reserved_for_after_extension"],
    }


def matches_expected(row: dict[str, Any], expected: dict[str, Any]) -> bool:
    component = expected.get("expected_component")
    path = expected.get("expected_path")
    component_ok = True
    path_ok = True
    if component:
        component_ok = any(component in value for value in row["source_constraint_component"])
    if path and path != "SPARQL":
        path_ok = any(path.split(":")[-1] in value for value in row["result_path"])
    if path == "SPARQL":
        path_ok = len(row["result_path"]) == 0
    return component_ok and path_ok


def run_case(case_id: str, specs: dict[str, dict[str, Any]], ont_graph: Graph) -> dict[str, Any]:
    case_dir = RUN_DIR / "cases" / case_id
    safe_reset(case_dir, RUN_DIR)
    (case_dir / "logs").mkdir(parents=True)

    shapes_path = normalized_shapes(case_dir)
    data_graph = base_graph()
    before_count = len(data_graph)
    mutator = MUTATORS[case_id]
    mutation_details = mutator(data_graph)
    after_count = len(data_graph)

    data_path = case_dir / "mutated_posts.ttl"
    data_graph.serialize(destination=data_path, format="turtle")

    manifest = {
        "case_id": case_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_posts": rel(POSTS),
        "baseline_shapes": rel(SHAPES),
        "normalized_shapes": rel(shapes_path),
        "namespace_normalization": {
            "source_shacl_namespace": "https://www.w3.org/ns/shacl#",
            "runtime_shacl_namespace": "http://www.w3.org/ns/shacl#",
            "reason": "The F123 run_shacl_conformance.py applies this normalization for pySHACL.",
        },
        "mutation_details": mutation_details,
        "triple_count_before_mutation": before_count,
        "triple_count_after_mutation": after_count,
        "input_hashes": {
            "baseline_posts_sha256": sha256_file(POSTS),
            "baseline_shapes_sha256": sha256_file(SHAPES),
            "mutated_posts_sha256": sha256_file(data_path),
            "normalized_shapes_sha256": sha256_file(shapes_path),
            "ontology_sha256": {rel(path): sha256_file(path) for path in ONTOLOGY_FILES},
        },
    }
    write_json(case_dir / "mutation_manifest.json", manifest)

    started = time.time()
    conforms, report_graph, report_text = validate_case(data_graph, shapes_path, ont_graph)
    elapsed = time.time() - started

    report_ttl = case_dir / "report.ttl"
    report_txt = case_dir / "report.txt"
    report_json = case_dir / "report_extracted.json"
    report_graph.serialize(destination=report_ttl, format="turtle")
    report_txt.write_text(report_text, encoding="utf-8")
    extracted = result_terms(report_graph)
    write_json(report_json, extracted)

    expected = expected_for(case_id, specs)
    result_count = len(extracted)
    original_detected = (not conforms) and result_count > 0
    matched = [row for row in extracted if matches_expected(row, expected)]
    if expected["should_alarm"]:
        if original_detected and matched:
            detection_status = "detected_by_original"
            localization_status = "localized_to_expected_constraint"
        elif original_detected:
            detection_status = "partial_detection"
            localization_status = "detected_without_expected_path_component_match"
        else:
            detection_status = "not_detected"
            localization_status = "not_localized"
    else:
        detection_status = "control_passed" if conforms and result_count == 0 else "false_positive"
        localization_status = "not_applicable"

    result = {
        "case_id": case_id,
        "family": "valid_control" if case_id.startswith("E1-C0") else "shacl",
        "oracle_should_alarm": expected["should_alarm"],
        "oracle_conforms_expected": expected["conforms"],
        "pyshacl_conforms": conforms,
        "validation_result_count": result_count,
        "original_detected": original_detected,
        "extension_detected": len(matched) > 0 if expected["should_alarm"] else False,
        "detection_status": detection_status,
        "localization_status": localization_status,
        "false_positive": (not expected["should_alarm"]) and original_detected,
        "non_detection": expected["should_alarm"] and not original_detected,
        "partial_detection": detection_status == "partial_detection",
        "expected": expected,
        "matched_result_count": len(matched),
        "results": extracted,
        "artefacts": {
            "case_dir": rel(case_dir),
            "data_graph": rel(data_path),
            "mutation_manifest": rel(case_dir / "mutation_manifest.json"),
            "report_ttl": rel(report_ttl),
            "report_text": rel(report_txt),
            "report_extracted": rel(report_json),
        },
        "runtime": {
            "elapsed_seconds": round(elapsed, 6),
            "command": f"{sys.executable} {rel(Path(__file__))}",
        },
    }
    write_json(case_dir / "result.json", result)
    (case_dir / "logs" / "stdout.txt").write_text("", encoding="utf-8")
    (case_dir / "logs" / "stderr.txt").write_text("", encoding="utf-8")
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    detected = [r for r in results if r["detection_status"] in {"detected_by_original", "control_passed"}]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(r["detection_status"] in {"detected_by_original", "control_passed"} for r in results) else "attention_required",
        "scope": {
            "executed_control_cases": 1,
            "executed_shacl_fault_cases": 10,
            "not_executed_from_full_e1_inventory": "identity, uniqueness, row alignment, column alignment, value, partition, RDF-link outside SHACL, prediction assignment, and stale-report cases are outside this subrun.",
            "semantic_boundary": "SHACL validation reports conformance to declared shapes only; it does not prove semantic correctness of ontology activations.",
        },
        "counts": {
            "total_cases": len(results),
            "controls": sum(1 for r in results if r["family"] == "valid_control"),
            "faults": sum(1 for r in results if r["family"] == "shacl"),
            "detected_by_original": sum(1 for r in results if r["detection_status"] == "detected_by_original"),
            "control_passed": sum(1 for r in results if r["detection_status"] == "control_passed"),
            "partial_detection": sum(1 for r in results if r["partial_detection"]),
            "not_detected": sum(1 for r in results if r["non_detection"]),
            "false_positive": sum(1 for r in results if r["false_positive"]),
        },
        "baseline_inputs": {
            "posts": rel(POSTS),
            "posts_sha256": sha256_file(POSTS),
            "shapes": rel(SHAPES),
            "shapes_sha256": sha256_file(SHAPES),
            "ontology_sha256": {rel(path): sha256_file(path) for path in ONTOLOGY_FILES},
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "rdflib": rdflib.__version__,
            "pyshacl": pyshacl.__version__,
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        },
        "cases": results,
        "detected_case_ids": [r["case_id"] for r in detected],
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# E1 SHACL Subrun Report",
        "",
        f"Created UTC: {summary['created_at_utc']}",
        "",
        "## Scope",
        "",
        "This subrun executes the intact SHACL control plus the 10 `shacl_cases` declared in `E1_CASES.json`. It does not execute the non-SHACL continuity families; those require separate continuity harnesses.",
        "",
        "SHACL is interpreted only as conformance to the declared shapes in `rr-shapes.ttl`. The run does not validate semantic correctness of extracted ontology activations.",
        "",
        "## Summary Counts",
        "",
        "| Measure | Count |",
        "|---|---:|",
    ]
    for key, value in summary["counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Expected alarm | Conforms | Results | Status | Localization |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in summary["cases"]:
        lines.append(
            f"| {row['case_id']} | {row['oracle_should_alarm']} | {row['pyshacl_conforms']} | "
            f"{row['validation_result_count']} | {row['detection_status']} | {row['localization_status']} |"
        )
    lines.extend(["", "## Constraint Details", ""])
    for row in summary["cases"]:
        lines.append(f"### {row['case_id']}")
        lines.append("")
        if not row["results"]:
            lines.append("- No validation results.")
        for result in row["results"]:
            focus = ", ".join(result["focus_node"]) or "none"
            path = ", ".join(result["result_path"]) or "none"
            comp = ", ".join(result["source_constraint_component"]) or "none"
            sev = ", ".join(result["result_severity"]) or "none"
            msg = " | ".join(result["message"]) or "none"
            lines.append(f"- focus: `{focus}`")
            lines.append(f"  path: `{path}`")
            lines.append(f"  component: `{comp}`")
            lines.append(f"  severity: `{sev}`")
            lines.append(f"  message: {msg}")
        lines.append("")
    lines.extend(
        [
            "## Reproducibility",
            "",
            f"- Script: `{rel(Path(__file__))}`",
            f"- Data graph baseline SHA-256: `{summary['baseline_inputs']['posts_sha256']}`",
            f"- Shapes baseline SHA-256: `{summary['baseline_inputs']['shapes_sha256']}`",
            f"- Python executable: `{summary['environment']['executable']}`",
            f"- rdflib: `{summary['environment']['rdflib']}`",
            f"- pySHACL: `{summary['environment']['pyshacl']}`",
        ]
    )
    (RUN_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global RUN_DIR, BASELINE, POSTS, SHAPES, ONTOLOGY_FILES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    BASELINE = args.baseline_root.resolve()
    RUN_DIR = scratch_root(args.output_dir, BASELINE)
    POSTS = BASELINE / "results/knowledge_graph/posts.ttl"
    SHAPES = BASELINE / "src/ontology/resources/rr-shapes.ttl"
    ONTOLOGY_FILES = [BASELINE / "src/ontology/resources" / name
                      for name in ("rr-core.ttl", "rr-domain.ttl", "rr-sentiment.ttl")]
    specs = read_case_specs()
    ont_graph = ontology_graph()
    (RUN_DIR / "cases").mkdir(parents=True, exist_ok=True)
    case_ids = [
        "E1-C0-pristine-shacl-control",
        "E1-SHACL1-missing-identifier",
        "E1-SHACL2-invalid-identifier-pattern",
        "E1-SHACL3-invalid-language",
        "E1-SHACL4-missing-author",
        "E1-SHACL5-probability-out-of-range",
        "E1-SHACL6-probability-sum",
        "E1-SHACL7-label-probability-inconsistency",
        "E1-SHACL8-invalid-domain-concept",
        "E1-SHACL9-duplicate-rdf-identifier",
        "E1-SHACL10-concept-without-scheme",
    ]
    results = [run_case(case_id, specs, ont_graph) for case_id in case_ids]
    summary = summarize(results)
    write_json(RUN_DIR / "summary.json", summary)
    write_markdown(summary)
    print(json.dumps({"status": summary["status"], "counts": summary["counts"]}, indent=2))
    reclassification.RUN_DIR = RUN_DIR
    reclassification.main()


if __name__ == "__main__":
    main()
