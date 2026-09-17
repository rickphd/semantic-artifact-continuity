#!/usr/bin/env python3
"""Reclassify E1 SHACL reports without rerunning pySHACL.

The original run preserved the pySHACL reports but compared expected QName values
against full IRIs. This script reads the preserved report TTL files and recomputes
only the localization classification.
"""

from __future__ import annotations

import hashlib
import os
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdflib import Graph, Namespace
from rdflib.namespace import RDF, SH


from replay_support import PACKAGE

RUN_DIR = None
REPO_ROOT = PACKAGE.parents[1]
PROTOCOL_DIR = PACKAGE
CASES_JSON = PROTOCOL_DIR / "E1_CASES.json"

PREFIXES = {
    "rr": "http://wfrp.ia/ontologia/ia-sentimiento#",
    "rrsent": "http://wfrp.ia/sentimiento/core#",
    "rrdom": "http://wfrp.ia/dominio/ciencia-tecnologia#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "sh": "http://www.w3.org/ns/shacl#",
}


def rel(path: Path) -> str:
    return os.path.relpath(path, RUN_DIR)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def expand_qname(value: str | None) -> str | None:
    if value is None or value == "SPARQL":
        return value
    if ":" not in value:
        return value
    prefix, local = value.split(":", 1)
    if prefix not in PREFIXES:
        return value
    return PREFIXES[prefix] + local


def read_specs() -> dict[str, dict[str, Any]]:
    with CASES_JSON.open(encoding="utf-8") as f:
        payload = json.load(f)
    return {case["case_id"]: case for case in payload["shacl_cases"]}


def extract_results(report_ttl: Path) -> list[dict[str, Any]]:
    graph = Graph()
    graph.parse(report_ttl, format="turtle")
    rows: list[dict[str, Any]] = []
    for result in sorted(set(graph.subjects(RDF.type, SH.ValidationResult)), key=str):
        rows.append(
            {
                "result_node": str(result),
                "focus_node": [str(o) for o in graph.objects(result, SH.focusNode)],
                "result_path": [str(o) for o in graph.objects(result, SH.resultPath)],
                "source_constraint_component": [
                    str(o) for o in graph.objects(result, SH.sourceConstraintComponent)
                ],
                "result_severity": [str(o) for o in graph.objects(result, SH.resultSeverity)],
                "source_shape": [str(o) for o in graph.objects(result, SH.sourceShape)],
                "message": [str(o) for o in graph.objects(result, SH.resultMessage)],
            }
        )
    return rows


def matches(row: dict[str, Any], expected_path: str | None, expected_component: str | None) -> bool:
    component_iri = expand_qname(expected_component)
    path_iri = expand_qname(expected_path)
    component_ok = True if component_iri is None else component_iri in row["source_constraint_component"]
    if expected_path == "SPARQL":
        path_ok = len(row["result_path"]) == 0
    elif path_iri is None:
        path_ok = True
    else:
        path_ok = path_iri in row["result_path"]
    return component_ok and path_ok


def reclassify_case(case_dir: Path, specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_id = case_dir.name
    old_result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    extracted = extract_results(case_dir / "report.ttl")
    if case_id == "E1-C0-pristine-shacl-control":
        expected_alarm = False
        expected_conforms = True
        expected_path = None
        expected_component = None
        shape_family = None
        reserved = False
    else:
        spec = specs[case_id]
        expected_alarm = bool(spec["oracle_expected"]["should_alarm"])
        expected_conforms = bool(spec["oracle_expected"]["conforms"])
        expected_path = spec["expected_path"]
        expected_component = spec["expected_component"]
        shape_family = spec["shape_family"]
        reserved = bool(spec["reserved_for_after_extension"])

    matched = [r for r in extracted if matches(r, expected_path, expected_component)]
    conforms = bool(old_result["pyshacl_conforms"])
    original_detected = (not conforms) and bool(extracted)
    if expected_alarm:
        if original_detected and matched:
            status = "detected_by_original"
            localization = "localized_to_expected_constraint"
        elif original_detected:
            status = "partial_detection"
            localization = "detected_without_expected_path_component_match"
        else:
            status = "not_detected"
            localization = "not_localized"
    else:
        status = "control_passed" if conforms and not extracted else "false_positive"
        localization = "not_applicable"

    return {
        **old_result,
        "results": extracted,
        "validation_result_count": len(extracted),
        "original_detected": original_detected,
        "extension_detected": bool(matched) if expected_alarm else False,
        "detection_status": status,
        "localization_status": localization,
        "false_positive": (not expected_alarm) and original_detected,
        "non_detection": expected_alarm and not original_detected,
        "partial_detection": status == "partial_detection",
        "matched_result_count": len(matched),
        "expected": {
            "should_alarm": expected_alarm,
            "conforms": expected_conforms,
            "expected_path": expected_path,
            "expected_path_iri": expand_qname(expected_path),
            "expected_component": expected_component,
            "expected_component_iri": expand_qname(expected_component),
            "shape_family": shape_family,
            "reserved_for_after_extension": reserved,
        },
        "reclassification_note": "Recomputed from preserved report.ttl; pySHACL was not rerun.",
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    report_files = sorted((RUN_DIR / "cases").glob("*/report.ttl"))
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed"
        if all(c["detection_status"] in {"control_passed", "detected_by_original"} for c in cases)
        else "attention_required",
        "correction": {
            "reason": "The original classifier compared expected QName strings with full IRIs reported by rdflib/pySHACL.",
            "method": "Reparse preserved report.ttl files and expand rr/sh/rrdom/skos QNames before comparing expected path/component.",
            "pyshacl_rerun": False,
            "mutated_graphs_changed": False,
        },
        "counts": {
            "total_cases": len(cases),
            "controls": sum(1 for c in cases if c["family"] == "valid_control"),
            "faults": sum(1 for c in cases if c["family"] == "shacl"),
            "detected_by_original": sum(1 for c in cases if c["detection_status"] == "detected_by_original"),
            "control_passed": sum(1 for c in cases if c["detection_status"] == "control_passed"),
            "partial_detection": sum(1 for c in cases if c["partial_detection"]),
            "not_detected": sum(1 for c in cases if c["non_detection"]),
            "false_positive": sum(1 for c in cases if c["false_positive"]),
        },
        "input_report_hashes": {rel(path): sha256_file(path) for path in report_files},
        "cases": cases,
        "semantic_boundary": "SHACL validation reports conformance to declared shapes only; it does not prove semantic correctness of ontology activations.",
        "scope_limit": "This SHACL subrun executes only the intact SHACL control plus the ten declared shacl_cases, not the full non-SHACL E1 continuity inventory.",
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# E1 SHACL Subrun Corrected Classification",
        "",
        f"Created UTC: {summary['created_at_utc']}",
        "",
        "The preserved pySHACL reports were not rerun. This file corrects only the report classifier by expanding expected `rr:`, `sh:`, `rrdom:`, and `skos:` QNames before comparing them with full IRIs in `report.ttl`.",
        "",
        "SHACL is interpreted as conformance to declared shapes only. These cases do not establish semantic correctness of ontology activations.",
        "",
        "## Counts",
        "",
        "| Measure | Count |",
        "|---|---:|",
    ]
    for key, value in summary["counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| Case | Expected alarm | Conforms | Results | Matches expected constraint | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for case in summary["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['oracle_should_alarm']} | {case['pyshacl_conforms']} | "
            f"{case['validation_result_count']} | {case['matched_result_count']} | {case['detection_status']} |"
        )
    lines.extend(["", "## Constraint Details", ""])
    for case in summary["cases"]:
        lines.append(f"### {case['case_id']}")
        lines.append("")
        if not case["results"]:
            lines.append("- No validation results.")
        for i, result in enumerate(case["results"], start=1):
            focus = ", ".join(result["focus_node"]) or "none"
            path = ", ".join(result["result_path"]) or "SPARQL/no path"
            component = ", ".join(result["source_constraint_component"]) or "none"
            severity = ", ".join(result["result_severity"]) or "none"
            message = " | ".join(result["message"]) or "none"
            lines.append(f"- Result {i}:")
            lines.append(f"  - focus: `{focus}`")
            lines.append(f"  - path: `{path}`")
            lines.append(f"  - component: `{component}`")
            lines.append(f"  - severity: `{severity}`")
            lines.append(f"  - message: {message}")
        lines.append("")
    lines.extend(["", "## Limitation", ""])
    lines.append(summary["scope_limit"])
    (RUN_DIR / "REPORT_RECLASSIFIED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    specs = read_specs()
    case_dirs = sorted((RUN_DIR / "cases").glob("E1-*"))
    expected_ids = {"E1-C0-pristine-shacl-control", *specs.keys()}
    found_ids = {p.name for p in case_dirs}
    if found_ids != expected_ids:
        missing = sorted(expected_ids - found_ids)
        extra = sorted(found_ids - expected_ids)
        raise SystemExit(f"Unexpected case directories. missing={missing} extra={extra}")
    cases = [reclassify_case(case_dir, specs) for case_dir in case_dirs]
    for case_dir, case in zip(case_dirs, cases, strict=True):
        write_json(case_dir / "result_reclassified.json", case)
    summary = summarize(cases)
    write_json(RUN_DIR / "summary_reclassified.json", summary)
    write_markdown(summary)
    print(json.dumps({"status": summary["status"], "counts": summary["counts"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit("Use run_e1_shacl.py --output-dir SCRATCH; reclassification runs there automatically.")
