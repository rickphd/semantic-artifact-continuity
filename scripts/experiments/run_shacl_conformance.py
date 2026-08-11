#!/usr/bin/env python3
"""Run SHACL conformance and controlled stress tests for generated RDF."""

from __future__ import annotations

import json

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

from ke_artifact_utils import (
    CANONICAL_ONTOLOGY_DIR,
    KNOWLEDGE_GRAPH_DIR,
    SHACL_DIR,
    ensure_generated_dirs,
    rel,
    write_json,
)


RR = Namespace("http://wfrp.ia/ontologia/ia-sentimiento#")
RRSENT = Namespace("http://wfrp.ia/sentimiento/core#")
RRDOM = Namespace("http://wfrp.ia/dominio/ciencia-tecnologia#")
BASE = Namespace("http://wfrp.ia/resource/reddit/")


def normalized_shapes_path() -> str:
    src = CANONICAL_ONTOLOGY_DIR / "rr-shapes.ttl"
    dst = SHACL_DIR / "canonical_shapes_normalized.ttl"
    text = src.read_text(encoding="utf-8")
    text = text.replace("https://www.w3.org/ns/shacl#", "http://www.w3.org/ns/shacl#")
    dst.write_text(text, encoding="utf-8")
    return str(dst)


def ontology_graph() -> Graph:
    g = Graph()
    for name in ["rr-core.ttl", "rr-domain.ttl", "rr-sentiment.ttl"]:
        g.parse(CANONICAL_ONTOLOGY_DIR / name, format="turtle")
    return g


def run_validation(data_graph: Graph, shapes_path: str):
    conforms, report_graph, report_text = validate(
        data_graph,
        shacl_graph=shapes_path,
        ont_graph=ontology_graph(),
        inference="rdfs",
        advanced=True,
        serialize_report_graph="turtle",
    )
    report_graph_text = report_graph.decode("utf-8") if isinstance(report_graph, bytes) else str(report_graph)
    return bool(conforms), report_graph_text, str(report_text)


def invalid_fixture() -> Graph:
    g = Graph()
    post = URIRef(f"{BASE}post/invalid_missing_fields")
    g.add((post, RDF.type, RR.Post))
    g.add((post, RR.identificador, Literal("invalid_missing_fields", datatype=XSD.string)))
    g.add((post, RR.idiomaDetectado, Literal("fr", datatype=XSD.string)))
    g.add((post, RR.longitudTexto, Literal(8, datatype=XSD.integer)))
    g.add((post, RR.etiquetaSentimiento, RRSENT.Positivo))
    g.add((post, RR.trataSobre, RRDOM.InteligenciaArtificial))
    return g


def main() -> None:
    ensure_generated_dirs()
    shapes_path = normalized_shapes_path()
    data_path = KNOWLEDGE_GRAPH_DIR / "posts.ttl"
    data = Graph()
    data.parse(data_path, format="turtle")

    conforms, report_ttl, report_text = run_validation(data, shapes_path)
    (SHACL_DIR / "baseline_report.ttl").write_text(report_ttl, encoding="utf-8")
    (SHACL_DIR / "baseline_report.txt").write_text(report_text, encoding="utf-8")

    stress = invalid_fixture()
    stress_path = SHACL_DIR / "invalid_fixture_missing_required.ttl"
    stress.serialize(destination=stress_path, format="turtle")
    stress_conforms, stress_ttl, stress_text = run_validation(stress, shapes_path)
    (SHACL_DIR / "stress_report.ttl").write_text(stress_ttl, encoding="utf-8")
    (SHACL_DIR / "stress_report.txt").write_text(stress_text, encoding="utf-8")

    baseline_violations = report_ttl.count("sh:ValidationResult")
    stress_violations = stress_ttl.count("sh:ValidationResult")
    payload = {
        "status": "passed" if (conforms and not stress_conforms and stress_violations > 0) else "attention_required",
        "namespace_normalization": {
            "source": rel(CANONICAL_ONTOLOGY_DIR / "rr-shapes.ttl"),
            "normalized_copy": rel(SHACL_DIR / "canonical_shapes_normalized.ttl"),
            "reason": "The source file used https://www.w3.org/ns/shacl#. pySHACL expects the W3C SHACL namespace http://www.w3.org/ns/shacl#.",
        },
        "baseline": {
            "conforms": conforms,
            "validation_result_count": baseline_violations,
            "report_ttl": rel(SHACL_DIR / "baseline_report.ttl"),
            "report_text": rel(SHACL_DIR / "baseline_report.txt"),
        },
        "stress_test": {
            "conforms": stress_conforms,
            "validation_result_count": stress_violations,
            "fixture": rel(stress_path),
            "report_ttl": rel(SHACL_DIR / "stress_report.ttl"),
            "report_text": rel(SHACL_DIR / "stress_report.txt"),
        },
    }
    write_json(SHACL_DIR / "baseline_report.json", payload)
    write_json(SHACL_DIR / "stress_test_report.json", payload["stress_test"])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
