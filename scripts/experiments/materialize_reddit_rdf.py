#!/usr/bin/env python3
"""Materialize RDF post instances from the gold enriched dataset."""

from __future__ import annotations

from datetime import timezone

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

from ke_artifact_utils import (
    CANONICAL_ONTOLOGY_DIR,
    KNOWLEDGE_GRAPH_DIR,
    TRACEABILITY_DIR,
    CONCEPT_URI_BY_MODULE,
    ensure_generated_dirs,
    feature_to_concept,
    feature_to_module,
    label_uri,
    load_gold,
    ontology_columns,
    rel,
    sanitize_fragment,
    stable_post_iri,
    write_csv,
    write_json,
)


RR = Namespace("http://wfrp.ia/ontologia/ia-sentimiento#")
RRDOM = Namespace("http://wfrp.ia/dominio/ciencia-tecnologia#")
BASE = Namespace("http://wfrp.ia/resource/reddit/")


def as_datetime_literal(value) -> Literal:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.Timestamp("1970-01-01T00:00:00Z")
    return Literal(ts.isoformat(), datatype=XSD.dateTime)


def row_text(row) -> str:
    title = "" if pd.isna(row.get("titulo_limpio")) else str(row.get("titulo_limpio"))
    text = "" if pd.isna(row.get("texto_limpio")) else str(row.get("texto_limpio"))
    return f"{title} {text}".strip()


def main() -> None:
    ensure_generated_dirs()
    df = load_gold()
    ont_cols = ontology_columns(df)
    g = Graph()
    g.bind("rr", RR)
    g.bind("rrdom", RRDOM)
    g.bind("base", BASE)

    trace_rows = []
    post_count = 0
    activation_count = 0
    concept_assertions: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        post_id = str(row["id"])
        post = URIRef(stable_post_iri(post_id))
        author = URIRef(f"{BASE}author/{sanitize_fragment(row.get('author_id') or 'unknown')}")
        subreddit = URIRef(f"{BASE}subreddit/{sanitize_fragment(row.get('subreddit_nombre') or 'unknown')}")
        g.add((post, RDF.type, RR.Post))
        g.add((author, RDF.type, RR.Author))
        g.add((subreddit, RDF.type, RR.Subreddit))
        g.add((post, RR.identificador, Literal(post_id, datatype=XSD.string)))
        g.add((post, RR.fechaCreacion, as_datetime_literal(row.get("fecha_creacion"))))
        g.add((post, RR.publicadoPor, author))
        g.add((post, RR.enSubreddit, subreddit))
        g.add((post, RR.idiomaDetectado, Literal("en")))
        g.add((post, RR.longitudTexto, Literal(len(row_text(row)), datatype=XSD.integer)))
        g.add((post, RR.etiquetaSentimiento, URIRef(label_uri(int(row["label"])))))
        post_count += 1

        active_for_post = 0
        for feature in ont_cols:
            value = row.get(feature)
            if pd.notna(value) and float(value) != 0.0:
                module = feature_to_module(feature)
                concept = feature_to_concept(feature)
                concept_uri = URIRef(str(RRDOM[concept])) if concept else None
                if concept_uri is not None:
                    g.add((post, RR.trataSobre, concept_uri))
                    concept_assertions.add((post_id, concept))
                trace_rows.append(
                    {
                        "post_id": post_id,
                        "split": row["split"],
                        "label": int(row["label"]),
                        "post_iri": str(post),
                        "feature": feature,
                        "module": module,
                        "concept_uri": str(concept_uri) if concept_uri else "",
                        "value": float(value),
                        "used_downstream": "",
                    }
                )
                active_for_post += 1
                activation_count += 1
        if active_for_post == 0:
            trace_rows.append(
                {
                    "post_id": post_id,
                    "split": row["split"],
                    "label": int(row["label"]),
                    "post_iri": str(post),
                    "feature": "",
                    "module": "",
                    "concept_uri": "",
                    "value": 0.0,
                    "used_downstream": "",
                }
            )

    rdf_path = KNOWLEDGE_GRAPH_DIR / "posts.ttl"
    g.serialize(destination=rdf_path, format="turtle")
    trace_path = TRACEABILITY_DIR / "traceability_map.csv"
    write_csv(trace_path, trace_rows)

    report = {
        "status": "passed",
        "canonical_ontology_dir": rel(CANONICAL_ONTOLOGY_DIR),
        "gold_rows": int(len(df)),
        "ontology_feature_count": len(ont_cols),
        "rdf_post_count": post_count,
        "rdf_triple_count": len(g),
        "nonzero_activation_count": activation_count,
        "posts_with_trace_rows": len({r["post_id"] for r in trace_rows}),
        "concept_assertion_count": len(concept_assertions),
        "rdf_output": rel(rdf_path),
        "traceability_output": rel(trace_path),
    }
    write_json(KNOWLEDGE_GRAPH_DIR / "materialization_report.json", report)
    print(f"Wrote {rel(rdf_path)} with {len(g)} triples")


if __name__ == "__main__":
    main()
