#!/usr/bin/env python3
"""Run KE-only sensitivity checks for the v04.2 VADER activation parameters.

The runner writes every artifact under
results/sensitivity and never overwrites the
canonical v04.2 dataset, lexicon, feature-selection report, RDF, or SHACL files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.feature_selection import f_classif


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT
OUT_ROOT = ROOT / "results" / "sensitivity"
SOURCE_GOLD = ROOT / "data" / "gold" / "gold_enriched_ontology.parquet"
SPLIT_PATH = ROOT / "data" / "gold" / "GEN_split_gld_reddit_ids_v02.json"
ONTOLOGY_DIR = ROOT / "src" / "ontology" / "resources"
SHAPES_PATH = ONTOLOGY_DIR / "rr-shapes.ttl"
ALPHA = 0.05


def load_build_module():
    path = ROOT / "scripts" / "experiments" / "build_train_only_ontology_dataset.py"
    spec = importlib.util.spec_from_file_location("build_train_only_ontology_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from ke_artifact_utils import (  # noqa: E402
    CONCEPT_URI_BY_MODULE,
    MODULES,
    feature_to_concept,
    feature_to_module,
    label_uri,
    sanitize_fragment,
    stable_post_iri,
)


build = None
rdflib = None
validate = None
Graph = Literal = Namespace = RDF = URIRef = XSD = None
OntologyEnricher = None
tqdm = None
RR = RRDOM = BASE = None


def load_runtime_dependencies() -> None:
    global build, rdflib, validate, Graph, Literal, Namespace, RDF, URIRef, XSD
    global OntologyEnricher, tqdm, RR, RRDOM, BASE

    import rdflib as rdflib_module
    from pyshacl import validate as pyshacl_validate
    from rdflib import Graph as RDFGraph
    from rdflib import Literal as RDFLiteral
    from rdflib import Namespace as RDFNamespace
    from rdflib import RDF as RDFType
    from rdflib import URIRef as RDFURIRef
    from rdflib.namespace import XSD as RDFXSD
    from src.ontology.enricher import OntologyEnricher as Enricher
    from tqdm import tqdm as progress

    build = load_build_module()
    rdflib = rdflib_module
    validate = pyshacl_validate
    Graph = RDFGraph
    Literal = RDFLiteral
    Namespace = RDFNamespace
    RDF = RDFType
    URIRef = RDFURIRef
    XSD = RDFXSD
    OntologyEnricher = Enricher
    tqdm = progress
    RR = Namespace("http://wfrp.ia/ontologia/ia-sentimiento#")
    RRDOM = Namespace("http://wfrp.ia/dominio/ciencia-tecnologia#")
    BASE = Namespace("http://wfrp.ia/resource/reddit/")


SMALL_VARIANTS = [
    {"variant_id": "canonical_w5_f3_v1p0_c2p0", "context_window": 5, "minimum_frequency": 3, "minimum_absolute_valence": 1.0, "maximum_absolute_context_weight": 2.0},
    {"variant_id": "window3_f3_v1p0_c2p0", "context_window": 3, "minimum_frequency": 3, "minimum_absolute_valence": 1.0, "maximum_absolute_context_weight": 2.0},
    {"variant_id": "window7_f3_v1p0_c2p0", "context_window": 7, "minimum_frequency": 3, "minimum_absolute_valence": 1.0, "maximum_absolute_context_weight": 2.0},
    {"variant_id": "w5_freq2_v1p0_c2p0", "context_window": 5, "minimum_frequency": 2, "minimum_absolute_valence": 1.0, "maximum_absolute_context_weight": 2.0},
    {"variant_id": "w5_freq5_v1p0_c2p0", "context_window": 5, "minimum_frequency": 5, "minimum_absolute_valence": 1.0, "maximum_absolute_context_weight": 2.0},
    {"variant_id": "w5_f3_val0p5_c2p0", "context_window": 5, "minimum_frequency": 3, "minimum_absolute_valence": 0.5, "maximum_absolute_context_weight": 2.0},
    {"variant_id": "w5_f3_val1p5_c2p0", "context_window": 5, "minimum_frequency": 3, "minimum_absolute_valence": 1.5, "maximum_absolute_context_weight": 2.0},
]


def token(value: float | int) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def expanded_variants() -> list[dict]:
    variants = []
    for window in [2, 3, 5, 7, 10]:
        for frequency in [1, 2, 3, 5, 8]:
            for valence in [0.25, 0.5, 1.0, 1.5, 2.0]:
                variants.append(
                    {
                        "variant_id": f"w{window}_f{frequency}_v{token(valence)}_c2p0",
                        "context_window": window,
                        "minimum_frequency": frequency,
                        "minimum_absolute_valence": valence,
                        "maximum_absolute_context_weight": 2.0,
                    }
                )
    return variants


def fmt(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    pd.DataFrame(rows, columns=keys).to_csv(path, index=False)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def configure_build_module(params: dict) -> None:
    build.CONTEXT_WINDOW = int(params["context_window"])
    build.MIN_FREQUENCY = int(params["minimum_frequency"])
    build.MIN_ABS_VADER_VALENCE = float(params["minimum_absolute_valence"])
    build.MAX_ABS_CONTEXT_WEIGHT = float(params["maximum_absolute_context_weight"])
    OntologyEnricher.CONTEXT_WINDOW = int(params["context_window"])


def combined_text(df: pd.DataFrame) -> pd.Series:
    return (df["titulo"].fillna("").astype(str) + " " + df["texto"].fillna("").astype(str)).str.strip()


def build_variant_dataset(df: pd.DataFrame, split: dict, variant_dir: Path, params: dict) -> tuple[pd.DataFrame, Path, dict]:
    configure_build_module(params)
    by_id = df.assign(id=df["id"].astype(str)).set_index("id", drop=False)
    all_split_ids = [str(i) for name in ["train", "val", "test"] for i in split[name]]
    train = by_id.loc[list(map(str, split["train"]))].copy()
    manifest = build.build_manifest(SOURCE_GOLD, split, train)
    manifest["sensitivity_variant"] = params
    manifest["canonical_dataset_not_overwritten"] = True
    lexicon_path = variant_dir / "lexicon_manifest.json"
    write_json(lexicon_path, manifest)

    enricher = OntologyEnricher(str(ONTOLOGY_DIR), lexicon_manifest=str(lexicon_path))
    base_columns = [c for c in df.columns if not c.startswith("ont_") and c != "split"]
    clean = by_id.loc[all_split_ids, base_columns].reset_index(drop=True)
    split_by_id = {str(i): name for name in ["train", "val", "test"] for i in split[name]}
    rows = []
    texts = combined_text(clean)
    for row, text in tqdm(zip(clean.to_dict("records"), texts), total=len(clean), desc=params["variant_id"]):
        row.update(enricher.enrich_text(text))
        row["split"] = split_by_id[str(row["id"])]
        rows.append(row)
    enriched = pd.DataFrame(rows)
    missing = [feature for feature in build.FEATURES if feature not in enriched]
    if missing:
        raise ValueError(f"{params['variant_id']} missing ontology features: {missing}")
    enriched = enriched[base_columns + build.FEATURES + ["split"]]
    parquet_path = variant_dir / "gold_enriched_ontology.parquet"
    enriched.to_parquet(parquet_path, index=False, compression="snappy")
    metadata = {
        "variant_id": params["variant_id"],
        "parameters": params,
        "rows": int(len(enriched)),
        "columns": int(len(enriched.columns)),
        "dataset_sha256": sha256(parquet_path),
        "lexicon_manifest": rel(lexicon_path),
        "lexicon_manifest_sha256": sha256(lexicon_path),
        "split_distribution": {k: int(v) for k, v in enriched["split"].value_counts().to_dict().items()},
        "canonical_dataset": rel(SOURCE_GOLD),
        "canonical_dataset_not_overwritten": True,
    }
    write_json(variant_dir / "dataset_metadata.json", metadata)
    return enriched, lexicon_path, manifest


def select_features(enriched: pd.DataFrame, split: dict, variant_dir: Path, lexicon_path: Path) -> dict:
    train = enriched.set_index("id").loc[list(map(str, split["train"]))].reset_index()
    ont_cols = [c for c in enriched.columns if c.startswith("ont_")]
    f_stats, p_values = f_classif(train[ont_cols].fillna(0).astype(float), train["label"])
    ranking = pd.DataFrame({"feature": ont_cols, "f_stat": f_stats, "p_value": p_values}).sort_values(
        ["p_value", "f_stat"], ascending=[True, False], na_position="last"
    )
    selected = ranking.loc[ranking["p_value"] < ALPHA, "feature"].tolist()
    payload = {
        "variant_id": variant_dir.name,
        "method": "ANOVA F-test (sklearn.feature_selection.f_classif), train-only",
        "alpha": ALPHA,
        "train_size": int(len(train)),
        "n_candidates": len(ont_cols),
        "n_selected": len(selected),
        "selected_features_train_only": selected,
        "lexicon_manifest": rel(lexicon_path),
        "lexicon_manifest_sha256": sha256(lexicon_path),
        "ranking": [
            {
                "feature": row.feature,
                "f_stat": None if np.isnan(row.f_stat) else float(row.f_stat),
                "p_value": None if np.isnan(row.p_value) else float(row.p_value),
            }
            for row in ranking.itertuples(index=False)
        ],
        "environment": {"scikit_learn": sklearn.__version__},
    }
    write_json(variant_dir / "feature_selection_anova_train_only.json", payload)
    return payload


def coverage_report(enriched: pd.DataFrame, selected: list[str], variant_dir: Path) -> dict:
    ont_cols = [c for c in enriched.columns if c.startswith("ont_")]
    feature_rows = []
    for col in ont_cols:
        s = enriched[col].fillna(0).astype(float)
        nonzero = s != 0
        feature_rows.append(
            {
                "feature": col,
                "module": feature_to_module(col),
                "selected_train_only": col in selected,
                "nonzero_count": int(nonzero.sum()),
                "nonzero_pct": float(nonzero.mean() * 100),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "max": float(s.max()),
                "missing_count": int(enriched[col].isna().sum()),
            }
        )
    write_csv(variant_dir / "coverage_sparsity.csv", feature_rows)
    module_rows = []
    for module, cols in MODULES.items():
        present = [c for c in cols if c in enriched.columns]
        if not present:
            continue
        active = enriched[present].fillna(0).astype(float).abs().sum(axis=1) > 0
        row = {
            "module": module,
            "feature_count": len(present),
            "active_posts": int(active.sum()),
            "active_pct": float(active.mean() * 100),
            "selected_feature_count": sum(c in selected for c in present),
        }
        module_rows.append(row)
    write_csv(variant_dir / "module_coverage.csv", module_rows)
    overall_active = enriched[ont_cols].fillna(0).astype(float).abs().sum(axis=1) > 0
    selected_active = enriched[selected].fillna(0).astype(float).abs().sum(axis=1) > 0 if selected else pd.Series(False, index=enriched.index)
    report = {
        "variant_id": variant_dir.name,
        "rows": int(len(enriched)),
        "ontology_feature_count": len(ont_cols),
        "selected_feature_count": len(selected),
        "posts_with_any_ontology_feature": int(overall_active.sum()),
        "overall_activation_pct": float(overall_active.mean() * 100),
        "posts_with_selected_feature": int(selected_active.sum()),
        "selected_activation_pct": float(selected_active.mean() * 100),
        "split_activation_pct": {
            split: float(overall_active[enriched["split"] == split].mean() * 100)
            for split in ["train", "val", "test"]
        },
    }
    write_json(variant_dir / "coverage_sparsity_report.json", report)
    return report


def as_datetime_literal(value) -> Literal:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.Timestamp("1970-01-01T00:00:00Z")
    return Literal(ts.isoformat(), datatype=XSD.dateTime)


def row_text(row) -> str:
    title = "" if pd.isna(row.get("titulo_limpio")) else str(row.get("titulo_limpio"))
    text = "" if pd.isna(row.get("texto_limpio")) else str(row.get("texto_limpio"))
    return f"{title} {text}".strip()


def materialize_rdf(enriched: pd.DataFrame, selected: list[str], variant_dir: Path) -> tuple[dict, Path]:
    g = Graph()
    g.bind("rr", RR)
    g.bind("rrdom", RRDOM)
    g.bind("base", BASE)
    ont_cols = [c for c in enriched.columns if c.startswith("ont_")]
    trace_rows = []
    activation_count = 0
    selected_activation_count = 0
    concept_assertions: set[tuple[str, str]] = set()
    for _, row in enriched.iterrows():
        post_id = str(row["id"])
        post = URIRef(stable_post_iri(post_id))
        author = URIRef(f"{BASE}author/{sanitize_fragment(row.get('autor') or 'unknown')}")
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
                used_downstream = feature in selected
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
                        "used_downstream": used_downstream,
                    }
                )
                active_for_post += 1
                activation_count += 1
                if used_downstream:
                    selected_activation_count += 1
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
                    "used_downstream": False,
                }
            )
    rdf_path = variant_dir / "rdf" / "posts.ttl"
    rdf_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=rdf_path, format="turtle")
    write_csv(variant_dir / "traceability_map.csv", trace_rows)
    selected_trace_rate = 100.0 if selected_activation_count == sum(
        1 for row in trace_rows if row.get("used_downstream") is True
    ) else 0.0
    report = {
        "variant_id": variant_dir.name,
        "rdf_post_count": int(len(enriched)),
        "rdf_triple_count": int(len(g)),
        "ontology_feature_count": len(ont_cols),
        "nonzero_activation_count": activation_count,
        "nonzero_selected_activations": selected_activation_count,
        "concept_assertion_count": len(concept_assertions),
        "posts_with_trace_rows": len({r["post_id"] for r in trace_rows}),
        "selected_trace_rate_pct": selected_trace_rate,
        "rdf_output": rel(rdf_path),
        "traceability_output": rel(variant_dir / "traceability_map.csv"),
    }
    write_json(variant_dir / "rdf_materialization_report.json", report)
    return report, rdf_path


def normalized_shapes(variant_dir: Path) -> Path:
    out = variant_dir / "shacl" / "canonical_shapes_normalized.ttl"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = SHAPES_PATH.read_text(encoding="utf-8")
    text = text.replace("https://www.w3.org/ns/shacl#", "http://www.w3.org/ns/shacl#")
    out.write_text(text, encoding="utf-8")
    return out


def ontology_graph() -> Graph:
    g = Graph()
    for name in ["rr-core.ttl", "rr-domain.ttl", "rr-sentiment.ttl"]:
        g.parse(ONTOLOGY_DIR / name, format="turtle")
    return g


def run_shacl(rdf_path: Path, variant_dir: Path) -> dict:
    shapes = normalized_shapes(variant_dir)
    data = Graph()
    data.parse(rdf_path, format="turtle")
    conforms, report_graph, report_text = validate(
        data_graph=data,
        shacl_graph=str(shapes),
        ont_graph=ontology_graph(),
        inference="rdfs",
        advanced=True,
        serialize_report_graph="turtle",
    )
    report_ttl = report_graph.decode("utf-8") if isinstance(report_graph, bytes) else str(report_graph)
    (variant_dir / "shacl" / "baseline_report.ttl").write_text(report_ttl, encoding="utf-8")
    (variant_dir / "shacl" / "baseline_report.txt").write_text(str(report_text), encoding="utf-8")
    payload = {
        "variant_id": variant_dir.name,
        "status": "passed" if conforms else "failed",
        "conforms": bool(conforms),
        "validation_result_count": report_ttl.count("sh:ValidationResult"),
        "inference": "rdfs",
        "advanced": True,
        "report_ttl": rel(variant_dir / "shacl" / "baseline_report.ttl"),
        "report_text": rel(variant_dir / "shacl" / "baseline_report.txt"),
    }
    write_json(variant_dir / "shacl" / "baseline_report.json", payload)
    return payload


def run_variant(df: pd.DataFrame, split: dict, params: dict) -> dict:
    variant_dir = OUT_ROOT / params["variant_id"]
    variant_dir.mkdir(parents=True, exist_ok=True)
    enriched, lexicon_path, manifest = build_variant_dataset(df, split, variant_dir, params)
    selection = select_features(enriched, split, variant_dir, lexicon_path)
    coverage = coverage_report(enriched, selection["selected_features_train_only"], variant_dir)
    rdf_report, rdf_path = materialize_rdf(enriched, selection["selected_features_train_only"], variant_dir)
    shacl = run_shacl(rdf_path, variant_dir)
    return {
        "variant_id": params["variant_id"],
        "context_window": params["context_window"],
        "minimum_frequency": params["minimum_frequency"],
        "minimum_absolute_valence": params["minimum_absolute_valence"],
        "maximum_absolute_context_weight": params["maximum_absolute_context_weight"],
        "lexicon_size": manifest["statistics"]["combined_count"],
        "positive_terms": manifest["statistics"]["selected_positive_count"],
        "negative_terms": manifest["statistics"]["selected_negative_count"],
        "train_context_token_count": manifest["statistics"]["train_context_token_count"],
        "train_context_vocabulary_size": manifest["statistics"]["train_context_vocabulary_size"],
        "selected_feature_count": selection["n_selected"],
        "selected_features": ";".join(selection["selected_features_train_only"]),
        "posts_with_any_ontology_feature": coverage["posts_with_any_ontology_feature"],
        "overall_activation_pct": coverage["overall_activation_pct"],
        "posts_with_selected_feature": coverage["posts_with_selected_feature"],
        "selected_activation_pct": coverage["selected_activation_pct"],
        "rdf_triple_count": rdf_report["rdf_triple_count"],
        "concept_assertion_count": rdf_report["concept_assertion_count"],
        "nonzero_activation_count": rdf_report["nonzero_activation_count"],
        "nonzero_selected_activations": rdf_report["nonzero_selected_activations"],
        "selected_trace_rate_pct": rdf_report["selected_trace_rate_pct"],
        "shacl_conforms": shacl["conforms"],
        "shacl_validation_result_count": shacl["validation_result_count"],
        "variant_dir": rel(variant_dir),
    }


def numeric_summary(values: list[float]) -> dict:
    clean_values = [value for value in values if value is not None and not pd.isna(value)]
    if not clean_values:
        return {"min": None, "q1": None, "median": None, "mean": None, "q3": None, "max": None, "std": None}
    series = pd.Series(clean_values, dtype=float)
    return {
        "min": float(series.min()),
        "q1": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "q3": float(series.quantile(0.75)),
        "max": float(series.max()),
        "std": float(series.std(ddof=0)),
    }


def write_summary(rows: list[dict], grid_name: str) -> None:
    write_csv(OUT_ROOT / "vader_ke_sensitivity_summary.csv", rows)
    successful_rows = [row for row in rows if row.get("variant_status") == "passed"]
    canonical = next(
        (row for row in successful_rows if row["context_window"] == 5 and row["minimum_frequency"] == 3 and float(row["minimum_absolute_valence"]) == 1.0),
        successful_rows[0] if successful_rows else rows[0],
    )
    compared = []
    canonical_selected = set(canonical["selected_features"].split(";")) if canonical["selected_features"] else set()
    for row in rows:
        selected = set(row.get("selected_features", "").split(";")) if row.get("selected_features") else set()
        compared.append(
            {
                **row,
                "delta_lexicon_size": row["lexicon_size"] - canonical["lexicon_size"] if row.get("lexicon_size") is not None and canonical.get("lexicon_size") is not None else None,
                "delta_overall_activation_pct": row["overall_activation_pct"] - canonical["overall_activation_pct"] if row.get("overall_activation_pct") is not None and canonical.get("overall_activation_pct") is not None else None,
                "delta_selected_feature_count": row["selected_feature_count"] - canonical["selected_feature_count"] if row.get("selected_feature_count") is not None and canonical.get("selected_feature_count") is not None else None,
                "selected_feature_jaccard_vs_canonical": (
                    len(selected & canonical_selected) / len(selected | canonical_selected)
                    if selected or canonical_selected else 1.0
                ) if row.get("variant_status") == "passed" else None,
            }
        )
    write_json(OUT_ROOT / "vader_ke_sensitivity_summary.json", compared)

    feature_counts: dict[str, int] = {}
    for row in successful_rows:
        selected = [f for f in row.get("selected_features", "").split(";") if f]
        for feature in selected:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    feature_stability = [
        {
            "feature": feature,
            "selected_count": count,
            "selected_pct": 100.0 * count / len(successful_rows),
            "canonical_selected": feature in canonical_selected,
        }
        for feature, count in sorted(feature_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    write_csv(OUT_ROOT / "selected_feature_stability.csv", feature_stability)

    aggregate = {
        "status": "passed",
        "grid_name": grid_name,
        "variant_count": len(compared),
        "successful_variant_count": len(successful_rows),
        "invalid_variant_count": len(compared) - len(successful_rows),
        "canonical_variant": canonical["variant_id"],
        "shacl_pass_count": sum(1 for row in successful_rows if row["shacl_conforms"]),
        "shacl_pass_pct_successful_variants": 100.0 * sum(1 for row in successful_rows if row["shacl_conforms"]) / len(successful_rows) if successful_rows else 0.0,
        "lexicon_size": numeric_summary([row.get("lexicon_size") for row in successful_rows]),
        "selected_feature_count": numeric_summary([row.get("selected_feature_count") for row in successful_rows]),
        "overall_activation_pct": numeric_summary([row.get("overall_activation_pct") for row in successful_rows]),
        "rdf_triple_count": numeric_summary([row.get("rdf_triple_count") for row in successful_rows]),
        "nonzero_activation_count": numeric_summary([row.get("nonzero_activation_count") for row in successful_rows]),
        "selected_feature_jaccard_vs_canonical": numeric_summary([row.get("selected_feature_jaccard_vs_canonical") for row in compared]),
        "selected_feature_stability": feature_stability,
        "invalid_variants": [
            {
                "variant_id": row["variant_id"],
                "context_window": row["context_window"],
                "minimum_frequency": row["minimum_frequency"],
                "minimum_absolute_valence": row["minimum_absolute_valence"],
                "error": row.get("error"),
            }
            for row in compared
            if row.get("variant_status") != "passed"
        ],
    }
    write_json(OUT_ROOT / "vader_ke_sensitivity_aggregate_stats.json", aggregate)
    md = [
        "# VADER KE-Only Sensitivity Summary",
        "",
        f"Status: `passed`",
        f"Grid: `{grid_name}`",
        f"Variants: `{len(compared)}`",
        f"Created at: `{datetime.now().isoformat()}`",
        "",
        "This KE-only sensitivity run does not retrain ML or DL models and does not overwrite canonical v04.2 artefacts.",
        "",
        "| Variant | Window | Min freq | Min valence | Lexicon | Selected | Activation % | RDF triples | SHACL | Jaccard selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in compared:
        md.append(
            f"| `{row['variant_id']}` | {row['context_window']} | {row['minimum_frequency']} | "
            f"{row['minimum_absolute_valence']} | {fmt(row.get('lexicon_size'), 0)} | {fmt(row.get('selected_feature_count'), 0)} | "
            f"{fmt(row.get('overall_activation_pct'))} | {fmt(row.get('rdf_triple_count'), 0)} | "
            f"{'pass' if row['shacl_conforms'] else 'fail'} | "
            f"{fmt(row.get('selected_feature_jaccard_vs_canonical'))} |"
        )
    md.extend(
        [
            "",
            "## Aggregate Statistics",
            "",
            f"- Successful variants: {aggregate['successful_variant_count']}/{aggregate['variant_count']}.",
            f"- Invalid variants: {aggregate['invalid_variant_count']}/{aggregate['variant_count']}.",
            f"- SHACL pass rate among successful variants: {aggregate['shacl_pass_pct_successful_variants']:.2f}% ({aggregate['shacl_pass_count']}/{aggregate['successful_variant_count']}).",
            f"- Lexicon size: median {fmt(aggregate['lexicon_size']['median'])}, range {fmt(aggregate['lexicon_size']['min'], 0)}-{fmt(aggregate['lexicon_size']['max'], 0)}.",
            f"- Selected variables: median {fmt(aggregate['selected_feature_count']['median'])}, range {fmt(aggregate['selected_feature_count']['min'], 0)}-{fmt(aggregate['selected_feature_count']['max'], 0)}.",
            f"- Jaccard vs canonical selected set: median {fmt(aggregate['selected_feature_jaccard_vs_canonical']['median'])}, range {fmt(aggregate['selected_feature_jaccard_vs_canonical']['min'])}-{fmt(aggregate['selected_feature_jaccard_vs_canonical']['max'])}.",
            "",
            "## Selected Feature Stability",
            "",
            "| Feature | Selected % | Canonical selected |",
            "|---|---:|---|",
        ]
    )
    for row in feature_stability:
        md.append(
            f"| `{row['feature']}` | {row['selected_pct']:.2f} | "
            f"{'yes' if row['canonical_selected'] else 'no'} |"
        )
    if aggregate["invalid_variants"]:
        md.extend(["", "## Invalid Parameter Settings", ""])
        for row in aggregate["invalid_variants"]:
            md.append(
                f"- `{row['variant_id']}`: {row['error']}"
            )
    md.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These results support only KE artefact sensitivity claims: lexicon size, ontology activation, RDF materialization, SHACL conformance, traceability, and train-only selected-variable stability. They do not support claims about downstream predictive performance because ML/DL models were not retrained.",
        ]
    )
    (OUT_ROOT / "vader_ke_sensitivity_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def collect_existing_variant(params: dict) -> dict:
    variant_dir = OUT_ROOT / params["variant_id"]
    lexicon_path = variant_dir / "lexicon_manifest.json"
    selection_path = variant_dir / "feature_selection_anova_train_only.json"
    coverage_path = variant_dir / "coverage_sparsity_report.json"
    rdf_path = variant_dir / "rdf_materialization_report.json"
    shacl_path = variant_dir / "shacl" / "baseline_report.json"
    if not all(path.exists() for path in [lexicon_path, selection_path, coverage_path, rdf_path, shacl_path]):
        error = "missing complete generated artefacts"
        if lexicon_path.exists():
            try:
                payload = json.loads(lexicon_path.read_text(encoding="utf-8"))
                combined = payload.get("combined_lexicon", {})
                if not combined.get("positive") or not combined.get("negative"):
                    error = "lexicon manifest lacks combined positive/negative entries"
            except Exception as exc:
                error = str(exc)
        return {
            "variant_id": params["variant_id"],
            "variant_status": "invalid",
            "context_window": params["context_window"],
            "minimum_frequency": params["minimum_frequency"],
            "minimum_absolute_valence": params["minimum_absolute_valence"],
            "maximum_absolute_context_weight": params["maximum_absolute_context_weight"],
            "lexicon_size": None,
            "positive_terms": None,
            "negative_terms": None,
            "train_context_token_count": None,
            "train_context_vocabulary_size": None,
            "selected_feature_count": None,
            "selected_features": "",
            "posts_with_any_ontology_feature": None,
            "overall_activation_pct": None,
            "posts_with_selected_feature": None,
            "selected_activation_pct": None,
            "rdf_triple_count": None,
            "concept_assertion_count": None,
            "nonzero_activation_count": None,
            "nonzero_selected_activations": None,
            "selected_trace_rate_pct": None,
            "shacl_conforms": False,
            "shacl_validation_result_count": None,
            "variant_dir": rel(variant_dir),
            "error": error,
        }

    lexicon = json.loads(lexicon_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    rdf = json.loads(rdf_path.read_text(encoding="utf-8"))
    shacl = json.loads(shacl_path.read_text(encoding="utf-8"))
    stats = lexicon["statistics"]
    selected = selection["selected_features_train_only"]
    return {
        "variant_id": params["variant_id"],
        "variant_status": "passed",
        "context_window": params["context_window"],
        "minimum_frequency": params["minimum_frequency"],
        "minimum_absolute_valence": params["minimum_absolute_valence"],
        "maximum_absolute_context_weight": params["maximum_absolute_context_weight"],
        "lexicon_size": stats["combined_count"],
        "positive_terms": stats["selected_positive_count"],
        "negative_terms": stats["selected_negative_count"],
        "train_context_token_count": stats["train_context_token_count"],
        "train_context_vocabulary_size": stats["train_context_vocabulary_size"],
        "selected_feature_count": selection["n_selected"],
        "selected_features": ";".join(selected),
        "posts_with_any_ontology_feature": coverage["posts_with_any_ontology_feature"],
        "overall_activation_pct": coverage["overall_activation_pct"],
        "posts_with_selected_feature": coverage["posts_with_selected_feature"],
        "selected_activation_pct": coverage["selected_activation_pct"],
        "rdf_triple_count": rdf["rdf_triple_count"],
        "concept_assertion_count": rdf["concept_assertion_count"],
        "nonzero_activation_count": rdf["nonzero_activation_count"],
        "nonzero_selected_activations": rdf["nonzero_selected_activations"],
        "selected_trace_rate_pct": rdf["selected_trace_rate_pct"],
        "shacl_conforms": shacl["conforms"],
        "shacl_validation_result_count": shacl["validation_result_count"],
        "variant_dir": rel(variant_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=["small", "expanded"], default="small")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of variants for smoke testing.")
    parser.add_argument("--summarize-existing", action="store_true", help="Build summaries from existing variant directories without rerunning variants.")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SOURCE_GOLD)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    rows = []
    all_variants = SMALL_VARIANTS if args.grid == "small" else expanded_variants()
    variants = all_variants[: args.limit] if args.limit else all_variants
    if args.summarize_existing:
        rows = [collect_existing_variant(params) for params in variants]
        write_summary(rows, args.grid)
        print(f"Wrote {rel(OUT_ROOT / 'vader_ke_sensitivity_summary.md')}")
        return
    load_runtime_dependencies()
    for params in variants:
        try:
            row = run_variant(df, split, params)
            row["variant_status"] = "passed"
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "variant_id": params["variant_id"],
                    "variant_status": "invalid",
                    "context_window": params["context_window"],
                    "minimum_frequency": params["minimum_frequency"],
                    "minimum_absolute_valence": params["minimum_absolute_valence"],
                    "maximum_absolute_context_weight": params["maximum_absolute_context_weight"],
                    "lexicon_size": None,
                    "positive_terms": None,
                    "negative_terms": None,
                    "train_context_token_count": None,
                    "train_context_vocabulary_size": None,
                    "selected_feature_count": None,
                    "selected_features": "",
                    "posts_with_any_ontology_feature": None,
                    "overall_activation_pct": None,
                    "posts_with_selected_feature": None,
                    "selected_activation_pct": None,
                    "rdf_triple_count": None,
                    "concept_assertion_count": None,
                    "nonzero_activation_count": None,
                    "nonzero_selected_activations": None,
                    "selected_trace_rate_pct": None,
                    "shacl_conforms": False,
                    "shacl_validation_result_count": None,
                    "variant_dir": rel(OUT_ROOT / params["variant_id"]),
                    "error": str(exc),
                }
            )
    manifest = {
        "status": "passed",
        "created_at": datetime.now().isoformat(),
        "variant_count": len(rows),
        "canonical_dataset": rel(SOURCE_GOLD),
        "canonical_dataset_sha256": sha256(SOURCE_GOLD),
        "canonical_artefacts_not_overwritten": True,
        "grid_design": args.grid,
        "variants": all_variants,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "rdflib": rdflib.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    write_json(OUT_ROOT / "vader_ke_sensitivity_manifest.json", manifest)
    write_summary(rows, args.grid)
    print(f"Wrote {rel(OUT_ROOT / 'vader_ke_sensitivity_summary.md')}")


if __name__ == "__main__":
    main()
