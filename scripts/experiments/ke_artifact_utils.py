"""Shared utilities for the knowledge-engineering artifact scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT
RESULTS_DIR = REPO_ROOT / "results"
ABLATION_DIR = RESULTS_DIR / "ablation"
ANOVA_DIR = RESULTS_DIR / "anova_revalidation"
COVERAGE_DIR = RESULTS_DIR / "validation" / "coverage"
KNOWLEDGE_GRAPH_DIR = RESULTS_DIR / "knowledge_graph"
MODEL_INPUT_DIR = RESULTS_DIR / "traceability" / "model_inputs"
PROVENANCE_DIR = RESULTS_DIR / "provenance"
SENSITIVITY_DIR = RESULTS_DIR / "sensitivity"
SHACL_DIR = RESULTS_DIR / "validation" / "shacl"
TRACEABILITY_DIR = RESULTS_DIR / "traceability"
VALIDATION_DIR = RESULTS_DIR / "validation"
GOLD_PATH = REPO_ROOT / "data" / "gold" / "gold_enriched_ontology.parquet"
GOLD_METADATA = REPO_ROOT / "data" / "gold" / "gold_enriched_ontology_metadata.json"
SPLIT_PATH = REPO_ROOT / "data" / "gold" / "GEN_split_gld_reddit_ids_v02.json"
FEATURE_SELECTION = REPO_ROOT / "results" / "feature_selection" / "ENR_selected_ont_features_anova_train_only.json"
LEXICON_MANIFEST = REPO_ROOT / "results" / "lexicon" / "ontology_lexicon_v04_2_train_only.json"
CANONICAL_ONTOLOGY_DIR = REPO_ROOT / "src" / "ontology" / "resources"
PACKAGED_ONTOLOGY_DIR = CANONICAL_ONTOLOGY_DIR


MODULES: dict[str, list[str]] = {
    "ai": [
        "ont_InteligenciaArtificial_Positivo",
        "ont_InteligenciaArtificial_Negativo",
        "ont_InteligenciaArtificial_Neutro",
    ],
    "ml": [
        "ont_AprendizajeAutomatico_Positivo",
        "ont_AprendizajeAutomatico_Negativo",
        "ont_AprendizajeAutomatico_Neutro",
    ],
    "technology": [
        "ont_Tecnologia_Positivo",
        "ont_Tecnologia_Negativo",
        "ont_Tecnologia_Neutro",
    ],
    "future": [
        "ont_Futuro_Positivo",
        "ont_Futuro_Negativo",
        "ont_Futuro_Neutro",
    ],
    "data": ["ont_Datos_Positivo", "ont_Datos_Negativo", "ont_Datos_Neutro"],
    "algorithm": [
        "ont_Algoritmo_Positivo",
        "ont_Algoritmo_Negativo",
        "ont_Algoritmo_Neutro",
    ],
    "robot": ["ont_Robot_Positivo", "ont_Robot_Negativo", "ont_Robot_Neutro"],
    "automation": [
        "ont_Automatizacion_Positivo",
        "ont_Automatizacion_Negativo",
        "ont_Automatizacion_Neutro",
    ],
    "ethics": ["ont_Etica_Positivo", "ont_Etica_Negativo", "ont_Etica_Neutro"],
    "innovation": [
        "ont_Innovacion_Positivo",
        "ont_Innovacion_Negativo",
        "ont_Innovacion_Neutro",
    ],
    "global_counts": [
        "ont_has_ontology_elements",
        "ont_count_domain_concepts",
        "ont_total_positivo_mentions",
        "ont_total_negativo_mentions",
        "ont_total_neutro_mentions",
        "ont_domain_score",
        "ont_domain_density",
    ],
}


CONCEPT_URI_BY_MODULE = {
    "ai": "InteligenciaArtificial",
    "ml": "AprendizajeAutomatico",
    "technology": "Tecnologia",
    "future": "Futuro",
    "data": "Datos",
    "algorithm": "Algoritmo",
    "robot": "Robot",
    "automation": "Automatizacion",
    "ethics": "Etica",
    "innovation": "Innovacion",
}


def ensure_generated_dirs() -> None:
    for path in [
        RESULTS_DIR,
        ABLATION_DIR,
        ANOVA_DIR,
        COVERAGE_DIR,
        KNOWLEDGE_GRAPH_DIR,
        MODEL_INPUT_DIR,
        PROVENANCE_DIR,
        SENSITIVITY_DIR,
        SHACL_DIR,
        TRACEABILITY_DIR,
        VALIDATION_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_gold() -> pd.DataFrame:
    return pd.read_parquet(GOLD_PATH)


def ontology_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ont_")]


def feature_to_module(feature: str) -> str:
    for module, cols in MODULES.items():
        if feature in cols:
            return module
    return "unmapped"


def feature_to_concept(feature: str) -> str | None:
    module = feature_to_module(feature)
    return CONCEPT_URI_BY_MODULE.get(module)


def stable_post_iri(post_id: str) -> str:
    return f"http://wfrp.ia/resource/reddit/post/{sanitize_fragment(post_id)}"


def sanitize_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\\-]", "_", str(value))
    return cleaned or "missing"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def selected_features() -> list[str]:
    payload = load_json(FEATURE_SELECTION)
    return payload["selected_features_train_only"]


def label_uri(label: int) -> str:
    return {
        0: "http://wfrp.ia/sentimiento/core#Negativo",
        1: "http://wfrp.ia/sentimiento/core#Neutro",
        2: "http://wfrp.ia/sentimiento/core#Positivo",
    }[int(label)]
