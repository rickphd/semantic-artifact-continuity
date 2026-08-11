#!/usr/bin/env python3
"""Build the leakage-remediated ontology dataset and lexicon.

The domain lexical layer is induced exclusively from canonical training IDs.
Validation and test labels are never read by the induction functions. The
resulting manifest is then frozen and applied to every split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import rdflib
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


ROOT = Path(__file__).resolve().parents[2]
SOURCE_GOLD = ROOT / "data" / "gold" / "gold_enriched_ontology.parquet"
GOLD_PATH = SOURCE_GOLD
METADATA_PATH = ROOT / "data" / "gold" / "gold_enriched_ontology_metadata.json"
SPLIT_PATH = ROOT / "data" / "gold" / "GEN_split_gld_reddit_ids_v02.json"
LEXICON_PATH = ROOT / "results" / "lexicon" / "ontology_lexicon_v04_2_train_only.json"
REPORT_PATH = ROOT / "results" / "lexicon" / "lexicon_induction_report.json"
ONTOLOGY_DIR = ROOT / "src" / "ontology" / "resources"

CONTEXT_WINDOW = 5
MIN_FREQUENCY = 3
MIN_ABS_VADER_VALENCE = 1.0
MAX_ABS_CONTEXT_WEIGHT = 2.0

CONCEPT_PATTERNS = [
    re.compile(r"\bai\b", re.IGNORECASE),
    re.compile(r"\bia\b", re.IGNORECASE),
    re.compile(r"artificial intelligence", re.IGNORECASE),
    re.compile(r"inteligencia artificial", re.IGNORECASE),
    re.compile(r"machine learning", re.IGNORECASE),
    re.compile(r"aprendizaje automático", re.IGNORECASE),
    re.compile(r"\bml\b", re.IGNORECASE),
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "should", "could", "may", "might", "must", "can", "that",
    "this", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "just", "don", "now", "el", "la", "los", "las", "un", "una", "de",
    "del", "al", "y", "o", "pero", "en", "con", "ai", "ia", "ml", "learning",
    "machine", "aprendizaje", "artificial", "intelligence", "inteligencia",
    "about", "after", "into", "over", "also", "one", "two", "new", "get",
    "make", "use", "their", "its",
}

FEATURES = [
    f"ont_{concept}_{polarity}"
    for concept in [
        "InteligenciaArtificial", "AprendizajeAutomatico", "Tecnologia", "Futuro",
        "Datos", "Algoritmo", "Robot", "Automatizacion", "Etica", "Innovacion",
    ]
    for polarity in ["Positivo", "Negativo", "Neutro"]
] + [
    "ont_has_ontology_elements", "ont_count_domain_concepts",
    "ont_total_positivo_mentions", "ont_total_negativo_mentions",
    "ont_total_neutro_mentions", "ont_domain_score", "ont_domain_density",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()


def combined_text(df: pd.DataFrame) -> pd.Series:
    return (df["titulo"].fillna("").astype(str) + " " + df["texto"].fillna("").astype(str)).str.strip()


def context_tokens(text: str) -> list[str]:
    lowered = str(text).lower()
    tokens = re.findall(r"\b\w+\b", lowered)
    selected: list[str] = []
    for pattern in CONCEPT_PATTERNS:
        for match in pattern.finditer(lowered):
            index = max(0, len(lowered[: match.start()].split()) - 1)
            start = max(0, index - CONTEXT_WINDOW)
            end = min(len(tokens), index + CONTEXT_WINDOW + 1)
            selected.extend(tokens[start:end])
    return [token for token in selected if token not in STOPWORDS and len(token) > 2]


def select_external_lexicon(train: pd.DataFrame) -> tuple[dict, dict, list[dict], dict]:
    """Select VADER terms observed in train concept windows without reading labels."""
    counts: Counter[str] = Counter()
    for row in train[["titulo", "texto"]].itertuples(index=False):
        counts.update(context_tokens(f"{row.titulo} {row.texto}"))

    analyzer = SentimentIntensityAnalyzer()
    vader = analyzer.lexicon
    rows = []
    positive: dict[str, float] = {}
    negative: dict[str, float] = {}
    for term, frequency in sorted(counts.items()):
        if frequency < MIN_FREQUENCY or term not in vader:
            continue
        valence = float(vader[term])
        if abs(valence) < MIN_ABS_VADER_VALENCE:
            continue
        bounded = max(-MAX_ABS_CONTEXT_WEIGHT, min(MAX_ABS_CONTEXT_WEIGHT, valence))
        rows.append(
            {
                "term": term,
                "train_context_frequency": int(frequency),
                "vader_valence": valence,
                "bounded_context_weight": bounded,
            }
        )
        if bounded > 0:
            positive[term] = bounded
        else:
            negative[term] = bounded
    rows.sort(key=lambda row: (-row["train_context_frequency"], row["term"]))
    vader_hash = hashlib.sha256(
        json.dumps(vader, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return positive, negative, rows, {
        "train_context_token_count": int(sum(counts.values())),
        "train_context_vocabulary_size": len(counts),
        "vader_full_lexicon_size": len(vader),
        "vader_lexicon_sha256": vader_hash,
        "vader_package_version": importlib.metadata.version("vaderSentiment"),
    }


def build_manifest(source: Path, split: dict, train: pd.DataFrame) -> dict:
    positive, negative, selected_rows, stats = select_external_lexicon(train)
    train_ids = list(map(str, split["train"]))
    val_ids = set(map(str, split["val"]))
    test_ids = set(map(str, split["test"]))
    used = set(train_ids)
    return {
        "version": "v04.2_vader_train_scope",
        "created_at": datetime.now().isoformat(),
        "status": "frozen_before_application_to_validation_or_test",
        "source_dataset": str(source.relative_to(ROOT)),
        "source_dataset_sha256": sha256(source),
        "split_source": str(SPLIT_PATH.relative_to(ROOT)),
        "split_sha256": sha256(SPLIT_PATH),
        "induction_scope": "train_text_only_no_labels",
        "induction_ids": train_ids,
        "induction_ids_sha256": ids_hash(train_ids),
        "induction_counts": {
            "train": len(train_ids),
            "validation_ids_used": len(used & val_ids),
            "test_ids_used": len(used & test_ids),
        },
        "parameters": {
            "context_window": CONTEXT_WINDOW,
            "minimum_frequency": MIN_FREQUENCY,
            "polarity_source": "VADER valence; labels are not read",
            "minimum_absolute_vader_valence": MIN_ABS_VADER_VALENCE,
            "maximum_absolute_context_weight": MAX_ABS_CONTEXT_WEIGHT,
            "concept_patterns": [p.pattern for p in CONCEPT_PATTERNS],
        },
        "external_polarity_resource": {
            "name": "VADER sentiment lexicon",
            "package": "vaderSentiment",
            "package_version": stats["vader_package_version"],
            "full_lexicon_sha256": stats["vader_lexicon_sha256"],
        },
        "train_scoped_external_lexicon": {
            "selection_rule": "VADER terms occurring at least three times in train concept windows",
            "selected_entries": selected_rows,
        },
        "combined_lexicon": {"positive": positive, "negative": negative},
        "statistics": {
            **stats,
            "selected_positive_count": len(positive),
            "selected_negative_count": len(negative),
            "combined_count": len(positive) + len(negative),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_GOLD)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    df = pd.read_parquet(source)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    by_id = df.assign(id=df["id"].astype(str)).set_index("id", drop=False)
    all_split_ids = [str(i) for name in ["train", "val", "test"] for i in split[name]]
    if len(all_split_ids) != len(set(all_split_ids)):
        raise ValueError("Canonical split contains duplicate or overlapping IDs")
    missing = sorted(set(all_split_ids) - set(by_id.index))
    if missing:
        raise ValueError(f"Source dataset lacks {len(missing)} canonical IDs")
    train = by_id.loc[list(map(str, split["train"]))].copy()
    manifest = build_manifest(source, split, train)
    if manifest["induction_counts"]["validation_ids_used"] or manifest["induction_counts"]["test_ids_used"]:
        raise AssertionError("Validation/test IDs entered lexicon induction")
    LEXICON_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEXICON_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.manifest_only:
        print(f"Wrote train-only manifest: {LEXICON_PATH}")
        return

    sys.path.insert(0, str(ROOT))
    from src.ontology.enricher import OntologyEnricher

    enricher = OntologyEnricher(str(ONTOLOGY_DIR), lexicon_manifest=str(LEXICON_PATH))
    base_columns = [c for c in df.columns if not c.startswith("ont_") and c != "split"]
    clean = by_id.loc[all_split_ids, base_columns].reset_index(drop=True)
    split_by_id = {str(i): name for name in ["train", "val", "test"] for i in split[name]}
    rows = []
    texts = combined_text(clean)
    for row, text in tqdm(zip(clean.to_dict("records"), texts), total=len(clean), desc="v04.2 enrichment"):
        row.update(enricher.enrich_text(text))
        row["split"] = split_by_id[str(row["id"])]
        rows.append(row)
    enriched = pd.DataFrame(rows)
    missing_features = [feature for feature in FEATURES if feature not in enriched]
    if missing_features:
        raise ValueError(f"Missing ontology features: {missing_features}")
    enriched = enriched[base_columns + FEATURES + ["split"]]
    if enriched[FEATURES].isna().any().any():
        raise ValueError("NaN values found in ontology features")
    if enriched["split"].value_counts().to_dict() != {"train": 968, "val": 323, "test": 323}:
        raise ValueError(f"Unexpected split counts: {enriched['split'].value_counts().to_dict()}")

    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(GOLD_PATH, index=False, compression="snappy")
    gold_hash = sha256(GOLD_PATH)
    ontology_nonzero = enriched[FEATURES].fillna(0).astype(float).ne(0)
    posts_with_ontology = int(ontology_nonzero.any(axis=1).sum())
    metadata = {
        "version": "v04.2_vader_train_scope",
        "created_at": datetime.now().isoformat(),
        "dataset_stats": {"total_rows": len(enriched), "total_columns": len(enriched.columns),
                          "file_size_mb": round(GOLD_PATH.stat().st_size / 1024**2, 2)},
        "split_distribution": {k: int(v) for k, v in enriched["split"].value_counts().to_dict().items()},
        "label_distribution": {str(k): int(v) for k, v in enriched["label"].value_counts().sort_index().to_dict().items()},
        "ontology_stats": {"enricher_version": "v04.2_vader_train_scope",
                           "lexicon_size": manifest["statistics"]["combined_count"],
                           "context_window": CONTEXT_WINDOW, "total_features": len(FEATURES),
                           "posts_with_ontology": posts_with_ontology,
                           "coverage_pct": round(100 * posts_with_ontology / len(enriched), 2)},
        "provenance": {"lexicon_manifest": str(LEXICON_PATH.relative_to(ROOT)),
                       "lexicon_manifest_sha256": sha256(LEXICON_PATH),
                       "split_sha256": sha256(SPLIT_PATH),
                       "induction_scope": "train_text_only_no_labels",
                       "validation_ids_used": 0, "test_ids_used": 0},
        "quality": {"no_nan_in_ontology": True, "sha256_hash": gold_hash},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "pyarrow": pyarrow.__version__,
                        "rdflib": rdflib.__version__},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "status": "passed",
        "lexicon_version": manifest["version"],
        "induction_scope": manifest["induction_scope"],
        "train_size": len(train), "validation_ids_used": 0, "test_ids_used": 0,
        "lexicon_manifest": str(LEXICON_PATH.relative_to(ROOT)),
        "lexicon_manifest_sha256": sha256(LEXICON_PATH),
        "dataset": str(GOLD_PATH.relative_to(ROOT)), "dataset_sha256": gold_hash,
        "ontology_feature_count": len(FEATURES), "selected_features_pending": True,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_PATH.with_suffix(".md").write_text(
        "# Train-only Lexicon Induction\n\n"
        f"Status: `{report['status']}`\n\n"
        f"Training records used: `{report['train_size']}`\n\n"
        "Validation IDs used: `0`\n\nTest IDs used: `0`\n\n"
        f"Combined lexicon entries: `{manifest['statistics']['combined_count']}`\n\n"
        f"Dataset SHA-256: `{gold_hash}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
