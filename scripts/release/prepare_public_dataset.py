#!/usr/bin/env python3
"""Create the minimized research-release dataset from the internal Gold table.

The release retains only fields required to reproduce enrichment, training,
RDF materialization, and evaluation. Reddit account names are replaced by
stable corpus-local identifiers; the mapping is not written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TEXT_COLUMNS = ["titulo", "texto", "titulo_limpio", "texto_limpio"]
RDF_COLUMNS = ["subreddit_nombre", "fecha_creacion"]
TARGET_COLUMNS = ["label", "split"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    metadata_path = args.metadata.resolve()
    frame = pd.read_parquet(source)

    ontology_columns = sorted(column for column in frame if column.startswith("ont_"))
    required = {
        "id",
        "autor",
        *TEXT_COLUMNS,
        *RDF_COLUMNS,
        *TARGET_COLUMNS,
        *ontology_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")
    if len(ontology_columns) != 37:
        raise ValueError(f"Expected 37 ontology variables, found {len(ontology_columns)}")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("Post IDs must be unique")

    author_values = frame["autor"].fillna("unknown").astype(str)
    author_lookup = {
        author: f"author_{index:04d}"
        for index, author in enumerate(sorted(author_values.unique()), start=1)
    }

    release = frame[
        ["id", *TEXT_COLUMNS, *RDF_COLUMNS, *TARGET_COLUMNS, *ontology_columns]
    ].copy()
    release.insert(1, "author_id", author_values.map(author_lookup))
    release["id"] = release["id"].astype(str)

    split_counts = release["split"].value_counts().to_dict()
    expected_splits = {"train": 968, "val": 323, "test": 323}
    if split_counts != expected_splits:
        raise ValueError(f"Unexpected split counts: {split_counts}")

    output.parent.mkdir(parents=True, exist_ok=True)
    release.to_parquet(output, index=False, compression="snappy")

    metadata = {
        "release_version": "v04.2-public",
        "rows": int(len(release)),
        "columns": int(len(release.columns)),
        "ontology_variable_count": len(ontology_columns),
        "split_distribution": {key: int(value) for key, value in split_counts.items()},
        "label_distribution": {
            str(key): int(value)
            for key, value in release["label"].value_counts().sort_index().items()
        },
        "author_pseudonym_count": len(author_lookup),
        "privacy_transform": {
            "retained": [
                "post ID",
                "post title and body",
                "cleaned title and body",
                "subreddit",
                "creation timestamp",
                "label",
                "ontology variables",
                "split",
            ],
            "removed": sorted(set(frame.columns) - set(release.columns)),
            "author_policy": (
                "Reddit account names are replaced by deterministic corpus-local "
                "ordinal identifiers. The source-to-release mapping is not exported."
            ),
        },
        "source_sha256": sha256(source),
        "release_sha256": sha256(output),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
