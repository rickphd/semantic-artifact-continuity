#!/usr/bin/env python3
"""Build the public artifact inventory and SHA-256 manifest."""

from __future__ import annotations

from pathlib import Path

from ke_artifact_utils import (
    ANOVA_DIR,
    CANONICAL_ONTOLOGY_DIR,
    GOLD_METADATA,
    GOLD_PATH,
    MODEL_INPUT_DIR,
    PROVENANCE_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    SENSITIVITY_DIR,
    SPLIT_PATH,
    TRACEABILITY_DIR,
    ensure_generated_dirs,
    rel,
    sha256,
    write_json,
)


SCAN_ROOTS = [
    REPO_ROOT / "data" / "gold",
    REPO_ROOT / "results" / "lexicon",
    REPO_ROOT / "results" / "feature_selection",
    REPO_ROOT / "docs",
    REPO_ROOT / "figures",
    REPO_ROOT / "src",
    REPO_ROOT / "scripts" / "experiments",
    REPO_ROOT / "scripts" / "figures",
    REPO_ROOT / "scripts" / "release",
    RESULTS_DIR,
    CANONICAL_ONTOLOGY_DIR,
]


EXCLUDED_PARTS = {
    ".DS_Store",
    "__pycache__",
    "archive",
}

EXCLUDED_ROOTS: list[Path] = []

# These files describe a verification run and therefore cannot consistently
# hash themselves while being written.  The reproducibility manifest hashes
# the inventory and hash manifest after this script completes.
CONTROL_OUTPUTS = {
    PROVENANCE_DIR / "artifact_inventory.json",
    PROVENANCE_DIR / "artifact_inventory.md",
    PROVENANCE_DIR / "hash_manifest.json",
    PROVENANCE_DIR / "reproducibility_manifest.json",
    PROVENANCE_DIR / "reproducibility_report.md",
    PROVENANCE_DIR / "figure_generation_manifest.json",
}

ACTIVE_SENSITIVITY_SUMMARIES = [
    SENSITIVITY_DIR / "selected_feature_stability.csv",
    SENSITIVITY_DIR / "vader_ke_sensitivity_aggregate_stats.json",
    SENSITIVITY_DIR / "vader_ke_sensitivity_manifest.json",
    SENSITIVITY_DIR / "vader_ke_sensitivity_summary.csv",
    SENSITIVITY_DIR / "vader_ke_sensitivity_summary.json",
    SENSITIVITY_DIR / "vader_ke_sensitivity_summary.md",
]


REQUIRED = {
    GOLD_PATH: "gold enriched dataset",
    GOLD_METADATA: "gold metadata",
    SPLIT_PATH: "canonical split",
    CANONICAL_ONTOLOGY_DIR / "rr-core.ttl": "canonical core ontology",
    CANONICAL_ONTOLOGY_DIR / "rr-domain.ttl": "canonical domain ontology",
    CANONICAL_ONTOLOGY_DIR / "rr-sentiment.ttl": "canonical sentiment ontology",
    CANONICAL_ONTOLOGY_DIR / "rr-shapes.ttl": "canonical SHACL shapes",
    REPO_ROOT / "results" / "feature_selection" / "ENR_selected_ont_features_anova_train_only.json": "train only selected semantic variables",
    TRACEABILITY_DIR / "chain_reconciliation_report.json": "evidence chain reconciliation",
    MODEL_INPUT_DIR / "selected_semantic_input_matrix.csv": "canonical selected semantic input matrix",
    MODEL_INPUT_DIR / "selected_semantic_input_cell_trace.csv": "cell-level selected input traceability",
    MODEL_INPUT_DIR / "downstream_semantic_input_fingerprints.csv": "downstream semantic input fingerprints",
    MODEL_INPUT_DIR / "selected_semantic_input_reconciliation_report.json": "selected input cell reconciliation",
    ANOVA_DIR / "models" / "LR_ENR_scaler_v2.joblib": "saved semantic-input scaler",
}

RELEASED_MODEL = ANOVA_DIR / "models" / "LR_ENR_scaler_v2.joblib"


def role_for(path: Path) -> str:
    s = rel(path)
    if s.endswith(".ttl"):
        return "ontology_or_shacl"
    if "gold_enriched_ontology.parquet" in s:
        return "semantic_feature_table"
    if "GEN_split" in s:
        return "split"
    if "feature_selection" in s:
        return "semantic_variable_selection"
    if "metrics" in s or s.endswith("_metrics.json"):
        return "metric"
    if "predictions" in s:
        return "prediction"
    if "/docs/" in s or s.endswith(".md"):
        return "documentation"
    return "supporting_artifact"


def include_path(path: Path) -> bool:
    resolved = path.resolve()
    if resolved in {item.resolve() for item in CONTROL_OUTPUTS}:
        return False
    if any(resolved.is_relative_to(root.resolve()) for root in EXCLUDED_ROOTS):
        return False
    parts = set(path.parts)
    if parts & EXCLUDED_PARTS:
        return False
    if path.suffix == ".pyc" or path.suffix in {".pt", ".pth", ".safetensors", ".ckpt"}:
        return False
    if path.suffix == ".joblib" and resolved != RELEASED_MODEL.resolve():
        return False
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    ensure_generated_dirs()
    artifacts = []
    seen: set[Path] = set()

    def add_artifact(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not include_path(path):
            return
        seen.add(resolved)
        artifacts.append(
            {
                "path": rel(path),
                "role": role_for(path),
                "size_bytes": path.stat().st_size,
                "hash_sha256": sha256(path),
                "regeneration_status": "stored_or_scripted",
            }
        )

    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            add_artifact(path)
    for path in ACTIVE_SENSITIVITY_SUMMARIES:
        if path.is_file() and path.stat().st_size > 0:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                artifacts.append(
                    {
                        "path": rel(path),
                        "role": role_for(path),
                        "size_bytes": path.stat().st_size,
                        "hash_sha256": sha256(path),
                        "regeneration_status": "stored_or_scripted",
                    }
                )

    missing = []
    for path, reason in REQUIRED.items():
        if not path.exists():
            missing.append({"path": rel(path), "reason": reason})

    payload = {
        "status": "passed" if not missing else "blocked",
        "scope": {
            "active_lineage": "v04.2",
            "excluded_paths": [rel(root) for root in EXCLUDED_ROOTS],
            "sensitivity_policy": "Hashes the six aggregate sensitivity summaries; variant-level intermediates are referenced by the sensitivity manifest and excluded from the canonical inventory count.",
            "control_output_policy": "Inventory and reproducibility reports do not hash themselves; the reproducibility manifest records the inventory and hash-manifest hashes after generation.",
        },
        "artifact_count": len(artifacts),
        "missing_required": missing,
        "canonical_ontology_decision": {
            "selected": rel(CANONICAL_ONTOLOGY_DIR),
            "reason": "The public ontology directory is the canonical RDF/SHACL source used by the release scripts.",
        },
        "artifacts": artifacts,
    }
    write_json(PROVENANCE_DIR / "artifact_inventory.json", payload)
    write_json(
        PROVENANCE_DIR / "hash_manifest.json",
        [{"path": item["path"], "hash_sha256": item["hash_sha256"]} for item in artifacts],
    )

    md = [
        "# Public Artifact Inventory",
        "",
        f"Status: `{payload['status']}`",
        f"Artefacts hashed: `{len(artifacts)}`",
        f"Missing required artefacts: `{len(missing)}`",
        "",
        "## Scope",
        "",
        "Active lineage: `v04.2`.",
        "",
        "- The sensitivity inventory includes six aggregate summaries; the 125 variant-level intermediate packages can be regenerated and are not distributed.",
        "- Inventory and reproducibility reports do not hash themselves. Their final hashes are recorded by the reproducibility manifest after inventory generation.",
        "",
        "## Canonical Ontology Decision",
        "",
        payload["canonical_ontology_decision"]["reason"],
        "",
        "## Required Artefact Check",
        "",
    ]
    if missing:
        for item in missing:
            md.append(f"- Missing `{item['path']}`: {item['reason']}")
    else:
        md.append("- All required artefacts are present.")
    md.extend(["", "## Role Counts", ""])
    counts = {}
    for item in artifacts:
        counts[item["role"]] = counts.get(item["role"], 0) + 1
    for role, count in sorted(counts.items()):
        md.append(f"- `{role}`: {count}")
    (PROVENANCE_DIR / "artifact_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {rel(PROVENANCE_DIR / 'artifact_inventory.json')} from {REPO_ROOT}")


if __name__ == "__main__":
    main()
