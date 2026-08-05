#!/usr/bin/env python3
"""Create the public reproducibility manifest from released outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ke_artifact_utils import (
    ABLATION_DIR,
    ANOVA_DIR,
    COVERAGE_DIR,
    KNOWLEDGE_GRAPH_DIR,
    MODEL_INPUT_DIR,
    PROVENANCE_DIR,
    REPO_ROOT,
    SHACL_DIR,
    TRACEABILITY_DIR,
    VALIDATION_DIR,
    rel,
    sha256,
    write_json,
)


REQUIRED_OUTPUTS = [
    PROVENANCE_DIR / "artifact_inventory.json",
    PROVENANCE_DIR / "hash_manifest.json",
    REPO_ROOT / "results" / "lexicon" / "lexicon_induction_report.json",
    KNOWLEDGE_GRAPH_DIR / "posts.ttl",
    TRACEABILITY_DIR / "traceability_map.csv",
    KNOWLEDGE_GRAPH_DIR / "materialization_report.json",
    SHACL_DIR / "baseline_report.json",
    COVERAGE_DIR / "coverage_sparsity_report.md",
    MODEL_INPUT_DIR / "model_input_trace.csv",
    MODEL_INPUT_DIR / "semantic_variable_model_input_report.md",
    MODEL_INPUT_DIR / "selected_semantic_input_matrix.csv",
    MODEL_INPUT_DIR / "selected_semantic_input_standardized.csv",
    MODEL_INPUT_DIR / "selected_semantic_input_cell_trace.csv",
    MODEL_INPUT_DIR / "downstream_semantic_input_fingerprints.csv",
    MODEL_INPUT_DIR / "selected_semantic_input_reconciliation_report.json",
    MODEL_INPUT_DIR / "selected_semantic_input_reconciliation_report.md",
    TRACEABILITY_DIR / "chain_reconciliation_report.json",
    TRACEABILITY_DIR / "chain_reconciliation_report.md",
    ABLATION_DIR / "classical" / "module_ablation_summary.md",
    ABLATION_DIR / "classical" / "module_ablation_predictions.csv",
    ABLATION_DIR / "cnn1d" / "cnn1d_module_ablation_manifest.json",
    ABLATION_DIR / "cnn1d" / "cnn1d_module_ablation_summary.csv",
    ABLATION_DIR / "cnn1d" / "cnn1d_module_ablation_predictions.csv",
    ANOVA_DIR / "metrics" / "multiseed_summary_v2.csv",
    ANOVA_DIR / "metrics" / "multiseed_raw_v2.csv",
    ANOVA_DIR / "README.md",
    VALIDATION_DIR / "experiment_chain_validation.json",
]


def main() -> None:
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    validation = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "experiments" / "validate_v04_2_experiment_chain.py")],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    hashes = [{"path": rel(p), "hash_sha256": sha256(p)} for p in REQUIRED_OUTPUTS if p.exists()]
    payload = {
        "status": "passed" if not missing and validation.returncode == 0 else "attention_required",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "project_root": ".",
        "generated_required_count": len(REQUIRED_OUTPUTS),
        "missing_generated_outputs": [rel(p) for p in missing],
        "v04_2_experiment_chain_validation": {
            "returncode": validation.returncode,
            "stdout": validation.stdout.strip(),
            "stderr": validation.stderr.strip() if validation.returncode else "",
        },
        "hashes": hashes,
        "release_scope_exclusions": [
            "non-authoritative experiment outputs",
            "accelerator-specific diagnostic outputs",
            "serialized trained model binaries other than the released input scaler",
        ],
        "known_non_reproducible_steps": [],
    }
    write_json(PROVENANCE_DIR / "reproducibility_manifest.json", payload)
    md = [
        "# Reproducibility Report",
        "",
        f"Status: `{payload['status']}`",
        f"Generated outputs checked: `{len(REQUIRED_OUTPUTS)}`",
        f"Missing outputs: `{len(missing)}`",
        f"v04.2 chain validation: `{validation.stdout.strip() or validation.stderr.strip()}`",
        "",
        "## Release Scope Exclusions",
        "",
    ]
    md.extend(f"- `{item}`" for item in payload["release_scope_exclusions"])
    md.extend([
        "",
        "## Known Non Reproducible Or Partial Steps",
        "",
    ])
    md.extend(f"- {item}" for item in payload["known_non_reproducible_steps"])
    (PROVENANCE_DIR / "reproducibility_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(payload["status"])


if __name__ == "__main__":
    main()
