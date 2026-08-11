#!/usr/bin/env python3
"""Analyze ontology feature coverage, sparsity, and module activation."""

from __future__ import annotations

import math

import pandas as pd

from ke_artifact_utils import COVERAGE_DIR, MODULES, ensure_generated_dirs, feature_to_module, load_gold, ontology_columns, selected_features, write_csv, write_json


def entropy(values: pd.Series) -> float:
    counts = values.value_counts(normalize=True)
    return float(-(counts * counts.map(lambda p: math.log2(p))).sum()) if len(counts) else 0.0


def main() -> None:
    ensure_generated_dirs()
    df = load_gold()
    ont_cols = ontology_columns(df)
    selected = set(selected_features())

    feature_rows = []
    for col in ont_cols:
        s = df[col].fillna(0).astype(float)
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
                "missing_count": int(df[col].isna().sum()),
            }
        )
    write_csv(COVERAGE_DIR / "coverage_sparsity.csv", feature_rows)

    module_rows = []
    for module, cols in MODULES.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        active = df[present].fillna(0).astype(float).abs().sum(axis=1) > 0
        row = {
            "module": module,
            "feature_count": len(present),
            "active_posts": int(active.sum()),
            "active_pct": float(active.mean() * 100),
            "selected_feature_count": sum(c in selected for c in present),
            "label_entropy_when_active": entropy(df.loc[active, "label"]) if active.any() else 0.0,
        }
        for split in ["train", "val", "test"]:
            split_active = active[df["split"] == split]
            row[f"{split}_active_pct"] = float(split_active.mean() * 100) if len(split_active) else 0.0
        module_rows.append(row)
    write_csv(COVERAGE_DIR / "module_coverage.csv", module_rows)

    overall_active = df[ont_cols].fillna(0).astype(float).abs().sum(axis=1) > 0
    selected_active = df[list(selected)].fillna(0).astype(float).abs().sum(axis=1) > 0
    report = {
        "status": "passed",
        "rows": int(len(df)),
        "ontology_feature_count": len(ont_cols),
        "selected_feature_count": len(selected),
        "posts_with_any_ontology_feature": int(overall_active.sum()),
        "overall_activation_pct": float(overall_active.mean() * 100),
        "posts_with_selected_feature": int(selected_active.sum()),
        "selected_activation_pct": float(selected_active.mean() * 100),
        "split_activation_pct": {
            split: float(overall_active[df["split"] == split].mean() * 100)
            for split in ["train", "val", "test"]
        },
    }
    write_json(COVERAGE_DIR / "coverage_sparsity_report.json", report)

    md = [
        "# Coverage And Sparsity Report",
        "",
        f"Rows: `{report['rows']}`",
        f"Ontology variables: `{report['ontology_feature_count']}`",
        f"Selected train only variables: `{report['selected_feature_count']}`",
        f"Posts with any ontology activation: `{report['posts_with_any_ontology_feature']}` ({report['overall_activation_pct']:.2f}%)",
        f"Posts with selected variable activation: `{report['posts_with_selected_feature']}` ({report['selected_activation_pct']:.2f}%)",
        "",
        "## Split Activation",
        "",
    ]
    for split, pct in report["split_activation_pct"].items():
        md.append(f"- `{split}`: {pct:.2f}%")
    md.extend(["", "## Module Coverage", ""])
    for row in module_rows:
        md.append(f"- `{row['module']}`: {row['active_posts']} posts ({row['active_pct']:.2f}%), selected features {row['selected_feature_count']}")
    (COVERAGE_DIR / "coverage_sparsity_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Overall ontology activation: {report['overall_activation_pct']:.2f}%")


if __name__ == "__main__":
    main()
