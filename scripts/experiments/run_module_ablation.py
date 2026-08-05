#!/usr/bin/env python3
"""Run real ontology-module ablations for LR, RF, and XGB.

Outputs are written under `results/ablation/classical/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from ke_artifact_utils import (
    ABLATION_DIR,
    MODULES,
    ensure_generated_dirs,
    load_gold,
    selected_features,
    write_csv,
    write_json,
)


SEEDS = [42, 123, 2024]
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
OUT_DIR = ABLATION_DIR / "classical"


def combine_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["titulo_limpio"].fillna("").astype(str)
        + ". "
        + df["texto_limpio"].fillna("").astype(str)
    ).str.strip()


def load_parts() -> dict[str, pd.DataFrame]:
    df = load_gold()
    return {
        split: df[df["split"] == split].copy().reset_index(drop=True)
        for split in ["train", "val", "test"]
    }


def condition_features(condition: str, all_features: list[str], selected: list[str]) -> list[str]:
    if condition == "BSL":
        return []
    if condition == "ENR_full_37":
        return all_features
    if condition == "ENR_selected":
        return selected
    if condition.startswith("drop_"):
        module = condition.removeprefix("drop_")
        drop = set(MODULES.get(module, []))
        return [f for f in all_features if f not in drop]
    if condition.startswith("module_only_"):
        module = condition.removeprefix("module_only_")
        return [f for f in MODULES.get(module, []) if f in all_features]
    raise ValueError(f"Unknown condition: {condition}")


def build_conditions(include_module_only: bool) -> list[str]:
    conditions = ["BSL", "ENR_full_37", "ENR_selected"]
    conditions.extend(f"drop_{m}" for m in MODULES)
    if include_module_only:
        conditions.extend(f"module_only_{m}" for m in MODULES)
    return conditions


def vectorize_text(train: pd.DataFrame, test: pd.DataFrame, model: str):
    max_features = 5000 if model == "LR" else 10000
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    x_train = vec.fit_transform(combine_text(train))
    x_test = vec.transform(combine_text(test))
    return x_train, x_test, max_features


def add_ontology_features(
    x_train,
    x_test,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_name: str,
):
    if not features:
        return x_train, x_test
    o_train = train[features].fillna(0).astype(float).values
    o_test = test[features].fillna(0).astype(float).values
    if model_name == "LR":
        scaler = StandardScaler()
        o_train = scaler.fit_transform(o_train)
        o_test = scaler.transform(o_test)
    return hstack([x_train, csr_matrix(o_train)]), hstack([x_test, csr_matrix(o_test)])


def semantic_input_profile(model_name: str, features: list[str]) -> str:
    if not features:
        return "text_only"
    return "train_standardized_float64" if model_name == "LR" else "raw_float64"


def make_model(model: str, seed: int):
    if model == "LR":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if model == "RF":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if model == "XGB":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric="mlogloss",
            n_jobs=1,
        )
    raise ValueError(f"Unknown model: {model}")


def fit_predict(model_name: str, estimator, x_train, y_train, x_test):
    if model_name == "XGB":
        weights = compute_class_weight("balanced", classes=np.array(CLASS_LABELS), y=y_train)
        sample_weight = pd.Series(y_train).map({i: w for i, w in enumerate(weights)}).values
        estimator.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        estimator.fit(x_train, y_train)
    return estimator.predict(x_test)


def metric_row(model: str, condition: str, seed: int, features: list[str], y_true, y_pred, seconds: float):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    per_class = {
        f"f1_{name.lower()}": float(f1_score(y_true == label, y_pred == label, zero_division=0))
        for label, name in zip(CLASS_LABELS, CLASS_NAMES)
    }
    return {
        "model": model,
        "condition": condition,
        "seed": seed,
        "n_ontology_features": len(features),
        "ontology_features": "|".join(features),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "support_negative": int(cm[0].sum()),
        "support_neutral": int(cm[1].sum()),
        "support_positive": int(cm[2].sum()),
        "runtime_seconds": round(seconds, 4),
        **per_class,
    }


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["model", "condition"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            n_ontology_features=("n_ontology_features", "first"),
            runs=("seed", "count"),
        )
        .fillna(0)
    )
    full = summary[summary["condition"] == "ENR_full_37"][["model", "macro_f1_mean"]].rename(
        columns={"macro_f1_mean": "full_macro_f1_mean"}
    )
    bsl = summary[summary["condition"] == "BSL"][["model", "macro_f1_mean"]].rename(
        columns={"macro_f1_mean": "bsl_macro_f1_mean"}
    )
    summary = summary.merge(full, on="model", how="left").merge(bsl, on="model", how="left")
    summary["delta_vs_full37"] = summary["macro_f1_mean"] - summary["full_macro_f1_mean"]
    summary["delta_vs_bsl"] = summary["macro_f1_mean"] - summary["bsl_macro_f1_mean"]

    drop = summary[summary["condition"].str.startswith("drop_")].copy()
    drop["module"] = drop["condition"].str.removeprefix("drop_")
    drop["estimated_module_contribution"] = -drop["delta_vs_full37"]
    rank = drop.sort_values(["model", "estimated_module_contribution"], ascending=[True, False])
    return summary.to_dict("records"), rank.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["LR", "RF", "XGB"], choices=["LR", "RF", "XGB"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--include-module-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run only LR seed 42 for smoke testing.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for candidate or canonical ablation artifacts.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_generated_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = load_parts()
    train, test = parts["train"], parts["test"]
    all_features = [c for c in train.columns if c.startswith("ont_")]
    selected = selected_features()
    conditions = build_conditions(args.include_module_only)
    models = ["LR"] if args.quick else args.models
    seeds = [42] if args.quick else args.seeds

    rows = []
    prediction_rows = []
    start_all = time.time()

    for model_name in models:
        for seed in seeds:
            x_train_text, x_test_text, max_features = vectorize_text(train, test, model_name)
            for condition in conditions:
                features = condition_features(condition, all_features, selected)
                if condition.startswith("module_only_") and not features:
                    continue
                t0 = time.time()
                x_train, x_test = add_ontology_features(
                    x_train_text, x_test_text, train, test, features, model_name
                )
                estimator = make_model(model_name, seed)
                y_pred = fit_predict(model_name, estimator, x_train, train["label"].values, x_test)
                seconds = time.time() - t0
                row = metric_row(
                    model_name,
                    condition,
                    seed,
                    features,
                    test["label"].values,
                    y_pred,
                    seconds,
                )
                row["tfidf_max_features"] = max_features
                row["semantic_input_profile"] = semantic_input_profile(model_name, features)
                rows.append(row)
                for post_id, y_true, pred in zip(test["id"], test["label"], y_pred):
                    prediction_rows.append(
                        {
                            "id": post_id,
                            "model": model_name,
                            "condition": condition,
                            "seed": seed,
                            "y_true": int(y_true),
                            "y_pred": int(pred),
                        }
                    )
                print(
                    f"{model_name} seed={seed} {condition}: "
                    f"macro_f1={row['macro_f1']:.4f} acc={row['accuracy']:.4f}"
                )

    summary, ranking = summarize(rows)
    write_csv(out_dir / "module_ablation_raw.csv", rows)
    write_csv(out_dir / "module_ablation_summary.csv", summary)
    write_csv(out_dir / "module_ablation_module_rank.csv", ranking)
    write_csv(out_dir / "module_ablation_predictions.csv", prediction_rows)

    payload = {
        "status": "passed",
        "created_at": datetime.now().isoformat(),
        "experiment": "LR/RF/XGB ontology module ablation",
        "models": models,
        "seeds": seeds,
        "conditions": conditions,
        "dataset": "data/gold/gold_enriched_ontology.parquet",
        "split": "data/gold/GEN_split_gld_reddit_ids_v02.json",
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "ontology_feature_count": len(all_features),
        "selected_feature_count": len(selected),
        "semantic_input_profiles": {
            "LR": "train_standardized_float64",
            "RF": "raw_float64",
            "XGB": "raw_float64",
        },
        "runtime_seconds": round(time.time() - start_all, 4),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "interpretation_limits": [
            "This is a model-side ablation of semantic variables, not causal evidence.",
            "CNN1D is evaluated separately on the same canonical dataset.",
            "Module contribution is interpreted through delta versus full 37-feature ENR.",
        ],
    }
    write_json(out_dir / "module_ablation_summary.json", payload)

    md = [
        "# Module Ablation Summary",
        "",
        "Status: `passed` for LR/RF/XGB local ablation.",
        "",
        f"Models: `{', '.join(models)}`",
        f"Seeds: `{', '.join(map(str, seeds))}`",
        f"Conditions: `{len(conditions)}`",
        f"Runtime seconds: `{payload['runtime_seconds']}`",
        "",
        "## Interpretation Limits",
        "",
    ]
    md.extend(f"- {item}" for item in payload["interpretation_limits"])
    md.extend(["", "## Outputs", ""])
    for name in [
        "module_ablation_raw.csv",
        "module_ablation_summary.csv",
        "module_ablation_module_rank.csv",
        "module_ablation_predictions.csv",
        "module_ablation_summary.json",
    ]:
        md.append(f"- `{name}`")
    (out_dir / "module_ablation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Wrote real LR/RF/XGB module ablation outputs.")


if __name__ == "__main__":
    main()
