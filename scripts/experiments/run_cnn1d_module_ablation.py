#!/usr/bin/env python3
"""CNN1D ontology-module ablation for the released evidence package.

Run this script from the repository root. It auto-detects the canonical
dataset layout:

    data/gold/gold_enriched_ontology.parquet
    data/gold/GEN_split_gld_reddit_ids_v02.json

It trains CNN1D under the same ID-based canonical split and exports row-level
test predictions plus multiseed summaries for these conditions:

    BSL
    ENR_full_37
    ENR_selected
    drop_<ontology_module>

The script is intentionally limited to the numeric semantic-variable interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


SEEDS = [42, 123, 2024]
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["Negative", "Neutral", "Positive"]

EXPECTED_TEST_SUPPORT = [48, 119, 156]

DEFAULT_SELECTED_FEATURES = [
    "ont_total_negativo_mentions",
    "ont_InteligenciaArtificial_Negativo",
    "ont_domain_density",
    "ont_Robot_Negativo",
    "ont_total_positivo_mentions",
    "ont_InteligenciaArtificial_Neutro",
    "ont_Datos_Negativo",
    "ont_Innovacion_Neutro",
    "ont_Automatizacion_Negativo",
]

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
    "data": [
        "ont_Datos_Positivo",
        "ont_Datos_Negativo",
        "ont_Datos_Neutro",
    ],
    "algorithm": [
        "ont_Algoritmo_Positivo",
        "ont_Algoritmo_Negativo",
        "ont_Algoritmo_Neutro",
    ],
    "robot": [
        "ont_Robot_Positivo",
        "ont_Robot_Negativo",
        "ont_Robot_Neutro",
    ],
    "automation": [
        "ont_Automatizacion_Positivo",
        "ont_Automatizacion_Negativo",
        "ont_Automatizacion_Neutro",
    ],
    "ethics": [
        "ont_Etica_Positivo",
        "ont_Etica_Negativo",
        "ont_Etica_Neutro",
    ],
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def detect_root(explicit_root: str | None) -> Path:
    candidates = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser())
    candidates.extend(
        [
            Path.cwd(),
            Path(__file__).resolve().parents[2],
        ]
    )
    for root in candidates:
        if dataset_path(root).exists() and split_path(root).exists():
            return root.resolve()
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find canonical dataset and split. Searched:\n" + searched
    )


def dataset_path(root: Path) -> Path:
    return root / "data" / "gold" / "gold_enriched_ontology.parquet"


def split_path(root: Path) -> Path:
    return root / "data" / "gold" / "GEN_split_gld_reddit_ids_v02.json"


def load_selected_features(root: Path, df: pd.DataFrame) -> list[str]:
    candidates = [
        root / "results" / "feature_selection" / "ENR_selected_ont_features_anova_train_only.json",
        root / "ENR_selected_ont_features_anova_train_only.json",
        root / "final_predictions_v2" / "ENR_selected_ont_features_anova_train_only.json",
        root / "results" / "feature_selection" / "ENR_selected_ont_features_anova_train_only.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload.get("selected_features_train_only", payload)
        if isinstance(features, list):
            missing = [f for f in features if f not in df.columns]
            if missing:
                raise ValueError(f"Selected feature file has missing columns: {missing}")
            return features
    return [f for f in DEFAULT_SELECTED_FEATURES if f in df.columns]


def combine_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["titulo_limpio"].fillna("").astype(str)
        + ". "
        + df["texto_limpio"].fillna("").astype(str)
    ).str.strip()


def tokenize(text: str) -> list[str]:
    return str(text).lower().split()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_parts(root: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(dataset_path(root))
    split = json.loads(split_path(root).read_text(encoding="utf-8"))
    required = {"id", "label", "titulo_limpio", "texto_limpio"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    index = df.set_index("id", drop=False)
    parts: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        ids = split[name]
        missing_ids = [post_id for post_id in ids if post_id not in index.index]
        if missing_ids:
            raise ValueError(f"{name} split has IDs missing from parquet: {missing_ids[:5]}")
        parts[name] = index.loc[ids].reset_index(drop=True).copy()
        parts[name]["text_combined"] = combine_text(parts[name])

    support = parts["test"]["label"].value_counts().sort_index().reindex(CLASS_LABELS, fill_value=0).tolist()
    if support != EXPECTED_TEST_SUPPORT:
        raise ValueError(
            f"Unexpected test support {support}; expected {EXPECTED_TEST_SUPPORT}. "
            "Do not use this run for the paper."
        )

    return parts, df, split


def ontology_columns(df: pd.DataFrame) -> list[str]:
    columns = [c for c in df.columns if c.startswith("ont_")]
    return [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]


def build_conditions(include_module_only: bool) -> list[str]:
    conditions = ["BSL", "ENR_full_37", "ENR_selected"]
    conditions.extend(f"drop_{module}" for module in MODULES)
    if include_module_only:
        conditions.extend(f"module_only_{module}" for module in MODULES)
    return conditions


def condition_features(condition: str, all_features: list[str], selected: list[str]) -> list[str]:
    if condition == "BSL":
        return []
    if condition == "ENR_full_37":
        return all_features
    if condition == "ENR_selected":
        return selected
    if condition.startswith("drop_"):
        module = condition.removeprefix("drop_")
        drop = set(MODULES[module])
        return [feature for feature in all_features if feature not in drop]
    if condition.startswith("module_only_"):
        module = condition.removeprefix("module_only_")
        return [feature for feature in MODULES[module] if feature in all_features]
    raise ValueError(f"Unknown condition: {condition}")


def build_vocab(texts: pd.Series, min_freq: int, max_vocab: int) -> dict[str, int]:
    counter = Counter(token for text in texts for token in tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counter.most_common():
        if count < min_freq:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= max_vocab:
            break
    return vocab


def encode_texts(texts: pd.Series, vocab: dict[str, int], max_len: int) -> np.ndarray:
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = [vocab.get(token, 1) for token in tokenize(text)[:max_len]]
        out[i, : len(ids)] = ids
    return out


def make_ontology_arrays(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not features:
        return (
            np.zeros((len(train), 1), dtype=np.float32),
            np.zeros((len(val), 1), dtype=np.float32),
            np.zeros((len(test), 1), dtype=np.float32),
        )
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train[features].fillna(0).astype(float)).astype(np.float32)
    val_arr = scaler.transform(val[features].fillna(0).astype(float)).astype(np.float32)
    test_arr = scaler.transform(test[features].fillna(0).astype(float)).astype(np.float32)
    return train_arr, val_arr, test_arr


def make_model(vocab_size: int, ont_dim: int, args: argparse.Namespace):
    import torch.nn as nn

    class CNN1D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, args.embedding_dim, padding_idx=0)
            self.convs = nn.ModuleList(
                [nn.Conv1d(args.embedding_dim, args.num_filters, k) for k in args.kernel_sizes]
            )
            self.dropout = nn.Dropout(args.dropout)
            self.ont_dim = ont_dim
            input_dim = args.num_filters * len(args.kernel_sizes) + ont_dim
            self.fc1 = nn.Linear(input_dim, args.hidden_dim)
            self.fc2 = nn.Linear(args.hidden_dim, len(CLASS_LABELS))

        def forward(self, text_ids, ontology_values=None):
            import torch

            emb = self.embedding(text_ids).transpose(1, 2)
            pooled = [torch.max(torch.relu(conv(emb)), dim=2)[0] for conv in self.convs]
            h = torch.cat(pooled, dim=1)
            if self.ont_dim:
                h = torch.cat([h, ontology_values], dim=1)
            h = torch.relu(self.fc1(self.dropout(h)))
            return self.fc2(self.dropout(h))

    return CNN1D()


def evaluate(model, x, ont, device, batch_size: int, use_ontology: bool) -> np.ndarray:
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.long, device=device)
            ob = None
            if use_ontology:
                ob = torch.tensor(ont[start : start + batch_size], dtype=torch.float32, device=device)
            logits = model(xb, ob)
            preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
    return np.asarray(preds, dtype=np.int64)


def train_one(
    parts: dict[str, pd.DataFrame],
    condition: str,
    features: list[str],
    seed: int,
    args: argparse.Namespace,
    vocab: dict[str, int],
    max_len: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    train, val, test = parts["train"], parts["val"], parts["test"]
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(args.cuda_index)

    x_train = encode_texts(train["text_combined"], vocab, max_len)
    x_val = encode_texts(val["text_combined"], vocab, max_len)
    x_test = encode_texts(test["text_combined"], vocab, max_len)
    o_train, o_val, o_test = make_ontology_arrays(train, val, test, features)
    y_train = train["label"].to_numpy(dtype=np.int64)
    y_val = val["label"].to_numpy(dtype=np.int64)
    y_test = test["label"].to_numpy(dtype=np.int64)

    use_ontology = bool(features)
    model = make_model(len(vocab), len(features) if use_ontology else 0, args).to(device)
    class_weights = compute_class_weight("balanced", classes=np.asarray(CLASS_LABELS), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.long),
        torch.tensor(o_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    train_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for xb, ob, yb in loader:
            xb = xb.to(device)
            ob = ob.to(device) if use_ontology else None
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, ob), yb)
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        val_pred = evaluate(model, x_val, o_val, device, args.eval_batch_size, use_ontology)
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(f"No best model state captured for {condition} seed={seed}")
    model.load_state_dict(best_state)
    model.to(device)
    y_pred = evaluate(model, x_test, o_test, device, args.eval_batch_size, use_ontology)
    runtime = time.time() - train_start
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)

    row = {
        "model": "CNN1D",
        "condition": condition,
        "seed": seed,
        "n_ontology_features": len(features),
        "ontology_features": "|".join(features),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "support_negative": int(cm[0].sum()),
        "support_neutral": int(cm[1].sum()),
        "support_positive": int(cm[2].sum()),
        "f1_negative": float(f1_score(y_test == 0, y_pred == 0, zero_division=0)),
        "f1_neutral": float(f1_score(y_test == 1, y_pred == 1, zero_division=0)),
        "f1_positive": float(f1_score(y_test == 2, y_pred == 2, zero_division=0)),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "runtime_seconds": round(runtime, 4),
        "device": str(device),
        "semantic_input_profile": "train_standardized_float32" if use_ontology else "text_only",
        "classifier_head": "pooled_plus_optional_semantic_to_hidden128_to_output",
    }
    prediction_rows = [
        {
            "id": post_id,
            "model": "CNN1D",
            "condition": condition,
            "seed": seed,
            "y_true": int(y_true),
            "y_pred": int(pred),
        }
        for post_id, y_true, pred in zip(test["id"].tolist(), y_test.tolist(), y_pred.tolist())
    ]
    return row, prediction_rows


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
            runtime_seconds_mean=("runtime_seconds", "mean"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="Working repository root. Auto-detected when omitted.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <root>/results/ablation/cnn1d.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cuda-index", type=int, default=0, help="Use 0 unless you have verified another GPU is healthy.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--quick", action="store_true", help="Smoke test: seed 42 with BSL and ENR_selected only.")
    parser.add_argument("--preflight-only", action="store_true", help="Validate files, split, features, and write the manifest without training.")
    parser.add_argument("--include-module-only", action="store_true")
    parser.add_argument("--max-vocab", type=int, default=1000000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=0, help="0 means train 95th percentile capped by --max-len-cap.")
    parser.add_argument("--max-len-cap", type=int, default=100000)
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--kernel-sizes", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = detect_root(args.root)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else root / "results" / "ablation" / "cnn1d"
    out_dir.mkdir(parents=True, exist_ok=True)

    parts, df, split = load_parts(root)
    all_features = ontology_columns(df)
    selected = load_selected_features(root, df)
    missing_module_features = sorted(
        {feature for features in MODULES.values() for feature in features if feature not in df.columns}
    )
    if len(all_features) < 30:
        raise ValueError(f"Expected a broad ontology feature set, found only {len(all_features)} columns.")

    train_text = parts["train"]["text_combined"]
    vocab = build_vocab(train_text, min_freq=args.min_freq, max_vocab=args.max_vocab)
    observed_len = int(np.percentile(train_text.apply(lambda text: len(tokenize(text))), 95))
    max_len = args.max_len if args.max_len > 0 else min(max(observed_len, max(args.kernel_sizes) + 1), args.max_len_cap)
    conditions = ["BSL", "ENR_selected"] if args.quick else build_conditions(args.include_module_only)
    seeds = [42] if args.quick else args.seeds

    manifest = {
        "created_at": datetime.now().isoformat(),
        "root": ".",
        "dataset": str(dataset_path(root).relative_to(root)),
        "dataset_sha256": sha256(dataset_path(root)),
        "split": str(split_path(root).relative_to(root)),
        "split_sha256": sha256(split_path(root)),
        "lexicon_manifest": "results/lexicon/ontology_lexicon_v04_2_train_only.json",
        "lexicon_manifest_sha256": sha256(root / "results" / "lexicon" / "ontology_lexicon_v04_2_train_only.json"),
        "feature_selection_sha256": sha256(root / "results" / "feature_selection" / "ENR_selected_ont_features_anova_train_only.json"),
        "split_sizes": {name: len(split[name]) for name in ("train", "val", "test")},
        "test_support": parts["test"]["label"].value_counts().sort_index().reindex(CLASS_LABELS, fill_value=0).tolist(),
        "all_ontology_features_count": len(all_features),
        "selected_features": selected,
        "conditions": conditions,
        "seeds": seeds,
        "missing_module_features": missing_module_features,
        "hyperparameters": {
            "max_vocab": args.max_vocab,
            "min_freq": args.min_freq,
            "max_len": max_len,
            "embedding_dim": args.embedding_dim,
            "num_filters": args.num_filters,
            "kernel_sizes": args.kernel_sizes,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "semantic_input_profile": "train_standardized_float32",
            "classifier_head": "pooled_plus_optional_semantic_to_hidden128_to_output",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    write_json(out_dir / "cnn1d_module_ablation_manifest.json", manifest)

    print(f"[preflight] root={root}")
    print(f"[preflight] out_dir={out_dir}")
    print(f"[preflight] split sizes={manifest['split_sizes']}")
    print(f"[preflight] test support={manifest['test_support']}")
    print(f"[preflight] ontology features={len(all_features)} selected={len(selected)}")
    print(f"[preflight] vocab={len(vocab)} max_len={max_len}")
    if args.preflight_only:
        print("[ok] preflight completed without training")
        print(f"[ok] manifest: {out_dir / 'cnn1d_module_ablation_manifest.json'}")
        return

    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        for condition in conditions:
            features = condition_features(condition, all_features, selected)
            if condition.startswith("module_only_") and not features:
                continue
            row, preds = train_one(parts, condition, features, seed, args, vocab, max_len)
            rows.append(row)
            prediction_rows.extend(preds)
            print(
                f"[run] seed={seed} condition={condition} "
                f"macro_f1={row['macro_f1']:.4f} acc={row['accuracy']:.4f} "
                f"best_epoch={row['best_epoch']}"
            )
            write_csv(out_dir / "cnn1d_module_ablation_raw_partial.csv", rows)
            write_csv(out_dir / "cnn1d_module_ablation_predictions_partial.csv", prediction_rows)

    summary, ranking = summarize(rows)
    write_csv(out_dir / "cnn1d_module_ablation_raw.csv", rows)
    write_csv(out_dir / "cnn1d_module_ablation_summary.csv", summary)
    write_csv(out_dir / "cnn1d_module_ablation_module_rank.csv", ranking)
    write_csv(out_dir / "cnn1d_module_ablation_predictions.csv", prediction_rows)
    manifest["completed_at"] = datetime.now().isoformat()
    manifest["runtime_seconds"] = round(time.time() - t0, 4)
    manifest["output_files"] = [
        "cnn1d_module_ablation_manifest.json",
        "cnn1d_module_ablation_raw.csv",
        "cnn1d_module_ablation_summary.csv",
        "cnn1d_module_ablation_module_rank.csv",
        "cnn1d_module_ablation_predictions.csv",
    ]
    write_json(out_dir / "cnn1d_module_ablation_manifest.json", manifest)

    print("[ok] completed")
    print(f"[ok] outputs: {out_dir}")


if __name__ == "__main__":
    main()
