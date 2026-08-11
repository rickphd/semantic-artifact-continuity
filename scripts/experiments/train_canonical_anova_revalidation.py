"""
ANOVA train-only revalidation (LR, RF, XGB, CNN1D x BSL/ENR x multi-seed).

Principios:
- Split por ID (GEN_split_gld_reddit_ids_v02.json), merge por `id`, nunca posicional.
- TF-IDF y escaladores con `fit` SOLO en train.
- Features ontologicas: todas las variables con ANOVA p<0.05 en train.
- Evaluacion en test UNA vez, exportando predicciones y metricas en la misma corrida.
- Outputs are written under `results/anova_revalidation/`.

Uso:
    python scripts/experiments/train_canonical_anova_revalidation.py
    python scripts/experiments/train_canonical_anova_revalidation.py --quick
"""

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import xgboost
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from model_input_provenance import matrix_fingerprint

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold" / "gold_enriched_ontology.parquet"
SPLIT_PATH = ROOT / "data" / "gold" / "GEN_split_gld_reddit_ids_v02.json"
FEAT_PATH = ROOT / "results" / "feature_selection" / "ENR_selected_ont_features_anova_train_only.json"
LEXICON_PATH = ROOT / "results" / "lexicon" / "ontology_lexicon_v04_2_train_only.json"
OUT = ROOT / "results" / "anova_revalidation"
PRED_DIR = OUT / "predictions"
MET_DIR = OUT / "metrics"
MODEL_DIR = OUT / "models"

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
SEEDS = [42, 123, 2024]
PRIMARY_SEED = 42


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_input_provenance(model, ont_features, matrices):
    profile = {
        "LR": "train_standardized_float64",
        "RF": "raw_float64",
        "XGB": "raw_float64",
        "CNN1D": "train_standardized_float32",
    }[model]
    return {
        split_name: matrix_fingerprint(
            part["id"].astype(str).tolist(),
            ont_features,
            matrices[split_name],
            profile,
        )
        for split_name, part in matrices["parts"].items()
    }


def combine_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["titulo_limpio"].fillna("").astype(str)
        + ". "
        + df["texto_limpio"].fillna("").astype(str)
    ).str.strip()


def load_data():
    gold = pd.read_parquet(GOLD_PATH)
    split = json.load(open(SPLIT_PATH))
    feat = json.load(open(FEAT_PATH))
    ont9 = feat["selected_features_train_only"]

    parts = {}
    for name in ("train", "val", "test"):
        ids = split[name]
        sub = gold.set_index("id").loc[ids].reset_index()  # orden determinista por lista de IDs
        assert len(sub) == len(ids)
        parts[name] = sub
    assert parts["test"]["label"].value_counts().sort_index().tolist() == [48, 119, 156]
    return parts, ont9


def metric_payload(model, condition, seed, y_true, y_pred, extra):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "model": model,
        "condition": condition,
        "seed": seed,
        "pipeline_version": "canonical_v04_2_vader_train_scope",
        "created_at": datetime.now().isoformat(),
        "split_source": "GEN_split_gld_reddit_ids_v02.json",
        "dataset": "gold_enriched_ontology.parquet",
        "dataset_sha256": sha256(GOLD_PATH),
        "lexicon_manifest": str(LEXICON_PATH.relative_to(ROOT)),
        "lexicon_manifest_sha256": sha256(LEXICON_PATH),
        "test_size": int(len(y_true)),
        "class_order": CLASS_NAMES,
        "support": cm.sum(axis=1).tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "torch": torch.__version__,
        },
        **extra,
    }


def save_run(model, condition, seed, test_df, y_pred, extra, suffix=""):
    tag = f"{model}_{condition}{suffix}"
    fname = f"{tag}_seed{seed}"
    pred = test_df[["id"]].copy()
    pred["split"] = "test"
    pred["y_true"] = test_df["label"].values
    pred["y_pred"] = y_pred
    pred["model"] = model
    pred["condition"] = condition
    pred["seed"] = seed
    pred.to_csv(PRED_DIR / f"{fname}_test_predictions_v2.csv", index=False)

    payload = metric_payload(model, condition, seed, pred["y_true"].values, y_pred, extra)
    with open(MET_DIR / f"{fname}_test_metrics_v2.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(
        f"  [{fname}] macro_f1={payload['macro_f1']:.4f} acc={payload['accuracy']:.4f} "
        f"support={payload['support']}"
    )
    return payload


# ----------------------------- ML clasico -----------------------------


def run_lr(parts, ont9, seed, ont_list_name="anova_p005_train_only_vader_scope"):
    tr, te = parts["train"], parts["test"]
    res = {}
    # BSL
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(combine_text(tr))
    Xte = vec.transform(combine_text(te))
    mod = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    mod.fit(Xtr, tr["label"])
    extra = {
        "hyperparameters": {
            "tfidf_max_features": 5000,
            "ngram_range": [1, 2],
            "max_iter": 1000,
            "class_weight": "balanced",
        },
        "n_features": Xtr.shape[1],
    }
    res["BSL"] = save_run("LR", "BSL", seed, te, mod.predict(Xte), extra)
    if seed == PRIMARY_SEED:
        joblib.dump(mod, MODEL_DIR / "LR_BSL_v2.joblib")
        joblib.dump(vec, MODEL_DIR / "LR_BSL_vec_v2.joblib")

    # ENR
    scaler = StandardScaler()
    Otr = scaler.fit_transform(tr[ont9].fillna(0).astype(float).values)
    Ote = scaler.transform(te[ont9].fillna(0).astype(float).values)
    Xtr_e = hstack([Xtr, csr_matrix(Otr)])
    Xte_e = hstack([Xte, csr_matrix(Ote)])
    mod_e = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    mod_e.fit(Xtr_e, tr["label"])
    extra_e = {
        "hyperparameters": extra["hyperparameters"],
        "ont_features": ont9,
        "ont_list": ont_list_name,
        "n_features": Xtr_e.shape[1],
        "semantic_input_provenance": semantic_input_provenance(
            "LR",
            ont9,
            {"parts": {"train": tr, "test": te}, "train": Otr, "test": Ote},
        ),
    }
    suffix = "" if ont_list_name == "anova_p005_train_only_vader_scope" else "_alternate"
    res["ENR"] = save_run("LR", "ENR", seed, te, mod_e.predict(Xte_e), extra_e, suffix)
    if seed == PRIMARY_SEED and ont_list_name == "anova_p005_train_only_vader_scope":
        joblib.dump(mod_e, MODEL_DIR / "LR_ENR_v2.joblib")
        joblib.dump(vec, MODEL_DIR / "LR_ENR_vec_v2.joblib")
        joblib.dump(scaler, MODEL_DIR / "LR_ENR_scaler_v2.joblib")
    return res


def run_rf(parts, ont9, seed, ont_list_name="anova_p005_train_only_vader_scope"):
    tr, te = parts["train"], parts["test"]
    res = {}
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(combine_text(tr))
    Xte = vec.transform(combine_text(te))
    hp = {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "tfidf_max_features": 10000,
        "ngram_range": [1, 2],
    }
    mod = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    mod.fit(Xtr, tr["label"])
    res["BSL"] = save_run(
        "RF", "BSL", seed, te, mod.predict(Xte), {"hyperparameters": hp, "n_features": Xtr.shape[1]}
    )
    if seed == PRIMARY_SEED:
        joblib.dump(mod, MODEL_DIR / "RF_BSL_v2.joblib")
        joblib.dump(vec, MODEL_DIR / "RF_BSL_vec_v2.joblib")

    Otr = tr[ont9].fillna(0).astype(float).values
    Ote = te[ont9].fillna(0).astype(float).values
    Xtr_e = hstack([Xtr, csr_matrix(Otr)])
    Xte_e = hstack([Xte, csr_matrix(Ote)])
    mod_e = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    mod_e.fit(Xtr_e, tr["label"])
    extra_e = {
        "hyperparameters": hp,
        "ont_features": ont9,
        "ont_list": ont_list_name,
        "n_features": Xtr_e.shape[1],
        "semantic_input_provenance": semantic_input_provenance(
            "RF",
            ont9,
            {"parts": {"train": tr, "test": te}, "train": Otr, "test": Ote},
        ),
    }
    suffix = "" if ont_list_name == "anova_p005_train_only_vader_scope" else "_alternate"
    res["ENR"] = save_run("RF", "ENR", seed, te, mod_e.predict(Xte_e), extra_e, suffix)
    if seed == PRIMARY_SEED and ont_list_name == "anova_p005_train_only_vader_scope":
        joblib.dump(mod_e, MODEL_DIR / "RF_ENR_v2.joblib")
        joblib.dump(vec, MODEL_DIR / "RF_ENR_vec_v2.joblib")
    return res


def run_xgb(parts, ont9, seed, ont_list_name="anova_p005_train_only_vader_scope"):
    tr, te = parts["train"], parts["test"]
    res = {}
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(combine_text(tr))
    Xte = vec.transform(combine_text(te))
    hp = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tfidf_max_features": 10000,
        "ngram_range": [1, 2],
    }
    # pesos de clase balanceados (XGB no tiene class_weight nativo multi-clase)
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=tr["label"])
    sw_tr = tr["label"].map({i: w for i, w in enumerate(cw)}).values

    mod = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="mlogloss",
        n_jobs=1,
    )
    mod.fit(Xtr, tr["label"], sample_weight=sw_tr)
    res["BSL"] = save_run(
        "XGB",
        "BSL",
        seed,
        te,
        mod.predict(Xte),
        {
            "hyperparameters": hp,
            "n_features": Xtr.shape[1],
            "class_weighting": "balanced_sample_weight",
        },
    )
    if seed == PRIMARY_SEED:
        joblib.dump(mod, MODEL_DIR / "XGB_BSL_v2.joblib")
        joblib.dump(vec, MODEL_DIR / "XGB_BSL_vec_v2.joblib")

    Otr = tr[ont9].fillna(0).astype(float).values
    Ote = te[ont9].fillna(0).astype(float).values
    Xtr_e = hstack([Xtr, csr_matrix(Otr)])
    Xte_e = hstack([Xte, csr_matrix(Ote)])
    mod_e = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="mlogloss",
        n_jobs=1,
    )
    mod_e.fit(Xtr_e, tr["label"], sample_weight=sw_tr)
    extra_e = {
        "hyperparameters": hp,
        "ont_features": ont9,
        "ont_list": ont_list_name,
        "n_features": Xtr_e.shape[1],
        "class_weighting": "balanced_sample_weight",
        "semantic_input_provenance": semantic_input_provenance(
            "XGB",
            ont9,
            {"parts": {"train": tr, "test": te}, "train": Otr, "test": Ote},
        ),
    }
    suffix = "" if ont_list_name == "anova_p005_train_only_vader_scope" else "_alternate"
    res["ENR"] = save_run("XGB", "ENR", seed, te, mod_e.predict(Xte_e), extra_e, suffix)
    if seed == PRIMARY_SEED and ont_list_name == "anova_p005_train_only_vader_scope":
        joblib.dump(mod_e, MODEL_DIR / "XGB_ENR_v2.joblib")
        joblib.dump(vec, MODEL_DIR / "XGB_ENR_vec_v2.joblib")
    return res


# ----------------------------- CNN 1D -----------------------------


class CNN1D(nn.Module):
    def __init__(
        self,
        vocab_size,
        ont_dim=0,
        embedding_dim=300,
        num_filters=128,
        kernel_sizes=(3, 4, 5),
        hidden_dim=128,
        num_classes=3,
        dropout=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.ont_dim = ont_dim
        feat_dim = num_filters * len(kernel_sizes) + ont_dim
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, ont=None):
        emb = self.embedding(x).transpose(1, 2)
        pooled = [torch.max(torch.relu(c(emb)), dim=2)[0] for c in self.convs]
        h = torch.cat(pooled, dim=1)
        if self.ont_dim:
            h = torch.cat([h, ont], dim=1)
        h = torch.relu(self.fc1(self.dropout(h)))
        return self.fc2(self.dropout(h))


def build_vocab(texts, min_freq=2):
    counter = Counter(t for txt in texts for t in str(txt).lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, c in counter.most_common():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode_batch(texts, vocab, max_len):
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, txt in enumerate(texts):
        toks = str(txt).lower().split()[:max_len]
        out[i, : len(toks)] = [vocab.get(t, 1) for t in toks]
    return out


def run_cnn(parts, ont9, seed, condition, device_name="auto"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device_name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        device = torch.device(device_name)

    tr, va, te = parts["train"], parts["val"], parts["test"]
    txt_tr, txt_va, txt_te = combine_text(tr), combine_text(va), combine_text(te)
    vocab = build_vocab(txt_tr)  # vocab SOLO de train
    lens = txt_tr.str.split().str.len()
    max_len = int(np.percentile(lens, 95))

    Xtr = torch.tensor(encode_batch(txt_tr.tolist(), vocab, max_len))
    Xva = torch.tensor(encode_batch(txt_va.tolist(), vocab, max_len))
    Xte = torch.tensor(encode_batch(txt_te.tolist(), vocab, max_len))
    ytr = torch.tensor(tr["label"].values)
    yva = torch.tensor(va["label"].values)

    use_ont = condition == "ENR"
    if use_ont:
        scaler = StandardScaler()
        Otr = torch.tensor(
            scaler.fit_transform(tr[ont9].fillna(0).astype(float)), dtype=torch.float32
        )
        Ova = torch.tensor(scaler.transform(va[ont9].fillna(0).astype(float)), dtype=torch.float32)
        Ote = torch.tensor(scaler.transform(te[ont9].fillna(0).astype(float)), dtype=torch.float32)
    else:
        Otr = Ova = Ote = torch.zeros(0)

    model = CNN1D(len(vocab), ont_dim=len(ont9) if use_ont else 0).to(device)
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=tr["label"])
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    ds = TensorDataset(Xtr, Otr if use_ont else torch.zeros(len(Xtr), 1), ytr)
    dl = DataLoader(ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))

    def evaluate(X, O):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                xb = X[i : i + 128].to(device)
                ob = O[i : i + 128].to(device) if use_ont else None
                preds.extend(torch.argmax(model(xb, ob), 1).cpu().numpy())
        return np.array(preds)

    best_f1, best_state, patience, bad, best_epoch = -1, None, 5, 0, 0
    for epoch in range(1, 21):
        model.train()
        for xb, ob, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            ob = ob.to(device) if use_ont else None
            opt.zero_grad()
            loss = crit(model(xb, ob), yb)
            loss.backward()
            opt.step()
        val_f1 = f1_score(yva, evaluate(Xva, Ova), average="macro")
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    y_pred = evaluate(Xte, Ote)
    extra = {
        "hyperparameters": {
            "embedding_dim": 300,
            "num_filters": 128,
            "kernel_sizes": [3, 4, 5],
            "hidden_dim": 128,
            "classifier_head": "pooled_plus_optional_semantic_to_hidden128_to_output",
            "dropout": 0.5,
            "batch_size": 32,
            "lr": 1e-3,
            "max_epochs": 20,
            "early_stopping_patience": 5,
            "max_len": max_len,
            "vocab_size": len(vocab),
            "vocab_fit": "train_only",
            "best_epoch": best_epoch,
            "best_val_f1": round(best_f1, 4),
            "device": str(device),
        },
    }
    if use_ont:
        extra["ont_features"] = ont9
        extra["ont_list"] = "anova_p005_train_only_vader_scope"
        extra["semantic_input_provenance"] = semantic_input_provenance(
            "CNN1D",
            ont9,
            {
                "parts": {"train": tr, "val": va, "test": te},
                "train": Otr.numpy(),
                "val": Ova.numpy(),
                "test": Ote.numpy(),
            },
        )
    payload = save_run("CNN1D", condition, seed, te, y_pred, extra)
    if seed == PRIMARY_SEED:
        torch.save(
            {"model_state_dict": model.state_dict(), "vocab": vocab, "max_len": max_len},
            MODEL_DIR / f"CNN1D_{condition}_v2.pt",
        )
    return payload


# ----------------------------- main -----------------------------


def main():
    global OUT, PRED_DIR, MET_DIR, MODEL_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="solo seed 42")
    ap.add_argument("--skip-cnn", action="store_true", help="omite CNN1D")
    ap.add_argument("--only-cnn", action="store_true", help="ejecuta solo CNN1D")
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="dispositivo para CNN1D; use cpu para reproducir las salidas canónicas",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        choices=["LR", "RF", "XGB"],
        default=["LR", "RF", "XGB"],
        help="modelos clásicos a ejecutar cuando CNN1D no es la única familia",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="directorio de salida para predicciones, métricas y modelos",
    )
    args = ap.parse_args()
    if args.skip_cnn and args.only_cnn:
        ap.error("--skip-cnn y --only-cnn son incompatibles")

    OUT = args.output_dir.resolve()
    PRED_DIR = OUT / "predictions"
    MET_DIR = OUT / "metrics"
    MODEL_DIR = OUT / "models"

    seeds = [PRIMARY_SEED] if args.quick else SEEDS

    for d in (PRED_DIR, MET_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    parts, ont9 = load_data()
    print(
        f"Split OK | train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}"
    )
    print(f"Features ont (ANOVA p<0.05, train-only): {ont9}\n")

    all_runs = []
    t0 = time.time()
    for seed in seeds:
        print(f"=== SEED {seed} ===")
        if not args.only_cnn:
            classical = {"LR": run_lr, "RF": run_rf, "XGB": run_xgb}
            for model_name in args.models:
                fn = classical[model_name]
                r = fn(parts, ont9, seed)
                all_runs += list(r.values())
        if not args.skip_cnn:
            for cond in ("BSL", "ENR"):
                all_runs.append(run_cnn(parts, ont9, seed, cond, args.device))

    # Preserve compatible runs from the model family intentionally skipped in
    # this invocation so the canonical multiseed summary remains complete.
    preserved_models = ["CNN1D"] if args.skip_cnn else (["LR", "RF", "XGB"] if args.only_cnn else [])
    for model_name in preserved_models:
        for condition in ("BSL", "ENR"):
            for seed in SEEDS:
                path = MET_DIR / f"{model_name}_{condition}_seed{seed}_test_metrics_v2.json"
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("pipeline_version") == "canonical_v04_2_vader_train_scope"
                    and payload.get("dataset_sha256") == sha256(GOLD_PATH)
                    and payload.get("lexicon_manifest_sha256") == sha256(LEXICON_PATH)
                ):
                    all_runs.append(payload)

    # Resumen multiseed
    rows = [
        {
            "model": r["model"],
            "condition": r["condition"],
            "seed": r["seed"],
            "macro_f1": r["macro_f1"],
            "accuracy": r["accuracy"],
        }
        for r in all_runs
    ]
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["model", "condition"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            n_seeds=("seed", "count"),
        )
        .round(4)
        .reset_index()
    )
    summary.to_csv(MET_DIR / "multiseed_summary_v2.csv", index=False)
    df.round(4).to_csv(MET_DIR / "multiseed_raw_v2.csv", index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\n[OK] Completado en {time.time()-t0:.0f}s. Salidas en {OUT}")


if __name__ == "__main__":
    main()
