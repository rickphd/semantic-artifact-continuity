"""Select ontology variables with train-only ANOVA F-tests.

This reproduces the feature-selection rule used by the original ENR pipeline
on the canonical ID split. All 37 ontology variables are tested, and every
variable with p < 0.05 is retained. Validation and test labels are not read.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import f_classif


ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold" / "gold_enriched_ontology.parquet"
SPLIT_PATH = ROOT / "data" / "gold" / "GEN_split_gld_reddit_ids_v02.json"
OUT_PATH = ROOT / "results" / "feature_selection" / "ENR_selected_ont_features_anova_train_only.json"
LEXICON_PATH = ROOT / "results" / "lexicon" / "ontology_lexicon_v04_2_train_only.json"
ALPHA = 0.05


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    gold = pd.read_parquet(GOLD_PATH)
    split = json.loads(SPLIT_PATH.read_text())
    train = gold.set_index("id").loc[split["train"]].reset_index()
    ontology_columns = [c for c in gold.columns if c.startswith("ont_")]

    assert len(train) == 968
    assert len(ontology_columns) == 37

    f_stats, p_values = f_classif(
        train[ontology_columns].fillna(0).astype(float), train["label"]
    )
    ranking = pd.DataFrame(
        {"feature": ontology_columns, "f_stat": f_stats, "p_value": p_values}
    ).sort_values(["p_value", "f_stat"], ascending=[True, False], na_position="last")
    selected = ranking.loc[ranking["p_value"] < ALPHA, "feature"].tolist()

    payload = {
        "version": "anova_train_only_v02_vader_train_scope",
        "created_at": datetime.now().isoformat(),
        "method": "ANOVA F-test (sklearn.feature_selection.f_classif), train-only",
        "alpha": ALPHA,
        "selection_rule": "retain every ontology variable with p_value < alpha",
        "train_size": int(len(train)),
        "split_source": str(SPLIT_PATH.relative_to(ROOT)),
        "dataset": str(GOLD_PATH.relative_to(ROOT)),
        "dataset_sha256": sha256(GOLD_PATH),
        "lexicon_manifest": str(LEXICON_PATH.relative_to(ROOT)),
        "lexicon_manifest_sha256": sha256(LEXICON_PATH),
        "lexicon_induction_scope": "train_text_only_no_labels",
        "n_candidates": len(ontology_columns),
        "n_selected": len(selected),
        "selected_features_train_only": selected,
        "ranking": [
            {
                "feature": row.feature,
                "f_stat": None if np.isnan(row.f_stat) else float(row.f_stat),
                "p_value": None if np.isnan(row.p_value) else float(row.p_value),
            }
            for row in ranking.itertuples(index=False)
        ],
        "environment": {
            "python_packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            }
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Selected {len(selected)}/{len(ontology_columns)} variables at p < {ALPHA}")
    for item in payload["ranking"][: len(selected)]:
        print(f"  {item['feature']}: F={item['f_stat']:.6f}, p={item['p_value']:.8g}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
