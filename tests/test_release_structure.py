from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ReleaseStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = pd.read_parquet(
            ROOT / "data" / "gold" / "gold_enriched_ontology.parquet"
        )

    def test_dataset_scope(self) -> None:
        self.assertEqual(self.data.shape, (1614, 47))
        self.assertEqual(
            self.data["split"].value_counts().to_dict(),
            {"train": 968, "val": 323, "test": 323},
        )
        self.assertEqual(
            len([column for column in self.data if column.startswith("ont_")]),
            37,
        )

    def test_direct_identifiers_are_not_released(self) -> None:
        self.assertIn("author_id", self.data)
        self.assertNotIn("autor", self.data)
        self.assertNotIn("autor_nombre_completo", self.data)
        self.assertNotIn("raw_payload", self.data)

    def test_selected_interface(self) -> None:
        path = (
            ROOT
            / "results"
            / "feature_selection"
            / "ENR_selected_ont_features_anova_train_only.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["n_selected"], 7)
        self.assertEqual(len(payload["selected_features_train_only"]), 7)

    def test_downstream_file_counts(self) -> None:
        root = ROOT / "results" / "anova_revalidation"
        self.assertEqual(len(list((root / "metrics").glob("*_test_metrics_v2.json"))), 24)
        self.assertEqual(
            len(list((root / "predictions").glob("*_test_predictions_v2.csv"))),
            24,
        )

    def test_release_directories_are_bounded(self) -> None:
        ablation_dirs = {
            path.name
            for path in (ROOT / "results" / "ablation").iterdir()
            if path.is_dir()
        }
        result_dirs = {path.name for path in (ROOT / "results").iterdir() if path.is_dir()}
        self.assertEqual(ablation_dirs, {"classical", "cnn1d"})
        self.assertEqual(
            result_dirs,
            {
                "ablation",
                "anova_revalidation",
                "feature_selection",
                "knowledge_graph",
                "lexicon",
                "provenance",
                "sensitivity",
                "traceability",
                "validation",
            },
        )


if __name__ == "__main__":
    unittest.main()
