#!/usr/bin/env python3
"""Generate result plots and provenance from the released evidence package."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
DOWNSTREAM = RESULTS / "anova_revalidation"
METRICS = DOWNSTREAM / "metrics"
ABLATION = RESULTS / "ablation" / "classical"
CNN_ABLATION = RESULTS / "ablation" / "cnn1d"
SENSITIVITY = RESULTS / "sensitivity"
FEATURE_COVERAGE = RESULTS / "validation" / "coverage" / "coverage_sparsity.csv"
MODULE_COVERAGE = RESULTS / "validation" / "coverage" / "module_coverage.csv"
MANUSCRIPT_FIGURES = PROJECT_ROOT / "figures"
PROVENANCE_MANIFEST = RESULTS / "provenance" / "figure_generation_manifest.json"
PRESENTATION_MAP = RESULTS / "traceability" / "model_inputs" / "semantic_variable_presentation_map.md"
FIGURE4_OUTPUT = MANUSCRIPT_FIGURES / "fig04_chain_governance_lineage_ledger.png"
FIGURE4_DRAWIO = PROJECT_ROOT / "figures" / "source" / "fig_chain_governance_lineage_ledger.drawio"
FIGURE4_SPEC = PROJECT_ROOT / "figures" / "source" / "fig_chain_governance_lineage_ledger.spec.json"
MODELS = ["LR", "RF", "XGB", "CNN1D"]
LABELS = ["Negative", "Neutral", "Positive"]
FEATURE_PRESENTATION = {
    "ont_domain_density": ("Domain-concept density", "Post-level aggregate", "Density of active domain concepts in a post"),
    "ont_InteligenciaArtificial_Neutro": ("Neutral AI evidence", "InteligenciaArtificial", "Neutral local lexical evidence around the AI concept"),
    "ont_Innovacion_Neutro": ("Neutral innovation evidence", "Innovacion", "Neutral local lexical evidence around the innovation concept"),
    "ont_total_negativo_mentions": ("Total negative mentions", "Post-level aggregate", "Aggregate negative mentions across controlled concepts"),
    "ont_InteligenciaArtificial_Negativo": ("Negative AI evidence", "InteligenciaArtificial", "Negative local lexical evidence around the AI concept"),
    "ont_Etica_Negativo": ("Negative ethics evidence", "Etica", "Negative local lexical evidence around the ethics concept"),
    "ont_Tecnologia_Positivo": ("Positive technology evidence", "Tecnologia", "Positive local lexical evidence around the technology concept"),
    "ont_Datos_Negativo": ("Negative data evidence", "Datos", "Negative local lexical evidence around the data concept"),
    "ont_Robot_Negativo": ("Negative robot evidence", "Robot", "Negative local lexical evidence around the robot concept"),
    "ont_total_positivo_mentions": ("Total positive mentions", "Post-level aggregate", "Aggregate positive mentions across controlled concepts"),
    "ont_Innovacion_Positivo": ("Positive innovation evidence", "Innovacion", "Positive local lexical evidence around the innovation concept"),
    "ont_Tecnologia_Negativo": ("Negative technology evidence", "Tecnologia", "Negative local lexical evidence around the technology concept"),
    "ont_Futuro_Negativo": ("Negative future evidence", "Futuro", "Negative local lexical evidence around the future concept"),
    "ont_Futuro_Positivo": ("Positive future evidence", "Futuro", "Positive local lexical evidence around the future concept"),
    "ont_Datos_Positivo": ("Positive data evidence", "Datos", "Positive local lexical evidence around the data concept"),
    "ont_Tecnologia_Neutro": ("Neutral technology evidence", "Tecnologia", "Neutral local lexical evidence around the technology concept"),
}
MODULE_PRESENTATION = {
    "ai": "AI",
    "algorithm": "Algorithm",
    "automation": "Automation",
    "data": "Data",
    "ethics": "Ethics",
    "future": "Future",
    "global_counts": "Global counts",
    "innovation": "Innovation",
    "ml": "Machine learning",
    "robot": "Robot",
    "technology": "Technology",
}
FIGURE_OUTPUTS = {
    "fig05_module_coverage": MANUSCRIPT_FIGURES / "fig05_module_coverage.png",
    "fig06_lexical_sensitivity": MANUSCRIPT_FIGURES / "fig06_lexical_sensitivity.png",
    "fig07_downstream_macro_f1": MANUSCRIPT_FIGURES / "fig07_downstream_macro_f1.png",
    "fig08_downstream_confusion_delta": MANUSCRIPT_FIGURES / "fig08_downstream_confusion_delta.png",
    "fig09_prediction_distribution": MANUSCRIPT_FIGURES / "fig09_prediction_distribution.png",
    "fig10_module_sensitivity": MANUSCRIPT_FIGURES / "fig10_module_sensitivity.png",
    "fig11_feature_stability": MANUSCRIPT_FIGURES / "fig11_feature_stability.png",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(METRICS / "multiseed_summary_v2.csv")
    expected = {(model, condition) for model in MODELS for condition in ["BSL", "ENR"]}
    observed = set(zip(summary["model"], summary["condition"]))
    if observed != expected or not (summary["n_seeds"] == 3).all():
        raise ValueError("downstream summary is incomplete or has unexpected groups")
    return summary


def plot_macro_f1(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=300)
    x = np.arange(len(MODELS))
    width = 0.36
    colors = {"BSL": "#3b82b8", "ENR": "#d95f4f"}
    for offset, condition in [(-width / 2, "BSL"), (width / 2, "ENR")]:
        rows = summary[summary["condition"] == condition].set_index("model").reindex(MODELS)
        values = rows["macro_f1_mean"].to_numpy()
        errors = rows["macro_f1_std"].fillna(0).to_numpy()
        ax.bar(x + offset, values, width, yerr=errors, capsize=3, label=condition,
               color=colors[condition], edgecolor="#222222", linewidth=0.7)
    ax.set_xticks(x, MODELS, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_ylim(0.45, 0.63)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUTS["fig07_downstream_macro_f1"], bbox_inches="tight")
    plt.close(fig)


def metric_path(model: str, condition: str, seed: int) -> Path:
    return METRICS / f"{model}_{condition}_seed{seed}_test_metrics_v2.json"


def mean_confusion_delta(model: str) -> np.ndarray:
    deltas = []
    for seed in [42, 123, 2024]:
        matrices = []
        for condition in ["BSL", "ENR"]:
            payload = json.loads(metric_path(model, condition, seed).read_text())
            cm = np.asarray(payload["confusion_matrix"], dtype=float)
            matrices.append(cm / cm.sum(axis=1, keepdims=True))
        deltas.append((matrices[1] - matrices[0]) * 100)
    return np.mean(deltas, axis=0)


def plot_confusion_delta() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.5), dpi=300, constrained_layout=True)
    all_delta = [mean_confusion_delta(model) for model in MODELS]
    vmax = max(5.0, np.ceil(max(np.abs(delta).max() for delta in all_delta) / 5) * 5)
    image = None
    for ax, model, delta in zip(axes.ravel(), MODELS, all_delta):
        image = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(model)
        ax.set_xticks(range(3), LABELS, rotation=30, ha="right")
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                value = delta[i, j]
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center",
                        color="white" if abs(value) > vmax * 0.55 else "#222222", fontsize=9)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="ENR - BSL (percentage points)")
    fig.savefig(FIGURE_OUTPUTS["fig08_downstream_confusion_delta"], bbox_inches="tight")
    plt.close(fig)


def plot_module_sensitivity() -> None:
    data = pd.concat(
        [
            pd.read_csv(ABLATION / "module_ablation_summary.csv"),
            pd.read_csv(CNN_ABLATION / "cnn1d_module_ablation_summary.csv"),
        ],
        ignore_index=True,
    )
    data = data[data["condition"].str.startswith("drop_")].copy()
    data["module"] = data["condition"].str.removeprefix("drop_")
    pivot = data.pivot(index="module", columns="model", values="delta_vs_full37").reindex(
        index=MODULE_PRESENTATION,
        columns=MODELS,
    )
    if pivot.isna().any().any():
        missing = pivot.isna().stack().loc[lambda values: values].index.tolist()
        raise ValueError(f"module-ablation figure is missing model/group values: {missing}")
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=300)
    image = ax.imshow(
        pivot.to_numpy(), cmap="RdBu_r", vmin=-0.08, vmax=0.08, aspect=0.7
    )
    ax.set_xticks(range(len(MODELS)), MODELS)
    display_labels = [MODULE_PRESENTATION.get(module, module.replace("_", " ").title())
                      for module in pivot.index]
    ax.set_yticks(range(len(pivot)), display_labels)
    ax.set_xlabel("Model")
    ax.set_ylabel("Removed domain-concept group")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Macro-F1 change vs full 37-variable condition")
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUTS["fig10_module_sensitivity"], bbox_inches="tight")
    plt.close(fig)


def plot_lexical_sensitivity() -> None:
    """Show parameter-space variation without implying downstream robustness."""
    data = pd.read_csv(SENSITIVITY / "vader_ke_sensitivity_summary.csv")
    stability = pd.read_csv(SENSITIVITY / "selected_feature_stability.csv")
    canonical = set(stability.loc[stability["canonical_selected"], "feature"])
    data["jaccard"] = data["selected_features"].fillna("").map(
        lambda value: len(canonical & set(value.split(";"))) /
        len(canonical | set(value.split(";"))) if value else 0.0
    )
    passed = data[data["variant_status"] == "passed"].copy()
    failed = data[data["variant_status"] != "passed"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), dpi=300, constrained_layout=True)
    ax = axes[0]
    scatter = ax.scatter(
        passed["lexicon_size"], passed["selected_feature_count"],
        c=passed["jaccard"], cmap="viridis", vmin=0, vmax=1,
        s=42, edgecolor="#222222", linewidth=0.35,
    )
    if not failed.empty:
        ax.scatter(failed["lexicon_size"], failed["selected_feature_count"],
                   marker="x", color="#b2182b", s=34, label="Invalid variant")
    ax.set_xlabel("Induced lexicon size")
    ax.set_ylabel("Selected semantic variables")
    ax.set_title("Selected interface")
    ax.grid(color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.colorbar(scatter, ax=ax, label="Jaccard vs canonical set")

    ax = axes[1]
    pivot = passed.pivot_table(index="context_window", columns="minimum_frequency",
                               values="jaccard", aggfunc="median")
    image = ax.imshow(pivot.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Minimum term frequency")
    ax.set_ylabel("Context window")
    ax.set_title("Median interface similarity")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}" if pd.notna(value) else "n/a",
                    ha="center", va="center", fontsize=8,
                    color="#222222" if pd.notna(value) else "#777777")
    fig.colorbar(image, ax=ax, label="Median Jaccard")
    fig.savefig(FIGURE_OUTPUTS["fig06_lexical_sensitivity"], bbox_inches="tight")
    plt.close(fig)


def plot_module_coverage() -> None:
    data = pd.read_csv(MODULE_COVERAGE).sort_values("active_pct")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=300)
    y = np.arange(len(data))
    bars = ax.barh(y, data["active_pct"], color="#4c78a8", edgecolor="#222222", linewidth=0.5)
    module_labels = (
        data["module"]
        .str.replace("_", " ", regex=False)
        .str.title()
        .replace({"Ai": "AI", "Ml": "ML"})
    )
    ax.set_yticks(y, module_labels)
    ax.set_xlabel("Posts with at least one active variable (%)")
    ax.set_xlim(0, max(115, data["active_pct"].max() * 1.12))
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value, selected in zip(bars, data["active_pct"], data["selected_feature_count"]):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}% | {int(selected)} selected", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUTS["fig05_module_coverage"], bbox_inches="tight")
    plt.close(fig)


def load_predictions() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted((METRICS.parent / "predictions").glob("*.csv"))]
    data = pd.concat(frames, ignore_index=True)
    expected = {(model, condition, seed) for model in MODELS for condition in ["BSL", "ENR"]
                for seed in [42, 123, 2024]}
    observed = set(zip(data["model"], data["condition"], data["seed"]))
    if observed != expected:
        raise ValueError("prediction package is incomplete")
    return data


def plot_prediction_distribution() -> None:
    data = load_predictions()
    rows = []
    for (model, condition, seed), group in data.groupby(["model", "condition", "seed"]):
        counts = group["y_pred"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
        for label, share in counts.items():
            rows.append({"model": model, "condition": condition, "seed": seed,
                         "label": LABELS[label], "share": share * 100})
    summary = pd.DataFrame(rows).groupby(["model", "condition", "label"], as_index=False)["share"].agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 3.0), dpi=300, sharey=True, constrained_layout=True)
    colors = {"Negative": "#4c78a8", "Neutral": "#f2cf5b", "Positive": "#e45756"}
    for ax, model in zip(axes, MODELS):
        subset = summary[summary["model"] == model]
        x = np.arange(2)
        bottom = np.zeros(2)
        for label in LABELS:
            values = subset[subset["label"] == label].set_index("condition").reindex(["BSL", "ENR"])
            ax.bar(x, values["mean"], bottom=bottom, color=colors[label], label=label,
                   edgecolor="white", linewidth=0.4)
            bottom += values["mean"].to_numpy()
        ax.set_title(model)
        ax.set_xticks(x, ["BSL", "ENR"])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Predicted class share (%)")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(FIGURE_OUTPUTS["fig09_prediction_distribution"], bbox_inches="tight")
    plt.close(fig)


def plot_feature_stability() -> None:
    data = pd.read_csv(SENSITIVITY / "selected_feature_stability.csv").sort_values("selected_pct")
    display_labels = data["feature"].map(lambda feature: FEATURE_PRESENTATION.get(feature, (None,))[0])
    if display_labels.isna().any():
        missing = data.loc[display_labels.isna(), "feature"].tolist()
        raise ValueError(f"missing publication labels for features: {missing}")
    fig, ax = plt.subplots(figsize=(8.2, 5.5), dpi=300)
    y = np.arange(len(data))
    colors = ["#d95f4f" if value else "#4c78a8" for value in data["canonical_selected"]]
    ax.barh(y, data["selected_pct"], color=colors, edgecolor="#222222", linewidth=0.45)
    ax.set_yticks(y, display_labels)
    ax.set_xlabel("Variants selecting the semantic variable (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#d95f4f", label="Canonical selected set"),
                       plt.Rectangle((0, 0), 1, 1, color="#4c78a8", label="Alternative semantic variable")],
              frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUTS["fig11_feature_stability"], bbox_inches="tight")
    plt.close(fig)


def write_provenance_manifest() -> None:
    metric_files = sorted(METRICS.glob("*_test_metrics_v2.json"))
    prediction_files = sorted((DOWNSTREAM / "predictions").glob("*_test_predictions_v2.csv"))
    source_groups = {
        "fig05_module_coverage": [MODULE_COVERAGE],
        "fig06_lexical_sensitivity": [
            SENSITIVITY / "vader_ke_sensitivity_summary.csv",
            SENSITIVITY / "selected_feature_stability.csv",
        ],
        "fig07_downstream_macro_f1": [METRICS / "multiseed_summary_v2.csv"],
        "fig08_downstream_confusion_delta": metric_files,
        "fig09_prediction_distribution": prediction_files,
        "fig10_module_sensitivity": [
            ABLATION / "module_ablation_summary.csv",
            CNN_ABLATION / "cnn1d_module_ablation_summary.csv",
        ],
        "fig11_feature_stability": [SENSITIVITY / "selected_feature_stability.csv"],
    }
    figures = [
        {
            "figure_id": "fig04_chain_governance_lineage_ledger",
            "included_path": relative(FIGURE4_OUTPUT),
            "png_sha256": sha256(FIGURE4_OUTPUT),
            "generation_method": "Draw.io export from the editable source; the specification and active evidence reports define the displayed callouts.",
            "editable_source": {
                "path": relative(FIGURE4_DRAWIO),
                "sha256": sha256(FIGURE4_DRAWIO),
            },
            "input_artifacts": [
                {"path": relative(FIGURE4_SPEC), "sha256": sha256(FIGURE4_SPEC)},
                {
                    "path": relative(RESULTS / "traceability" / "chain_reconciliation_report.json"),
                    "sha256": sha256(RESULTS / "traceability" / "chain_reconciliation_report.json"),
                },
                {
                    "path": relative(RESULTS / "knowledge_graph" / "materialization_report.json"),
                    "sha256": sha256(RESULTS / "knowledge_graph" / "materialization_report.json"),
                },
                {
                    "path": relative(RESULTS / "validation" / "shacl" / "baseline_report.json"),
                    "sha256": sha256(RESULTS / "validation" / "shacl" / "baseline_report.json"),
                },
            ],
        }
    ]
    for figure_id, output in FIGURE_OUTPUTS.items():
        inputs = source_groups[figure_id]
        figures.append(
            {
                "figure_id": figure_id,
                "included_path": relative(output),
                "png_sha256": sha256(output),
                "input_artifacts": [
                    {"path": relative(path), "sha256": sha256(path)} for path in inputs
                ],
            }
        )
    payload = {
        "status": "passed",
        "scope": "Figures 4-11 included by the manuscript",
        "figures": figures,
    }
    PROVENANCE_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_presentation_map() -> None:
    stability = pd.read_csv(SENSITIVITY / "selected_feature_stability.csv")
    rows = []
    for feature in stability.sort_values("selected_pct", ascending=False)["feature"]:
        presentation = FEATURE_PRESENTATION.get(feature)
        if presentation is None:
            raise ValueError(f"missing publication mapping for feature: {feature}")
        label, source_term, evidence = presentation
        rows.append(f"| {label} | `{feature}` | `{source_term}` | {evidence} |")

    PRESENTATION_MAP.write_text(
        "# Semantic Variable Presentation Map\n\n"
        "This release-facing map relates the English labels used in the manuscript to the "
        "executed semantic-variable identifiers and the original Spanish ontology terms. "
        "The Spanish forms remain in the Gold records and ontology vocabulary; English labels "
        "are presentation glosses and do not alter the activation protocol.\n\n"
        "| Publication label | Executed variable identifier | Original Spanish source term | Evidence represented |\n"
        "|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    plot_macro_f1(summary)
    plot_confusion_delta()
    plot_module_sensitivity()
    plot_lexical_sensitivity()
    plot_module_coverage()
    plot_prediction_distribution()
    plot_feature_stability()
    write_provenance_manifest()
    write_presentation_map()
    print(PROVENANCE_MANIFEST)
    print(PRESENTATION_MAP)


if __name__ == "__main__":
    main()
