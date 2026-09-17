"""Filesystem boundaries shared by the portable E1 replay entry points."""
from pathlib import Path
import shutil

PACKAGE = Path(__file__).resolve().parent


def scratch_root(value, baseline):
    root = Path(value).resolve()
    baseline = Path(baseline).resolve()
    for protected in (PACKAGE, baseline):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError(f"Scratch must be disjoint from released evidence and baseline: {root}")
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Use a new or empty scratch directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_reset(path, root):
    path, root = Path(path), Path(root).resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents or path.is_symlink():
        raise ValueError(f"Refusing reset outside scratch descendants: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_baseline(source, target, campaign, scripts):
    # Deliberately omit models, experiments, provenance, and unrelated results.
    directories = ["data", "inputs", "src", "scripts",
                   "results/feature_selection", "results/knowledge_graph",
                   "results/lexicon", "results/prepared_model_inputs",
                   "results/traceability", "results/validation"]
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    for relative in directories:
        shutil.copytree(source / relative, target / relative, ignore=ignore)
    for relative in ("metrics", "predictions"):
        shutil.copytree(campaign / "results/anova_revalidation" / relative,
                        target / "results/anova_revalidation" / relative, ignore=ignore)
    scaler = Path("results/anova_revalidation/models/LR_ENR_scaler_v2.joblib")
    (target / scaler).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(campaign / scaler, target / scaler)
    for name in ("audit_model_inputs.py", "audit_model_input_cells.py", "reconcile_artifact_chain.py"):
        shutil.copy2(scripts / name, target / "scripts/experiments" / name)
