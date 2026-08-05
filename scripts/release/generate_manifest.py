#!/usr/bin/env python3
"""Generate the repository SHA-256 manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv"}
EXCLUDED_NAMES = {".DS_Store", "MANIFEST.sha256"}
RELEASED_MODEL = (
    ROOT
    / "results"
    / "anova_revalidation"
    / "models"
    / "LR_ENR_scaler_v2.joblib"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included(path: Path) -> bool:
    if not path.is_file() or path.name in EXCLUDED_NAMES:
        return False
    if set(path.relative_to(ROOT).parts) & EXCLUDED_PARTS:
        return False
    if path.suffix in {".pyc", ".pt", ".pth", ".safetensors", ".ckpt"}:
        return False
    if path.suffix == ".joblib" and path.resolve() != RELEASED_MODEL.resolve():
        return False
    return True


def main() -> None:
    paths = sorted(path for path in ROOT.rglob("*") if included(path))
    lines = [f"{sha256(path)}  {path.relative_to(ROOT)}" for path in paths]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(lines)} entries")


if __name__ == "__main__":
    main()
