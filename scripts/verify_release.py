#!/usr/bin/env python3
"""Validate the scientific release and its file-integrity manifest."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> list[str]:
    errors: list[str] = []
    if not MANIFEST.exists():
        return ["MANIFEST.sha256 is missing"]
    for line_number, line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"Malformed manifest line {line_number}")
            continue
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing manifest file: {relative}")
        elif sha256(path) != expected:
            errors.append(f"Hash mismatch: {relative}")
    return errors


def main() -> None:
    validation = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "experiments" / "validate_v04_2_experiment_chain.py"),
        ],
        cwd=ROOT,
        check=False,
    )
    manifest_errors = verify_manifest()
    for error in manifest_errors:
        print(f"- {error}")
    if validation.returncode or manifest_errors:
        raise SystemExit(1)
    print("Release verification passed")


if __name__ == "__main__":
    main()
