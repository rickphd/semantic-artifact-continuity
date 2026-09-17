#!/usr/bin/env python3
"""Validate the scientific release and its file-integrity manifest."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
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
    if sys.flags.optimize:
        raise SystemExit('Run without -O; the E2 replay uses assertion-based checks.')
    validation = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "experiments" / "validate_release.py"),
        ],
        cwd=ROOT,
        check=False,
    )
    e1 = subprocess.run([sys.executable, '-B', str(ROOT / 'experiments/controlled_discontinuities/verify_package.py')], cwd=ROOT, check=False)
    with tempfile.TemporaryDirectory(prefix='semantic-release-verify-') as temporary:
        e2 = subprocess.run([
            sys.executable, '-B', str(ROOT / 'experiments/human_assessment/evaluate.py'),
            '--output-dir', str(Path(temporary) / 'e2'),
        ], cwd=ROOT, check=False)
    manifest_errors = verify_manifest()
    for error in manifest_errors:
        print(f"- {error}")
    if validation.returncode or e1.returncode or e2.returncode or manifest_errors:
        raise SystemExit(1)
    print("Release verification passed")


if __name__ == "__main__":
    main()
