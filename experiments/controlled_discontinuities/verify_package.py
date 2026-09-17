#!/usr/bin/env python3
"""Check exported public bytes against the source/public provenance inventory."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "SOURCE_PUBLIC_PROVENANCE.json").read_text())
    failures = []
    for entry in manifest["files"]:
        path = root / entry["public"]
        if root not in path.resolve().parents or not path.is_file():
            failures.append({"path": entry["public"], "reason": "missing or outside package"})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["public_sha256"]:
            failures.append({"path": entry["public"], "reason": "public hash mismatch"})
    print(json.dumps({"status": "failed" if failures else "passed",
                      "exported_files": len(manifest["files"]), "failures": failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
