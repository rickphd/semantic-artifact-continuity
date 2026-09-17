#!/usr/bin/env python3
"""Maintainer-only bounded export from the frozen E1 protocol directory."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    dest = Path(__file__).resolve().parent
    workspace = source.parents[2]
    replacements = [
        (str(source / "E1_runs"), "experiments/controlled_discontinuities/evidence"),
        (str(workspace / "JOWS/revisions/f123_20260914/package"), "${PUBLIC_REPO}"),
        (str(workspace / "JOWS/revisions/campaign_f123_20260914/package"), "${PUBLIC_REPO}"),
        (str(workspace / "JOWS/github_repo"), "${PUBLIC_REPO}"),
        (str(workspace), "${SOURCE_WORKSPACE}"),
        (str(Path.home()), "${SOURCE_HOME}"),
        ("/private/tmp/information_a0_env/bin/python", "${PYTHON}"),
        ("JOWS/revisions/e123_protocols_20260915/E1_runs", "experiments/controlled_discontinuities/evidence"),
        ("JOWS/revisions/f123_20260914/package", "${PUBLIC_REPO}"),
        ("JOWS/revisions/campaign_f123_20260914/package", "${PUBLIC_REPO}"),
        ("JOWS/github_repo", "${PUBLIC_REPO}"),
    ]
    files = [(source / "E1_CASES.json", dest / "E1_CASES.json")]
    for suite in ("continuity", "shacl"):
        root = source / "E1_runs" / suite
        for path in sorted(root.iterdir()):
            if path.suffix in {".json", ".csv", ".md"}:
                files.append((path, dest / "evidence" / suite / path.name))
        for case in sorted((root / "cases").iterdir()):
            for path in sorted(case.iterdir()):
                if path.is_file() and path.suffix in {".json", ".ttl", ".txt"}:
                    files.append((path, dest / "evidence" / suite / "cases" / case.name / path.name))
        names = [f"run_e1_{suite}.py"]
        if suite == "shacl":
            names.append("reclassify_shacl_reports.py")
        for name in names:
            # Do not replace an already adapted public entry point.
            if not (dest / name).exists():
                files.append((root / name, dest / name))
    records = []
    provenance = dest / "SOURCE_PUBLIC_PROVENANCE.json"
    if provenance.exists():
        records = [r for r in json.loads(provenance.read_text())["files"]
                   if r["public"].endswith(".py")]
    for src, public in files:
        original = src.read_bytes()
        text = original.decode("utf-8")
        for old, new in replacements:
            text = text.replace(old, new)
        normalized = text.encode("utf-8")
        public.parent.mkdir(parents=True, exist_ok=True)
        public.write_bytes(normalized)
        records.append({"source": str(src.relative_to(workspace)),
                        "public": str(public.relative_to(dest)),
                        "source_sha256": digest(original), "public_sha256": digest(normalized),
                        "path_normalized": normalized != original})
    (dest / "SOURCE_PUBLIC_PROVENANCE.json").write_text(json.dumps({
        "normalization": "Local workspace, frozen package, run and interpreter paths replaced with public paths or explicit ${...} placeholders. Embedded historical hashes remain source hashes. Portable scripts are subsequently adapted; their final public hashes are updated separately.",
        "files": records}, indent=2) + "\n")
    print(f"Exported {len(records)} bounded files")


if __name__ == "__main__":
    main()
