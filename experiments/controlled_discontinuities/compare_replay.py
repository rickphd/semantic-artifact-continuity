#!/usr/bin/env python3
"""Compare scientific case decisions, excluding paths, timings and RDF blank IDs."""
import argparse
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
FIELDS = {
    "continuity": ["family", "oracle_should_alarm", "original_detected", "extension_detected",
                   "detection_status", "expected_exception_detected", "evaluation_error",
                   "false_positive", "non_detection", "partial_detection",
                   "original_detected_before_refresh", "original_detected_after_refresh",
                   "extension_check_ids"],
    "shacl": ["family", "oracle_should_alarm", "pyshacl_conforms", "validation_result_count",
              "original_detected", "extension_detected", "detection_status",
              "localization_status", "false_positive", "non_detection",
              "partial_detection", "matched_result_count", "expected"],
}


def load(suite, root):
    name = "E1_RESULTS.json" if suite == "continuity" else "summary_reclassified.json"
    data = json.loads((root / name).read_text())
    rows = data if suite == "continuity" else data["cases"]
    result = {r["case_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate case IDs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=FIELDS)
    parser.add_argument("replay_dir", type=Path)
    args = parser.parse_args()
    reference = load(args.suite, PACKAGE / "evidence" / args.suite)
    replay = load(args.suite, args.replay_dir)
    differences = []
    for case_id in sorted(reference.keys() | replay.keys()):
        if case_id not in reference or case_id not in replay:
            differences.append({"case_id": case_id, "error": "missing/extra case"})
            continue
        for field in FIELDS[args.suite]:
            expected, actual = reference[case_id].get(field), replay[case_id].get(field)
            if expected != actual:
                differences.append({"case_id": case_id, "field": field,
                                    "expected": expected, "actual": actual})
    print(json.dumps({"suite": args.suite, "cases": len(replay),
                      "status": "passed" if not differences else "failed",
                      "differences": differences}, indent=2))
    raise SystemExit(bool(differences))


if __name__ == "__main__":
    main()
