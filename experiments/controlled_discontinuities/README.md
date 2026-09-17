# E1 controlled discontinuities

This bounded package preserves the final E1 case definitions and scientific
evidence and provides portable replay against the canonical public repository.
`E1_CASES.json` defines the mutations and independent expected outcomes.
Its historical `protocol_only_not_executed` status is retained as source metadata;
execution status is established by the final result matrices, not that label.

The preserved continuity matrix has 26 cases: four valid controls, 15 faults
detected by existing scripts, and seven detected only by supplementary checks.
Original detector runs, supplementary diagnostics, mutation manifests, and
oracle comparisons remain separate under `evidence/continuity/cases/`.
`E1_RESULTS.json` and `E1_RESULTS.csv` are the final continuity matrix.

SHACL has one intact control and ten faults. Its authoritative classification
is `evidence/shacl/summary_reclassified.json` and each case's
`result_reclassified.json`, with `REPORT_RECLASSIFIED.md` for inspection.
The original `result.json`, `report.ttl`, extracted report and mutated RDF graph
are preserved. The correction expands expected QNames to full IRIs before
localization matching; it does not alter the shapes or validator behavior.
The historical SHACL field `extension_detected` means a matching constraint
result, not an additional detector. All ten SHACL faults were detected by the
original validator after this classification correction.

## Replay

Run from the public repository root with Python 3.13 and the dependencies in
`experiments/controlled_discontinuities/requirements-replay.txt`. These are the
direct versions used for validation, not a lock of every transitive dependency.
Use an isolated environment and install with:

```sh
python -m pip install -r experiments/controlled_discontinuities/requirements-replay.txt
```

Both output directories must be new or empty and outside the public repository.
The scripts refuse overlaps with the baseline or released E1 evidence. They use
the invoking interpreter (`sys.executable`) for original detector subprocesses.

```sh
python -B experiments/controlled_discontinuities/run_e1_continuity.py --output-dir /tmp/e1-continuity
python -B experiments/controlled_discontinuities/compare_replay.py continuity /tmp/e1-continuity
python -B experiments/controlled_discontinuities/run_e1_shacl.py --output-dir /tmp/e1-shacl
python -B experiments/controlled_discontinuities/compare_replay.py shacl /tmp/e1-shacl
python -B experiments/controlled_discontinuities/verify_package.py
```

Continuity accepts case IDs as positional arguments for a bounded smoke test,
for example `E1-C0-pristine-copy E1-R1-swap-selected-matrix-rows`. Unexecuted
cases remain explicitly marked `not_executed`; the full-matrix comparison then
fails by design. Each invocation requires a fresh output directory. SHACL runs
all eleven cases and then automatically applies the preserved reclassification.

`--baseline-root PATH` selects another canonical package. Continuity additionally
accepts `--campaign-root PATH` and `--detector-scripts PATH` for validating the
frozen F123/campaign assembly. Defaults point to the public root and its original
`scripts/experiments/` detector implementations.

## Inputs and scope

Continuity copies only `data/`, `inputs/`, `src/`, `scripts/`, and these result
directories: `feature_selection`, `knowledge_graph`, `lexicon`,
`prepared_model_inputs`, `traceability`, and `validation`. Campaign inputs are
restricted to `results/anova_revalidation/metrics`, `predictions`, and the
727-byte `models/LR_ENR_scaler_v2.joblib` required by the original cell auditor.
No model weights, sensitivity results, or top-level `experiments/` are copied.
Fresh producer reports are prepared once, then each case mutates an isolated
copy of this bounded baseline. Original consumer-before-refresh and
producer-after-refresh behavior is unchanged. Supplementary matrix, RDF,
prediction fingerprint and stale-report checks retain their separate outputs.

SHACL reads the canonical `results/knowledge_graph/posts.ttl` and the
`rr-core`, `rr-domain`, `rr-sentiment`, and `rr-shapes` Turtle resources under
`src/ontology/resources/`. It preserves the historical HTTPS-to-HTTP SHACL
namespace normalization and the original RDFS/advanced validation settings.

## Provenance and reproducibility limits

`SOURCE_PUBLIC_PROVENANCE.json` records the workspace-relative scientific source,
source SHA-256, public path, and final public SHA-256 for each exported artifact.
Private workspace, home-directory, and interpreter paths are normalized to public paths or
explicit `${SOURCE_WORKSPACE}`, `${SOURCE_HOME}`, `${PUBLIC_REPO}`, and
`${PYTHON}` placeholders. Hashes embedded in historical reports and before/after
hash inventories still describe the original source files, including omitted
per-case package copies. They are not claimed to hash normalized public bytes.
No private correspondence is included. Detector stdout/stderr embedded in the
original result JSON is retained as evidence; separate duplicate log files are
omitted. The maintainer exporter `package_evidence.py SOURCE_PROTOCOL_DIRECTORY`
is not needed for replay and does not overwrite adapted entry points.

`compare_replay.py` compares every case ID and the recorded decision fields,
including original versus supplementary detection, false positives, missing
detections, evaluation errors, and SHACL constraint localization counts/oracles.
It does not require byte-identical RDF serialization, blank-node IDs, timestamps,
absolute paths, or timings. It also does not establish semantic correctness,
recall over arbitrary faults, or generalization beyond the declared mutations.
Models are neither trained nor used for inference; stored predictions are
audited, and the scaler is a preprocessing artifact. Trusted local joblib
artifacts are required. Full model-training reproduction is outside this replay.

`REPLAY_VALIDATION.json` records the release-packaging verification outcome.
