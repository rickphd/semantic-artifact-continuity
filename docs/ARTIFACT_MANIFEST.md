# Artifact Map

| Path | Role |
| --- | --- |
| `inputs/original_gold.parquet` | Minimized input to corrected enrichment; same source records, labels and split. |
| `data/gold/` | Corrected 47-column Gold, metadata and fixed IDs. |
| `src/ontology/` | Span extraction, context-window implementation and RDF/SHACL resources. |
| `results/lexicon/` | Train-only induction manifest and lexical entries. |
| `results/feature_selection/` | Training-only ANOVA ranking and six-variable order. |
| `results/knowledge_graph/` | Canonical RDF and materialization report. |
| `results/prepared_model_inputs/` | Raw/standardized matrices, ordered IDs, scaler and profiles. |
| `results/traceability/` | Record/variable/cell traces and reconciliation reports. |
| `results/validation/` | Baseline SHACL, combined historical fixture and coverage. |
| `results/anova_revalidation/` | 24 main metrics and predictions; LR input scaler. |
| `results/ablation/` | 126 classical and 42 CNN1D final runs and predictions. |
| `results/sensitivity/` | 125 lexicons, complete artifacts for 97 admissible settings and aggregate outputs. |
| `experiments/controlled_discontinuities/` | Independent SHACL and continuity suites, mutations, detectors, oracles and replay. |
| `experiments/human_assessment/` | Final guide, sample, anonymous judgments, adjudications, reference, comparisons and replay. |
| `results/provenance/campaign/` | Full-campaign summary, environment, protocol and reuse provenance. |
| `results/provenance/canonical/` | Frozen corrected-generation execution and environment records. |
| `results/provenance/sensitivity_audit/` | Admissibility audit for all 125 settings. |
| `results/provenance/figures/` | Historical figure provenance and public dependency mapping. |
| `results/provenance/public_projection.json` | Computational source/public SHA-256 pairs and transformations. |
| `MANIFEST.sha256` | Integrity of the curated public release, including code and documentation. |

Excluded: duplicated E1 per-case full packages, large trained classifier binaries,
partial run tables, raw API payloads, account-name mappings, private correspondence,
reviewer letters, manuscript drafts, and superseded human-assessment rounds.
Mutation recipes and shared baseline artifacts replace duplicated E1 packages;
the frozen outputs remain separately identifiable. No excluded experiment is
represented as completed.
