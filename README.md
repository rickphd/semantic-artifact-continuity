# Traceable Knowledge Engineering Artifacts

Research artifacts for **Evaluating Semantic-Artifact Continuity for Noisy Social
Text: RDF Materialization, SHACL Conformance, and Model-Input Evidence**.

**Version 1.1.0** publishes the corrected extraction and evaluation artifacts,
controlled-discontinuity experiments, and final human assessment. The original
package remains available at [v1.0.0](https://github.com/rickphd/semantic-artifact-continuity/tree/v1.0.0).
Use [v1.1.0](https://github.com/rickphd/semantic-artifact-continuity/tree/v1.1.0)
when citing this evidence; do not mix results from the two versions.

The evaluated workflow connects 1,614 source posts, ontology activations, RDF
resources, SHACL conformance reports, 37 numeric semantic variables, six
train-only selected model inputs, and predictions. Structural conformance,
continuity checks, human semantic correspondence, and downstream predictive
behavior provide distinct evidence. SHACL conformance is not semantic accuracy,
and downstream differences do not establish universal predictive gains.

## Evidence

| Family | Scope | Location |
| --- | --- | --- |
| Corrected canonical artifacts | 1,614 posts; 15,616 triples; 37 variables; six selected variables | `data/`, `inputs/`, `src/`, `results/knowledge_graph/`, `results/lexicon/` |
| Input continuity | 9,684 selected cells including zeros; 27 matched runtime profiles | `results/prepared_model_inputs/`, `results/traceability/` |
| Downstream comparison | 24 runs; LR/RF/XGB/CNN1D, BSL/ENR, three seeds | `results/anova_revalidation/` |
| Module ablation | 126 classical and 42 CNN1D runs | `results/ablation/` |
| Lexical sensitivity | 125 settings: 97 evaluated, 28 rejected before downstream evaluation | `results/sensitivity/`, `results/provenance/sensitivity_audit/` |
| Independent SHACL tests | 10 faults and one control | `experiments/controlled_discontinuities/` |
| Continuity discontinuities | 22 faults and four controls; 15 original and seven supplementary detections | `experiments/controlled_discontinuities/` |
| Human assessment | 120 texts, ten concepts, 1,200 units; anonymous A/B decisions and 77 adjudications | `experiments/human_assessment/` |
| Figures | Current Figures 1-11, editable diagrams and plotting code | `figures/`, `scripts/figures/` |

All 62,016 main/ablation prediction rows are included. The 97 admissible
sensitivity settings include their numeric tables, graphs and reports; the
28 rejected settings retain their lexicons and rejection records, not fabricated
downstream results. Eight seed-42 main runs and the canonical sensitivity setting
were reused from the recorded pilot, not trained again during the campaign.
No transformer experiment is represented as completed.

## Verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
python -m unittest discover -s tests
```

The verifier checks released file hashes and recomputes main/ablation metrics
from predictions. It also checks split assignments, the selected interface,
runtime profiles and sensitivity coverage, verifies the E1 evidence inventory,
and replays E2 in a temporary directory. This is stored-result verification,
not model retraining. Portable replay commands for the independent experiments
are documented in their directories. See [reproducibility](docs/REPRODUCIBILITY.md),
[result traceability](docs/RESULTS_TRACEABILITY.md), and the
[artifact map](docs/ARTIFACT_MANIFEST.md).

## Data And Provenance

The original and corrected Gold tables retain the existing 47-column public
schema and corpus-local author identifiers. Account-name mappings, raw API
payloads, private correspondence and annotator identities are not distributed.
Post IDs and text remain linkable to public posts; this is data minimization,
not anonymization of the source texts. Human-assessment projections retain
structured decisions but omit private statements and unreviewed free prose.

`results/provenance/public_projection.json` links frozen computational source
hashes to public files and records path projections. Experiment directories
provide their own projection provenance. Historical hashes identify original
executions; `MANIFEST.sha256` identifies this public version. Metadata-only path
normalization does not mean experiments were rerun.

The source corpus has a version-specific [Hugging Face DOI](https://doi.org/10.57967/hf/9852).
That deposit is not asserted to contain the corrected v1.1.0 semantic values.
Use this tagged repository for those values and the revised experiments.
See [DATA_NOTICE.md](DATA_NOTICE.md) before reuse.

## License And Citation

Code is MIT-licensed; see [LICENSE](LICENSE). Author-created labels, split
assignments, semantic variables, documentation and release metadata are CC BY
4.0. Ontology files retain their embedded licenses. Neither license transfers
rights to Reddit-authored text. Citation metadata is in [CITATION.cff](CITATION.cff).
No article or repository DOI is implied where none has been assigned.
