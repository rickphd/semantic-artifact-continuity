# Traceable Knowledge Engineering Artifacts

Research artifact for the manuscript:

**Evaluating Semantic-Artifact Continuity for Noisy Social Text: RDF
Materialization, SHACL Conformance, and Model-Input Evidence**

The repository contains the code, ontology modules, minimized research dataset,
RDF materialization, SHACL reports, semantic-input traces, model predictions,
aggregate results, and figure sources needed to inspect the evidence reported in
the manuscript.

The evaluated chain connects 1,614 Reddit posts, 37 ontology-derived variables,
seven train-only selected variables, RDF resources, SHACL conformance reports,
downstream model inputs, and predictions. Downstream results are secondary
evidence: the repository does not claim universal predictive gains or semantic
truth from SHACL conformance.

## Repository Layout

```text
data/gold/                     Gold dataset, metadata, and fixed split
docs/                          Artifact map and reproducibility documentation
figures/                       Figures 1-11 and editable sources
results/
  ablation/                    Classical and CNN1D module-ablation results
  anova_revalidation/          Downstream metrics, predictions, and input scaler
  feature_selection/           Train-only semantic-variable selection
  knowledge_graph/             Materialized RDF graph and report
  lexicon/                     Frozen lexicon and induction report
  provenance/                  Inventories, hashes, and figure provenance
  sensitivity/                 VADER-based lexical sensitivity summaries
  traceability/                Evidence-chain and model-input traces
  validation/                  Coverage and SHACL validation results
scripts/experiments/           Evidence generation and model evaluation
scripts/figures/               Result-figure generation
scripts/release/               Release preparation and manifest utilities
src/ontology/                  Enrichment code and canonical RDF/SHACL resources
tests/                         Release integrity checks
```

## Quick Verification

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

The verifier checks the minimized dataset, split sizes, ontology-variable count,
selected interface, 24 metric files, 24 prediction files, RDF/SHACL reports,
traceability relations, active ablations, figure provenance, and the repository
SHA-256 manifest.

## Main Evidence

| Evidence | Location |
|---|---|
| RDF graph and materialization report | `results/knowledge_graph/` |
| SHACL baseline and controlled invalid fixture | `results/validation/shacl/` |
| Cell-level selected-input traceability | `results/traceability/model_inputs/selected_semantic_input_cell_trace.csv` |
| Evidence-chain reconciliation | `results/traceability/chain_reconciliation_report.json` |
| Coverage and sparsity | `results/validation/coverage/` |
| Lexical sensitivity aggregates | `results/sensitivity/` |
| LR/RF/XGB/CNN1D metrics and predictions | `results/anova_revalidation/` |
| Classical and CNN1D module ablations | `results/ablation/` |
| Figure provenance | `results/provenance/figure_generation_manifest.json` |

See [docs/RESULTS_TRACEABILITY.md](docs/RESULTS_TRACEABILITY.md) for the
manuscript-to-artifact mapping and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for regeneration commands.

## Data Minimization

The public Parquet is a 47-column projection of the 126-column internal Gold
table. It retains the post text and fields required for enrichment, RDF
materialization, training, and evaluation. Reddit account names, full-name
fields, URLs, interaction metadata, and raw API payloads are removed. Authors
are represented by corpus-local identifiers whose source mapping is not
released.

The public metadata records both the internal source hash referenced by the
stored runs and the hash of the minimized release file. See
[DATA_NOTICE.md](DATA_NOTICE.md) before redistribution or reuse. A
version-specific copy of the Gold corpus is also available from Hugging Face at
[https://doi.org/10.57967/hf/9852](https://doi.org/10.57967/hf/9852).

## Licenses

Source code is released under the MIT License in [LICENSE](LICENSE). That
license does not cover Reddit-authored text or third-party data. The ontology
files retain their embedded license statements. The labels, fixed split
assignments, ontology-derived variables, documentation, and release metadata
created by the authors are licensed under CC BY 4.0. Reddit-authored text
remains subject to the rights of its original authors and applicable platform
terms; see [DATA_NOTICE.md](DATA_NOTICE.md).

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). The Gold corpus
should be cited through its version-specific dataset DOI. The article DOI and a
repository-release DOI can be added after they are assigned.
