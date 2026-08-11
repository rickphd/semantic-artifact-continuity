# Artifact Manifest

## Data And Frozen Inputs

| Path | Role |
|---|---|
| `data/gold/gold_enriched_ontology.parquet` | Minimized v04.2 release table with text, pseudonymous author IDs, labels, 37 ontology variables, and split assignments. |
| `data/gold/gold_enriched_ontology_metadata.json` | Source/release hashes, distributions, and data-minimization record. |
| `data/gold/GEN_split_gld_reddit_ids_v02.json` | Fixed ID-based train/validation/test split. |
| `results/lexicon/ontology_lexicon_v04_2_train_only.json` | Frozen train-scope VADER lexical manifest. |
| `results/feature_selection/ENR_selected_ont_features_anova_train_only.json` | Train-only ANOVA selection of seven semantic variables. |

## Ontology And Governance

| Path | Role |
|---|---|
| `src/ontology/resources/rr-core.ttl` | Core Reddit and provenance vocabulary. |
| `src/ontology/resources/rr-domain.ttl` | AI and technology domain concepts. |
| `src/ontology/resources/rr-sentiment.ttl` | Sentiment vocabulary. |
| `src/ontology/resources/rr-shapes.ttl` | Canonical SHACL constraints. |
| `results/knowledge_graph/posts.ttl` | Materialized post-level RDF graph. |
| `results/validation/shacl/` | Normalized shapes, baseline report, invalid fixture, and stress report. |

## Traceability And Evaluation

| Path | Role |
|---|---|
| `results/traceability/traceability_map.csv` | Post-to-variable and post-to-RDF trace rows. |
| `results/traceability/model_inputs/selected_semantic_input_matrix.csv` | Persisted selected seven-variable matrix. |
| `results/traceability/model_inputs/selected_semantic_input_standardized.csv` | Train-standardized selected matrix. |
| `results/traceability/model_inputs/selected_semantic_input_cell_trace.csv` | Cell-level source, value, split, and RDF links. |
| `results/traceability/model_inputs/downstream_semantic_input_fingerprints.csv` | Ordered row, feature, and matrix fingerprints. |
| `results/traceability/model_inputs/selected_semantic_input_reconciliation_report.json` | Cell and run reconciliation summary. |
| `results/traceability/chain_reconciliation_report.json` | Five relation-level checks across the active chain. |
| `results/validation/coverage/coverage_sparsity_report.json` | Dataset and selected-interface activation coverage. |
| `results/sensitivity/` | Aggregate output for the 125-condition KE-only sensitivity grid. |

## Downstream Evidence

| Path | Role |
|---|---|
| `results/anova_revalidation/metrics/` | LR, RF, XGB, and CNN1D metrics for BSL/ENR across three seeds. |
| `results/anova_revalidation/predictions/` | Corresponding test predictions. |
| `results/ablation/classical/` | Full-37 classical model module ablations. |
| `results/ablation/cnn1d/` | Full-37 CNN1D module ablation used by the manuscript. |

The release contains the active v04.2 lineage and its authoritative output
families. Serialized model binaries are not distributed, except for the small
LR semantic-input scaler required for matrix reconciliation.
