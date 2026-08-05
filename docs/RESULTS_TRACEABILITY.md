# Results Traceability

This map identifies the released source for the principal quantitative
statements in the manuscript.

| Manuscript evidence | Released source | Interpretation limit |
|---|---|---|
| 1,614 RDF post resources, 15,603 triples, 9,208 nonzero activations, and 1,888 concept assertions | `results/knowledge_graph/materialization_report.json`, `results/knowledge_graph/posts.ttl` | RDF materialization does not establish semantic correctness. |
| 11,298 selected cells and 2,425 nonzero selected activations reconciled | `results/traceability/model_inputs/selected_semantic_input_reconciliation_report.json`, `results/traceability/model_inputs/selected_semantic_input_cell_trace.csv` | Traceability is at post, variable, and cell level; token/span provenance is not claimed. |
| Baseline SHACL conformance and six controlled validation results | `results/validation/shacl/baseline_report.json`, `results/validation/shacl/stress_test_report.json` | SHACL tests declared constraints, not factual truth. |
| 92.26% any-variable coverage and 87.24% selected-variable coverage | `results/validation/coverage/coverage_sparsity_report.json`, `results/validation/coverage/module_coverage.csv` | Coverage is descriptive, not a quality threshold. |
| 125 lexical variants, 104 valid and 21 invalid | `results/sensitivity/vader_ke_sensitivity_summary.csv` | Models were not retrained across this grid. |
| Two variables selected in all 104 valid lexical variants | `results/sensitivity/selected_feature_stability.csv` | Stability is conditional on the tested grid. |
| Selected-seven LR/RF/XGB/CNN1D comparison across three seeds | `results/anova_revalidation/metrics/multiseed_summary_v2.csv` | Effects are model-dependent and do not support universal superiority. |
| 126 classical and 42 CNN1D full-37 ablation runs | `results/ablation/classical/module_ablation_raw.csv`, `results/ablation/cnn1d/cnn1d_module_ablation_raw.csv` | Module removal is model-side sensitivity, not causal concept importance. |
| Five chain relations with zero exceptions | `results/traceability/chain_reconciliation_report.json` | The result is bounded to the five declared relations. |
| Figures 4-11 | `results/provenance/figure_generation_manifest.json` | The manifest records exact figure and input hashes. |

The stored run metadata references the SHA-256 hash of the original 126-column
Gold table. The public 47-column projection has a different file hash because
unused and identifying columns were removed. Both hashes and the projection
policy are recorded in
`data/gold/gold_enriched_ontology_metadata.json`.
