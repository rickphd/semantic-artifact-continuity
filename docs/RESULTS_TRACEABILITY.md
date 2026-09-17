# Result Traceability

| Result | Public evidence | Boundary |
| --- | --- | --- |
| 1,614 posts; 15,616 triples; 9,329 nonzero variable values | `results/knowledge_graph/`, `data/gold/` | RDF is a projection, not a semantic truth test. |
| 37 variables, six selected using 968 training records | `results/feature_selection/`, `results/lexicon/` | Held-out texts and labels are excluded from induction/selection as specified. |
| 1,789 nonzero selected values; 9,684 selected cells including zeros | `results/prepared_model_inputs/`, `results/traceability/model_inputs/` | Prepared matrices and runtime consumption evidence are distinct. |
| 27 runtime profiles match prepared profiles | Main metric JSON `semantic_input_provenance`; prepared manifest | These are profile records, not 27 model runs. |
| 24 main, 126 classical-ablation, 42 CNN1D-ablation runs | `results/anova_revalidation/`, `results/ablation/` | Eight main seed-42 runs are recorded pilot reuse. |
| 125 sensitivity settings: 97 admissible, 28 rejected | `results/sensitivity/`; `results/provenance/sensitivity_audit/` | Rejected lexicons have no evaluated RDF or SHACL outcome. |
| 10 isolated SHACL faults and one control | `experiments/controlled_discontinuities/` | Reclassified constraint names derive from preserved reports. |
| 22 continuity faults and four controls | `experiments/controlled_discontinuities/` | 15 detections by original checks; seven only by supplementary checks. |
| 120 texts and 1,200 human units | `experiments/human_assessment/` | Ten concepts; 1,123 agreement-derived reference units and 77 adjudications. |
| Presence TP/FP/FN/TN = 117/13/205/861 | Human-assessment comparisons and summary | Four uncertain units excluded; 1,196 definite units. |
| Exact local polarity 52/322 | Human-assessment summary | Denominator includes human-present units missed by the extractor; 52/117 is supplementary detected-only agreement. |
| Current Figures 4-11 | `figures/`; `results/provenance/figures/` | Historical output/source hashes are distinct from public adaptation hashes. |

Computational source/public pairs are in `results/provenance/public_projection.json`.
Experiment projections preserve their own source/public provenance. A path
projection may change a file hash without changing its metrics or decisions.
Historical source paths are identifiers, not promises that private working
directories are shipped. The old seven-variable/15,603-triple package is retained
only under tag `v1.0.0`; it is not the active result set.
