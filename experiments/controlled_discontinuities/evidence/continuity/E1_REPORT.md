# E1 Continuity Fault Injection Results

Status: executed continuity controls and non-SHACL fault cases from `E1_CASES.json`.

Scope: this run excludes the independent SHACL fixture suite. It uses isolated per-case copies and does not train models or mutate the freeze, campaign, public release, root protocol files, or `.tex` files.

## Denominators

- Total executed cases: 26
- Valid controls: 4
- Continuity fault cases: 22
- Independent SHACL fixture cases: 0 in this run; delegated separately.
- Cases with an additional operational error after a contract signal: 1
- Fixture correction note: `E1-I3-add-extra-valid-rdf-post` had an initial invalid fixture attempt caused by a typed language literal; the case was rerun with the plain literal required by the existing shape. The overwritten initial logs are not counted as preserved executable evidence; see `cases/E1-I3-add-extra-valid-rdf-post/invalid_fixture_attempt_note.json`.

## Detection Summary

| Status | Cases |
|---|---:|
| `control_passed` | 4 |
| `detected_by_extension` | 7 |
| `detected_by_original` | 15 |

## Family Summary

| Family | Cases | Original detected | Extension detected | Misses | False positives |
|---|---:|---:|---:|---:|---:|
| `column_alignment` | 2 | 1 | 1 | 0 | 0 |
| `identity` | 3 | 1 | 2 | 0 | 0 |
| `partition` | 2 | 2 | 1 | 0 | 0 |
| `prediction_assignment` | 2 | 1 | 2 | 0 | 0 |
| `rdf_link` | 3 | 2 | 2 | 0 | 0 |
| `row_alignment` | 2 | 1 | 2 | 0 | 0 |
| `stale_report` | 2 | 2 | 1 | 0 | 0 |
| `uniqueness` | 3 | 3 | 1 | 0 | 0 |
| `valid_control` | 4 | 0 | 0 | 0 | 0 |
| `value` | 3 | 2 | 2 | 0 | 0 |

## Case Results

| Case | Family | Oracle alarm | Original before refresh | Original after refresh | Extension | Status |
|---|---|---:|---:|---:|---:|---|
| `E1-C0-pristine-copy` | `valid_control` | False | False | False | False | `control_passed` |
| `E1-C1-trace-row-permutation` | `valid_control` | False | False | False | False | `control_passed` |
| `E1-C2-rdf-triple-order-reserialization` | `valid_control` | False | False | False | False | `control_passed` |
| `E1-C3-regenerated-reports-unchanged-inputs` | `valid_control` | False | False | False | False | `control_passed` |
| `E1-I1-remove-selected-trace-row` | `identity` | True | True | True | False | `detected_by_original` |
| `E1-I2-remove-rdf-post-resource` | `identity` | True | False | False | True | `detected_by_extension` |
| `E1-I3-add-extra-valid-rdf-post` | `identity` | True | False | False | True | `detected_by_extension` |
| `E1-U1-duplicate-trace-key` | `uniqueness` | True | True | True | False | `detected_by_original` |
| `E1-U2-duplicate-split-id` | `uniqueness` | True | False | False | False | `detected_by_original` |
| `E1-U3-duplicate-prediction-id` | `uniqueness` | True | True | True | True | `detected_by_original` |
| `E1-R1-swap-selected-matrix-rows` | `row_alignment` | True | False | False | True | `detected_by_extension` |
| `E1-R2-reorder-predictions-against-contract` | `row_alignment` | True | False | True | True | `detected_by_original` |
| `E1-COL1-swap-matrix-columns` | `column_alignment` | True | False | False | True | `detected_by_extension` |
| `E1-COL2-unknown-selected-feature` | `column_alignment` | True | True | True | False | `detected_by_original` |
| `E1-V1-corrupt-selected-gold-cell` | `value` | True | False | True | True | `detected_by_original` |
| `E1-V2-corrupt-trace-value` | `value` | True | False | True | False | `detected_by_original` |
| `E1-V3-corrupt-standardized-matrix-cell` | `value` | True | False | False | True | `detected_by_extension` |
| `E1-P1-move-id-between-splits` | `partition` | True | True | True | False | `detected_by_original` |
| `E1-P2-nontest-prediction-row` | `partition` | True | True | True | True | `detected_by_original` |
| `E1-RDF1-break-trace-post-iri` | `rdf_link` | True | True | True | False | `detected_by_original` |
| `E1-RDF2-remove-trataSobre-for-active-concept` | `rdf_link` | True | False | False | True | `detected_by_extension` |
| `E1-RDF3-invalid-domain-concept-iri` | `rdf_link` | True | False | True | True | `detected_by_original` |
| `E1-PR1-wrong-y-true` | `prediction_assignment` | True | True | True | True | `detected_by_original` |
| `E1-PR2-permute-y-pred-only` | `prediction_assignment` | True | False | False | True | `detected_by_extension` |
| `E1-SR1-stale-traceability-success-report` | `stale_report` | True | True | True | False | `detected_by_original` |
| `E1-SR2-stale-shacl-success-report` | `stale_report` | True | False | True | True | `detected_by_original` |

## Operational Errors Kept Separate

- `E1-COL2-unknown-selected-feature`: detector status remains `detected_by_original` because another original script signaled the contract fault, but an additional script produced an operational error. Inspect the case `original_detector_result.json` before using this case as evidence of clean localization.
- `E1-I3-add-extra-valid-rdf-post`: the initial fixture accidentally violated `rr:idiomaDetectado` because it used an `xsd:string` typed literal while the shape allows a plain literal. The corrected rerun is the counted result: original scripts did not detect the extra valid post, while the RDF inventory diagnostic did.

## Interpretation Limits

- `detected_by_original` means at least one original script returned a nonzero code or wrote an attention/failure status after the mutation was applied.
- `original_detected_before_refresh` preserves the stale-consumer phase. Several cases only become visible after a producer is rerun against the mutated artefact.
- Extension diagnostics are reported separately and should not be described as behavior of the submitted detector.
- Matrix and prediction-output cases expose gaps in the original detector surface when it relies on metadata, IDs, or labels but not the mutated matrix/prediction assignment bytes.
- No result here validates semantic correctness of ontology activations or SHACL as truth.
