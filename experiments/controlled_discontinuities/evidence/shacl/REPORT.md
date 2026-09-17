# E1 SHACL Subrun Report

Created UTC: 2026-09-15T03:07:11.190174+00:00

## Scope

This subrun executes the intact SHACL control plus the 10 `shacl_cases` declared in `E1_CASES.json`. It does not execute the non-SHACL continuity families; those require separate continuity harnesses.

SHACL is interpreted only as conformance to the declared shapes in `rr-shapes.ttl`. The run does not validate semantic correctness of extracted ontology activations.

## Summary Counts

| Measure | Count |
|---|---:|
| total_cases | 11 |
| controls | 1 |
| faults | 10 |
| detected_by_original | 0 |
| control_passed | 1 |
| partial_detection | 10 |
| not_detected | 0 |
| false_positive | 0 |

## Cases

| Case | Expected alarm | Conforms | Results | Status | Localization |
|---|---:|---:|---:|---|---|
| E1-C0-pristine-shacl-control | False | True | 0 | control_passed | not_applicable |
| E1-SHACL1-missing-identifier | True | False | 1 | partial_detection | detected_without_expected_path_component_match |
| E1-SHACL2-invalid-identifier-pattern | True | False | 1 | partial_detection | detected_without_expected_path_component_match |
| E1-SHACL3-invalid-language | True | False | 1 | partial_detection | detected_without_expected_path_component_match |
| E1-SHACL4-missing-author | True | False | 1 | partial_detection | detected_without_expected_path_component_match |
| E1-SHACL5-probability-out-of-range | True | False | 1 | partial_detection | detected_without_expected_path_component_match |
| E1-SHACL6-probability-sum | True | False | 1 | partial_detection | detected_without_expected_path_component_match |
| E1-SHACL7-label-probability-inconsistency | True | False | 1 | partial_detection | detected_without_expected_path_component_match |
| E1-SHACL8-invalid-domain-concept | True | False | 2 | partial_detection | detected_without_expected_path_component_match |
| E1-SHACL9-duplicate-rdf-identifier | True | False | 2 | partial_detection | detected_without_expected_path_component_match |
| E1-SHACL10-concept-without-scheme | True | False | 1 | partial_detection | detected_without_expected_path_component_match |

## Constraint Details

### E1-C0-pristine-shacl-control

- No validation results.

### E1-SHACL1-missing-identifier

- focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  path: `http://wfrp.ia/ontologia/ia-sentimiento#identificador`
  component: `http://www.w3.org/ns/shacl#MinCountConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: rr:identificador debe ser alfanumérico base36.

### E1-SHACL2-invalid-identifier-pattern

- focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  path: `http://wfrp.ia/ontologia/ia-sentimiento#identificador`
  component: `http://www.w3.org/ns/shacl#PatternConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: rr:identificador debe ser alfanumérico base36.

### E1-SHACL3-invalid-language

- focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  path: `http://wfrp.ia/ontologia/ia-sentimiento#idiomaDetectado`
  component: `http://www.w3.org/ns/shacl#InConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: Idioma debe ser 'es' o 'en' (xsd:string, cardinalidad 1).

### E1-SHACL4-missing-author

- focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  path: `http://wfrp.ia/ontologia/ia-sentimiento#publicadoPor`
  component: `http://www.w3.org/ns/shacl#MinCountConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: rr:publicadoPor debe apuntar a exactamente 1 rr:Author (IRI).

### E1-SHACL5-probability-out-of-range

- focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  path: `http://wfrp.ia/ontologia/ia-sentimiento#probPos`
  component: `http://www.w3.org/ns/shacl#MaxInclusiveConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: rr:probPos fuera de [0,1] o con tipo inválido.

### E1-SHACL6-probability-sum

- focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  path: `none`
  component: `http://www.w3.org/ns/shacl#SPARQLConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: Probabilidades no normalizadas (|sum-1| > 0.02).

### E1-SHACL7-label-probability-inconsistency

- focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  path: `none`
  component: `http://www.w3.org/ns/shacl#SPARQLConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: rr:etiquetaSentimiento no coincide con la probabilidad máxima (probPos/probNeu/probNeg).

### E1-SHACL8-invalid-domain-concept

- focus: `http://wfrp.ia/dominio/ciencia-tecnologia#E1InvalidConcept`
  path: `http://www.w3.org/2004/02/skos/core#inScheme`
  component: `http://www.w3.org/ns/shacl#MinCountConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: Todo skos:Concept debe pertenecer a un skos:ConceptScheme.
- focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  path: `none`
  component: `http://www.w3.org/ns/shacl#SPARQLConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: rr:trataSobre/rr:perteneceDominio debe apuntar a skos:Concept en el ConceptScheme de dominio.

### E1-SHACL9-duplicate-rdf-identifier

- focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  path: `none`
  component: `http://www.w3.org/ns/shacl#SPARQLConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: Duplicado de rr:identificador detectado en el grafo/dataset.
- focus: `http://wfrp.ia/resource/reddit/post/e1_duplicate_identifier_fixture`
  path: `none`
  component: `http://www.w3.org/ns/shacl#SPARQLConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: Duplicado de rr:identificador detectado en el grafo/dataset.

### E1-SHACL10-concept-without-scheme

- focus: `http://wfrp.ia/dominio/ciencia-tecnologia#E1ConceptWithoutScheme`
  path: `http://www.w3.org/2004/02/skos/core#inScheme`
  component: `http://www.w3.org/ns/shacl#MinCountConstraintComponent`
  severity: `http://www.w3.org/ns/shacl#Violation`
  message: Todo skos:Concept debe pertenecer a un skos:ConceptScheme.

## Reproducibility

- Script: `experiments/controlled_discontinuities/evidence/shacl/run_e1_shacl.py`
- Data graph baseline SHA-256: `fdbac59aa40039e1134863571fa2653fdf0b326b88d8a178dd0cf942ad0c0a55`
- Shapes baseline SHA-256: `36f9b52a41df2f96e30620b1965a0f5acdb474e2e17e0ed6be2ec0a3d40d4afc`
- Python executable: `${PYTHON}`
- rdflib: `7.6.0`
- pySHACL: `0.40.1`
