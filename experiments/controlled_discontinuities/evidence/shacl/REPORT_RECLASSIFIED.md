# E1 SHACL Subrun Corrected Classification

Created UTC: 2026-09-15T03:09:54.797811+00:00

The preserved pySHACL reports were not rerun. This file corrects only the report classifier by expanding expected `rr:`, `sh:`, `rrdom:`, and `skos:` QNames before comparing them with full IRIs in `report.ttl`.

SHACL is interpreted as conformance to declared shapes only. These cases do not establish semantic correctness of ontology activations.

## Counts

| Measure | Count |
|---|---:|
| total_cases | 11 |
| controls | 1 |
| faults | 10 |
| detected_by_original | 10 |
| control_passed | 1 |
| partial_detection | 0 |
| not_detected | 0 |
| false_positive | 0 |

## Case Results

| Case | Expected alarm | Conforms | Results | Matches expected constraint | Status |
|---|---:|---:|---:|---:|---|
| E1-C0-pristine-shacl-control | False | True | 0 | 0 | control_passed |
| E1-SHACL1-missing-identifier | True | False | 1 | 1 | detected_by_original |
| E1-SHACL10-concept-without-scheme | True | False | 1 | 1 | detected_by_original |
| E1-SHACL2-invalid-identifier-pattern | True | False | 1 | 1 | detected_by_original |
| E1-SHACL3-invalid-language | True | False | 1 | 1 | detected_by_original |
| E1-SHACL4-missing-author | True | False | 1 | 1 | detected_by_original |
| E1-SHACL5-probability-out-of-range | True | False | 1 | 1 | detected_by_original |
| E1-SHACL6-probability-sum | True | False | 1 | 1 | detected_by_original |
| E1-SHACL7-label-probability-inconsistency | True | False | 1 | 1 | detected_by_original |
| E1-SHACL8-invalid-domain-concept | True | False | 2 | 1 | detected_by_original |
| E1-SHACL9-duplicate-rdf-identifier | True | False | 2 | 2 | detected_by_original |

## Constraint Details

### E1-C0-pristine-shacl-control

- No validation results.

### E1-SHACL1-missing-identifier

- Result 1:
  - focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  - path: `http://wfrp.ia/ontologia/ia-sentimiento#identificador`
  - component: `http://www.w3.org/ns/shacl#MinCountConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: rr:identificador debe ser alfanumérico base36.

### E1-SHACL10-concept-without-scheme

- Result 1:
  - focus: `http://wfrp.ia/dominio/ciencia-tecnologia#E1ConceptWithoutScheme`
  - path: `http://www.w3.org/2004/02/skos/core#inScheme`
  - component: `http://www.w3.org/ns/shacl#MinCountConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: Todo skos:Concept debe pertenecer a un skos:ConceptScheme.

### E1-SHACL2-invalid-identifier-pattern

- Result 1:
  - focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  - path: `http://wfrp.ia/ontologia/ia-sentimiento#identificador`
  - component: `http://www.w3.org/ns/shacl#PatternConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: rr:identificador debe ser alfanumérico base36.

### E1-SHACL3-invalid-language

- Result 1:
  - focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  - path: `http://wfrp.ia/ontologia/ia-sentimiento#idiomaDetectado`
  - component: `http://www.w3.org/ns/shacl#InConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: Idioma debe ser 'es' o 'en' (xsd:string, cardinalidad 1).

### E1-SHACL4-missing-author

- Result 1:
  - focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  - path: `http://wfrp.ia/ontologia/ia-sentimiento#publicadoPor`
  - component: `http://www.w3.org/ns/shacl#MinCountConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: rr:publicadoPor debe apuntar a exactamente 1 rr:Author (IRI).

### E1-SHACL5-probability-out-of-range

- Result 1:
  - focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  - path: `http://wfrp.ia/ontologia/ia-sentimiento#probPos`
  - component: `http://www.w3.org/ns/shacl#MaxInclusiveConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: rr:probPos fuera de [0,1] o con tipo inválido.

### E1-SHACL6-probability-sum

- Result 1:
  - focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  - path: `SPARQL/no path`
  - component: `http://www.w3.org/ns/shacl#SPARQLConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: Probabilidades no normalizadas (|sum-1| > 0.02).

### E1-SHACL7-label-probability-inconsistency

- Result 1:
  - focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  - path: `SPARQL/no path`
  - component: `http://www.w3.org/ns/shacl#SPARQLConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: rr:etiquetaSentimiento no coincide con la probabilidad máxima (probPos/probNeu/probNeg).

### E1-SHACL8-invalid-domain-concept

- Result 1:
  - focus: `http://wfrp.ia/dominio/ciencia-tecnologia#E1InvalidConcept`
  - path: `http://www.w3.org/2004/02/skos/core#inScheme`
  - component: `http://www.w3.org/ns/shacl#MinCountConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: Todo skos:Concept debe pertenecer a un skos:ConceptScheme.
- Result 2:
  - focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  - path: `SPARQL/no path`
  - component: `http://www.w3.org/ns/shacl#SPARQLConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: rr:trataSobre/rr:perteneceDominio debe apuntar a skos:Concept en el ConceptScheme de dominio.

### E1-SHACL9-duplicate-rdf-identifier

- Result 1:
  - focus: `http://wfrp.ia/resource/reddit/post/1kkfjn0`
  - path: `SPARQL/no path`
  - component: `http://www.w3.org/ns/shacl#SPARQLConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: Duplicado de rr:identificador detectado en el grafo/dataset.
- Result 2:
  - focus: `http://wfrp.ia/resource/reddit/post/e1_duplicate_identifier_fixture`
  - path: `SPARQL/no path`
  - component: `http://www.w3.org/ns/shacl#SPARQLConstraintComponent`
  - severity: `http://www.w3.org/ns/shacl#Violation`
  - message: Duplicado de rr:identificador detectado en el grafo/dataset.


## Limitation

This SHACL subrun executes only the intact SHACL control plus the ten declared shacl_cases, not the full non-SHACL E1 continuity inventory.
