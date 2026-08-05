# Evidence Chain Reconciliation

Status: `passed`

This report reconciles the active evidence lineage at the level of scientific records and evidence relations.

| Evidence relation | Unit | Expected | Verified | Exceptions | Status |
|---|---:|---:|---:|---:|---|
| Canonical records to RDF resources | canonical records | 1614 | 1614 | 0 | passed |
| Selected semantic activations to source records | nonzero selected activations | 2425 | 2425 | 0 | passed |
| Selected semantic input cells to governed source values | selected input cells | 11298 | 11298 | 0 | passed |
| RDF graph to declared structural constraints | baseline graph | 1 | 1 | 0 | passed |
| Test records to prediction records | record assignments across evaluation units | 7752 | 7752 | 0 | passed |

## Interpretation

- Each canonical record is represented by one stable RDF resource.
- Each selected activation retains a record, semantic-variable, module, and RDF-resource link; domain variables additionally retain their concept links.
- All cells in the seven-variable interface retain ordered record and variable identity; every nonzero value matches its traceability relation.
- The baseline RDF graph conforms to the declared structural constraints.
- All 323 test records are reconciled across 24 evaluation units with matching reference labels.
