# Final E2 human semantic correspondence

This package replays the final v2.2 evaluation of F123 ontology activations on
120 validation posts and ten concepts (1,200 post/concept units). It contains
anonymous A/B labels, 77 final author adjudications with explicit AI-assistance
attribution, the final guide, blinded texts, item/post map, consolidated reference,
pre-adjudication agreement, and extraction comparisons and metrics.

## Reproduce

From the repository root, use Python 3.12 or newer and install the two pinned
dependencies in a virtual environment. The tested release environment used
Python 3.13.13; label/reference/agreement replay otherwise uses the standard library.
Do not use Python's `-O` mode: integrity and equality checks use assertions.

```sh
python -m pip install -r experiments/human_assessment/requirements.txt
python -B experiments/human_assessment/build_reference.py --output-dir /tmp/e2-reference-replay
python -B experiments/human_assessment/analyze.py --output-dir /tmp/e2-agreement-replay
python -B experiments/human_assessment/evaluate.py --output-dir /tmp/e2-evaluation-replay
```

Each output directory must be new and outside the repository. Use a different
directory name for another run. The evaluator also rebuilds the reference and
agreement, verifies the package manifest and canonical Gold hash, and requires
exact equality of every public reference row, agreement statistic, comparison,
and overall/per-concept summary. It writes only to the requested scratch folder.
Gold defaults to `data/gold/gold_enriched_ontology.parquet` relative to the
repository; `--gold PATH` can explicitly select the same canonical bytes.
No private directory, old annotation round, manuscript, or original local path is
needed. This is a replay of fixed labels and activations, not fresh annotation,
extraction, training, or an independent replication of human judgments.

## Files and rules

- `annotations_A.json`, `annotations_B.json`: structured original final labels
  and language; A/B are anonymous roles.
- `adjudications.json`: 77 keyed decisions, original A/B labels, closure status,
  anonymous author role, AI-assistance mode, recorded rationale-origin metadata,
  and rejected assistant label recommendations where present.
- `human_reference.json`: 1,123 A/B agreement-derived decisions plus 77 assisted
  decisions. An adjudication takes precedence, including 20 shared-uncertainty
  cases in addition to 57 disagreements. Four final uncertainties remain.
- `items_blinded.json`, `sampling_manifest.json`, `GUIA_E2_FINAL_V22.md`: exact
  blinded source texts, minimized map, and exact Spanish guide. Historical form
  instructions remain part of the guide; the form is not distributed here.
- `agreement_summary.json`: full overall and per-concept pre-adjudication
  agreement matrices, marginals, raw agreement and Cohen's kappa. Presence uses
  all three categories; polarity uses only both-present cases and retains
  insufficient polarity as a category. Undefined kappa remains null.
- `comparisons.json`, `summary.json`: exact final scientific evaluation files.
  Presence is any concept-polarity count greater than zero. Four uncertain
  presences are excluded, not recoded absent. Mixed means {positive, negative};
  exact polarity uses set equality, not argmax. Human-present misses stay in the
  principal polarity denominator. Undefined precision remains null.
- `reference_freeze.json`, `provenance.json`, `MANIFEST.sha256`: distinct original
  and public hashes, relative source/public mapping, deterministic projection
  version and omitted fields. Original source paths identify provenance only;
  they are not public runtime dependencies.
- `agreement_verification.json`, `historical_evaluation_verification.json`:
  selected historical checks, not claims of a new validation run.
- `project_sources.py`: maintainer-only allowlist projection from the authorized
  private final records. Public users do not need to run it. It refuses existing
  projected files. Portable script adaptations are mapped to historical source
  script hashes; new code is not claimed to have produced historical bytes.

## Expected results

Presence agreement: 1,160/1,200, kappa 0.9223627600038818. Conditional polarity
agreement: 288/305, kappa 0.910193123755088. These are pre-adjudication statistics,
not agreement with the final author-assisted reference.

The reference has 874 absent, 322 present and four uncertain units. Extraction
over 1,196 certain units gives TP=117, FP=13, FN=205, TN=861; precision 0.9,
recall 0.36335403726708076 and F1 0.5176991150442478. Exact polarity is 52/322
(0.16149068322981366); detected-only 52/117 is supplementary. The 322 units
comprise 52 exact sets, 65 detected polarity mismatches and 205 missed concepts.

## Privacy and interpretation

The public projection omits participant notes, evidence excerpts, justifications,
adjudication rationale prose, coordinator notes, private statements, identifying
metadata, export timestamps, and the single nested history field. Omitted fields
are absent, not replaced by invented text or empty strings. All structured final
scientific decisions are retained. Rationale-origin strings preserve whether an
assistant recommendation/clarification was accepted; absent original origin
metadata is not inferred. All 77 adjudications are explicitly author/AI-assisted,
not a third independent annotation. Thirteen rejected AI label recommendations
remain distinguishable from the author's final decisions.

Source post IDs and text are unchanged and remain linkable to public posts; this
is not anonymization of source content. The projection supports label-based
replay, not reproduction of evidence-quotation checks or a rationale audit.
Historical free-text checks are therefore not included as newly reproducible
results. Participant process confirmations and private correspondence are omitted.

The same participants reannotated the same texts with new ordering/IDs; recall
is possible and this is not a new independent sample or a causal test of guide
changes. No exhaustive semantic audit of the 1,123 agreements was performed.
R113/Algoritmo retains the original present/neutral agreement; adjudication of
another concept does not propagate a label. Fifteen adjudications had no additional
author rationale; this release does not invent one. Small concept denominators
(two human presences each for ML and Robot) limit interpretation. These metrics
measure correspondence to this reference, not SHACL correctness, universal
semantic accuracy, or downstream sentiment performance.
