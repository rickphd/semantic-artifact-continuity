# Final E2 results

This public summary reports the final v2.2 reference only. Numeric details and
per-concept results are preserved exactly in `summary.json` and keyed outcomes
in `comparisons.json`.

The 120 validation posts yield 1,200 post/concept units: 874 absent, 322 present,
and four uncertain. Reference origins are 1,123 independent A/B agreements and
77 author decisions assisted by AI. The latter are not independent annotations.

Presence correspondence over the 1,196 certain units is TP=117, FP=13, FN=205,
TN=861: precision 90.00%, recall 36.34%, and F1 51.77%. The 205 missed concepts
are 63.66% of human presences; high precision alone does not establish broad
semantic coverage.

Exact local polarity is 52/322 (16.15%). This includes missed concepts in the
denominator: 52 exact sets + 65 detected polarity mismatches + 205 misses = 322.
The detected-only diagnostic is 52/117 (44.44%), not the principal metric.
Mixed human polarity is the set {positive, negative}; additional automatic
labels count as mismatches. No thresholds, lexicons, or models are changed.

Pre-adjudication presence agreement is 1,160/1,200 (kappa 0.9223627600038818).
Polarity agreement conditional on both annotators marking presence is 288/305
(kappa 0.910193123755088). No post-adjudication agreement is substituted.

The same participants reannotated the same texts, so recall is possible and
guide effects are not causally isolated. Fifteen adjudications lack an additional
author rationale; no explanation is invented. R113/Algoritmo retains the original
present/neutral agreement. The 1,123 agreements were not exhaustively audited for
semantic correctness. ML and Robot each have only two human presences. These
results characterize extraction correspondence to this reference and are distinct
from structural SHACL conformance and downstream classification performance.

This report adapts the final evaluation's `E2_RESULTS.md` by retaining final
findings and interpretation limits, excluding prior-round comparisons, private
process confirmation, editorial discussion, and local execution paths. Public
label replay does not reproduce omitted evidence/rationale audits. See README
and provenance for the exact release transformations.
