# Module Ablation Summary

Status: `passed` for LR/RF/XGB local ablation.

Models: `LR, RF, XGB`
Seeds: `42, 123, 2024`
Conditions: `14`
Runtime seconds: `209.0956`

## Interpretation Limits

- This is a model-side ablation of semantic variables, not causal evidence.
- CNN1D is evaluated separately on the same canonical dataset.
- Module contribution is interpreted through delta versus full 37-feature ENR.

## Outputs

- `module_ablation_raw.csv`
- `module_ablation_summary.csv`
- `module_ablation_module_rank.csv`
- `module_ablation_predictions.csv`
- `module_ablation_summary.json`
