# Corrected Downstream Campaign

Twenty-four CPU runs use corrected Gold version `information_f123_20260914`,
six selected variables, and seeds 42, 123 and 2024. Eight seed-42 runs were
reused from the matching pilot. LR/CNN1D use training-standardized semantic
inputs; RF/XGB use raw values. BSL/ENR CNN1D share the hidden-128 classifier
head. Test support is 48 negative, 119 neutral and 156 positive records.

Gold SHA-256: `8730955465e71bd9782d6f9bd42cc240f0dc80c6af7432db1f57299010bbacce`.
Lexicon SHA-256: `8bd5d395007819b41ff4a1615004a62d9ce43c4383f18408047246bf201c7a53`.
All 24 prediction files contain 323 fixed test records. The full-campaign
summary used by Figure 7 is `results/provenance/campaign/main_summary.csv`.
Per-run metrics and the original multiseed tables are retained unchanged.

Model-input profiles captured during the runs are in metric JSON files.
The released input scaler supports matrix checks; classifier weights are
not included. Publishing or verifying this package does not rerun training.
