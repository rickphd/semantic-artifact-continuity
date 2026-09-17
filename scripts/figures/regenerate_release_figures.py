#!/usr/bin/env python3
"""Replay Figures 5-11 without overwriting the released figures."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import generate_results_plots as plots

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT):
        parser.error('Use a scratch output directory outside the released checkout.')
    out.mkdir(parents=True, exist_ok=True)
    plots.FIGURE_OUTPUTS = {key: out / value.name for key, value in plots.FIGURE_OUTPUTS.items()}
    plots.FEATURE_PRESENTATION.update({
        'ont_InteligenciaArtificial_Positivo': ('Positive AI evidence', 'InteligenciaArtificial', 'Positive local evidence'),
        'ont_AprendizajeAutomatico_Positivo': ('Positive machine-learning evidence', 'AprendizajeAutomatico', 'Positive local evidence'),
    })
    # Match the frozen display wrapper: rejected lexicons have no evaluated RDF.
    source = inspect.getsource(plots.plot_lexical_sensitivity)
    original = 'failed = data[data["variant_status"] != "passed"].copy()'
    if source.count(original) != 1:
        raise ValueError('Plot implementation changed; recheck the admissible-only display rule.')
    exec(compile(source.replace(original, 'failed = data.iloc[0:0].copy()'), __file__, 'exec'), plots.__dict__)
    plots.plot_macro_f1(plots.pd.read_csv(ROOT / 'results/provenance/campaign/main_summary.csv'))
    plots.plot_confusion_delta()
    plots.plot_module_sensitivity()
    plots.plot_lexical_sensitivity()
    plots.plot_module_coverage()
    plots.plot_prediction_distribution()
    plots.plot_feature_stability()
    plots.PRESENTATION_MAP = out / 'semantic_variable_presentation_map.md'
    plots.write_presentation_map()
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in plots.FIGURE_OUTPUTS.values()}
    (out / 'replay_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({'figures': len(hashes), 'output': str(out), 'training_performed': False}))

if __name__ == '__main__':
    main()
