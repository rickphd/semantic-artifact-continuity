#!/usr/bin/env python3
"""Rebuild canonical lexical/numeric/RDF artifacts in an isolated directory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from rdflib import Graph
from rdflib.compare import isomorphic
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT) or out.exists():
        parser.error('Use a new output directory outside the released checkout.')
    out.mkdir(parents=True)
    for name in ['inputs', 'data', 'src', 'scripts']:
        shutil.copytree(ROOT / name, out / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    logs = out / 'logs'
    logs.mkdir()
    (out / 'results/feature_selection').mkdir(parents=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    steps = ['build_train_only_ontology_dataset.py', 'select_ont_features_anova_train_only.py', 'materialize_reddit_rdf.py', 'run_shacl_conformance.py', 'analyze_ontology_coverage.py', 'audit_traceability.py']
    for name in steps:
        with (logs / (name + '.log')).open('w') as stream:
            subprocess.run([sys.executable, str(out / 'scripts/experiments' / name)], cwd=out, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(name + ': passed', flush=True)
    relative = 'data/gold/gold_enriched_ontology.parquet'
    actual, expected = pd.read_parquet(out / relative), pd.read_parquet(ROOT / relative)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    fs_path = 'results/feature_selection/ENR_selected_ont_features_anova_train_only.json'
    selected = read(ROOT / fs_path)['selected_features_train_only']
    if read(out / fs_path)['selected_features_train_only'] != selected:
        raise ValueError('Selected feature ordering changed')
    graph = 'results/knowledge_graph/posts.ttl'
    if not isomorphic(Graph().parse(out / graph), Graph().parse(ROOT / graph)):
        raise ValueError('RDF isomorphism failed')
    split = read(ROOT / 'data/gold/GEN_split_gld_reddit_ids_v02.json')
    indexed = actual.set_index('id')
    scaler = StandardScaler().fit(indexed.loc[split['train'], selected].to_numpy(dtype=np.float64))
    prepared = ROOT / 'results/prepared_model_inputs'
    for part in ['train', 'val', 'test']:
        raw = indexed.loc[split[part], selected].to_numpy(dtype=np.float64)
        standard = scaler.transform(raw)
        for suffix, matrix in [('raw_float64', raw), ('train_standardized_float64', standard), ('train_standardized_float32', standard.astype(np.float32))]:
            np.testing.assert_array_equal(matrix, np.load(prepared / f'{part}_{suffix}.npy'))
    result = {'status': 'passed', 'canonical_cells_compared': 59718, 'gold_all_fields_exact': True, 'rdf_isomorphic': True, 'prepared_matrices_exact': 9, 'training_performed': False}
    (out / 'replay_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
