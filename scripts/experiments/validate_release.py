#!/usr/bin/env python3
"""Read-only checks of the corrected computational release, without training."""
from pathlib import Path
import hashlib
import itertools
import json
from collections import Counter

import numpy as np
import pandas as pd
from rdflib import Graph
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
GOLD_SHA = '8730955465e71bd9782d6f9bd42cc240f0dc80c6af7432db1f57299010bbacce'
LEXICON_SHA = '8bd5d395007819b41ff4a1615004a62d9ce43c4383f18408047246bf201c7a53'
SELECTED = [
    'ont_domain_density', 'ont_total_negativo_mentions',
    'ont_InteligenciaArtificial_Negativo', 'ont_Innovacion_Neutro',
    'ont_Etica_Negativo', 'ont_Tecnologia_Neutro',
]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate():
    gold_path = ROOT / 'data/gold/gold_enriched_ontology.parquet'
    require(sha(gold_path) == GOLD_SHA, 'Canonical Gold hash')
    gold = pd.read_parquet(gold_path)
    require(gold.shape == (1614, 47), 'Gold shape')
    require(gold['id'].is_unique, 'Unique Gold IDs')
    require(gold.split.value_counts().to_dict() == {'train': 968, 'val': 323, 'test': 323}, 'Split counts')
    require(not {'autor', 'autor_nombre_completo', 'raw_payload'} & set(gold.columns), 'Private data fields')
    ontology = [c for c in gold if c.startswith('ont_')]
    require(len(ontology) == 37, 'Semantic variable count')
    require(int((gold[ontology] != 0).sum().sum()) == 9329, 'Nonzero semantic cells')
    require(int((gold[SELECTED] != 0).sum().sum()) == 1789, 'Nonzero selected cells')
    original = pd.read_parquet(ROOT / 'inputs/original_gold.parquet')
    pd.testing.assert_frame_equal(gold[[c for c in gold if c not in ontology]], original[[c for c in gold if c not in ontology]])
    split = read(ROOT / 'data/gold/GEN_split_gld_reddit_ids_v02.json')
    gold = gold.set_index('id')
    require(set().union(*(set(split[p]) for p in ['train', 'val', 'test'])) == set(gold.index), 'Split identity coverage')
    for part in ['train', 'val', 'test']:
        ids = split[part]
        require(gold.loc[ids, 'split'].eq(part).all(), 'Split membership ' + part)
    lexicon_path = RESULTS / 'lexicon/ontology_lexicon_v04_2_train_only.json'
    require(sha(lexicon_path) == LEXICON_SHA, 'Lexicon hash')
    lexicon = read(lexicon_path)
    require(lexicon['induction_scope'] == 'train_text_only_no_labels', 'Lexicon scope')
    require(lexicon['induction_counts']['validation_ids_used'] == lexicon['induction_counts']['test_ids_used'] == 0, 'No held-out induction')
    fs = read(RESULTS / 'feature_selection/ENR_selected_ont_features_anova_train_only.json')
    require(fs['selected_features_train_only'] == SELECTED and fs['train_size'] == 968 and fs['n_candidates'] == 37, 'Selected interface')
    require(fs['dataset_sha256'] == GOLD_SHA and fs['lexicon_manifest_sha256'] == LEXICON_SHA, 'Selection lineage')
    rdf = read(RESULTS / 'knowledge_graph/materialization_report.json')
    require(rdf['rdf_triple_count'] == 15616, 'RDF report cardinality')
    require(len(Graph().parse(RESULTS / 'knowledge_graph/posts.ttl')) == 15616, 'Actual RDF cardinality')
    require(read(RESULTS / 'validation/shacl/baseline_report.json')['status'] == 'passed', 'Baseline conformance')
    profiles = read(RESULTS / 'prepared_model_inputs/manifest.json')['profiles']
    fingerprints = 0
    rows = []
    labels = gold.loc[split['test'], 'label'].tolist()

    def predictions(frame, row):
        require(frame.id.tolist() == split['test'] and frame.y_true.tolist() == labels, 'Prediction assignment/order')
        require(len(frame) == 323 and frame.id.is_unique and set(frame.y_pred) <= {0, 1, 2}, 'Prediction scope')
        for key, value in [('macro_f1', f1_score(frame.y_true, frame.y_pred, average='macro')), ('accuracy', accuracy_score(frame.y_true, frame.y_pred))]:
            require(np.isclose(row[key], value, rtol=0, atol=1e-12), 'Recomputed ' + key)

    downstream = RESULTS / 'anova_revalidation'
    require(len(list((downstream / 'metrics').glob('*_test_metrics_v2.json'))) == 24, 'Main metric files')
    require(len(list((downstream / 'predictions').glob('*_test_predictions_v2.csv'))) == 24, 'Main prediction files')
    for model, condition, seed in itertools.product(['LR', 'RF', 'XGB', 'CNN1D'], ['BSL', 'ENR'], [42, 123, 2024]):
        name = f'{model}_{condition}_seed{seed}'
        row = read(downstream / f'metrics/{name}_test_metrics_v2.json')
        frame = pd.read_csv(downstream / f'predictions/{name}_test_predictions_v2.csv')
        require((row['model'], row['condition'], row['seed']) == (model, condition, seed), 'Run identity')
        require(set(frame.model) == {model} and set(frame.condition) == {condition} and set(frame.seed) == {seed}, 'Prediction run identity')
        predictions(frame, row)
        require(row['dataset_sha256'] == GOLD_SHA and row['lexicon_manifest_sha256'] == LEXICON_SHA, 'Main input hashes')
        require(row['confusion_matrix'] == confusion_matrix(frame.y_true, frame.y_pred, labels=[0, 1, 2]).tolist(), 'Confusion matrix')
        if condition == 'ENR':
            require(row['ont_features'] == SELECTED, 'Main variable order')
            for part, profile in row['semantic_input_provenance'].items():
                require(profile == profiles[part][profile['profile']], 'Runtime profile')
                fingerprints += 1
        rows.append({k: row[k] for k in ['model', 'condition', 'seed', 'macro_f1', 'accuracy']})
    require(fingerprints == 27, 'Runtime profile count')
    from run_module_ablation import build_conditions, condition_features
    conditions = build_conditions(False)
    for family, prefix, models, n in [('classical', 'module_ablation', ['LR', 'RF', 'XGB'], 126), ('cnn1d', 'cnn1d_module_ablation', ['CNN1D'], 42)]:
        folder = RESULTS / 'ablation' / family
        raw = pd.read_csv(folder / f'{prefix}_raw.csv')
        pred = pd.read_csv(folder / f'{prefix}_predictions.csv')
        expected = set(itertools.product(models, [42, 123, 2024], conditions))
        require(len(raw) == n and set(zip(raw.model, raw.seed, raw.condition)) == expected, family + ' run grid')
        require(len(pred) == n * 323, family + ' predictions')
        for row in raw.to_dict('records'):
            frame = pred[(pred.model == row['model']) & (pred.condition == row['condition']) & (pred.seed == row['seed'])]
            predictions(frame, row)
            expected_features = condition_features(row['condition'], ontology, SELECTED)
            reported = [] if pd.isna(row['ontology_features']) else row['ontology_features'].split('|')
            require(reported == expected_features and row['n_ontology_features'] == len(expected_features), family + ' variable order')
    sens = pd.read_csv(RESULTS / 'sensitivity/vader_ke_sensitivity_summary.csv')
    require(len(sens) == 125 and sens.variant_status.value_counts().to_dict() == {'passed': 97, 'invalid': 28}, 'Sensitivity outcomes')
    require(set(zip(sens.context_window, sens.minimum_frequency, sens.minimum_absolute_valence)) == set(itertools.product([2, 3, 5, 7, 10], [1, 2, 3, 5, 8], [.25, .5, 1., 1.5, 2.])), 'Sensitivity parameter grid')
    for row in sens.to_dict('records'):
        folder = ROOT / row['variant_dir']
        require((folder / 'lexicon_manifest.json').is_file(), 'Sensitivity lexicon')
        if row['variant_status'] == 'passed':
            data = pd.read_parquet(folder / 'gold_enriched_ontology.parquet')
            require(data.shape == (1614, 47) and data.id.tolist() == original.id.tolist(), 'Sensitivity Gold identity')
            require(row['shacl_conforms'] and row['shacl_validation_result_count'] == 0, 'Sensitivity conformance')
            require((folder / 'rdf/posts.ttl').is_file() and (folder / 'traceability_map.csv').is_file(), 'Sensitivity artifacts')
        else:
            require(not (folder / 'rdf/posts.ttl').exists(), 'Rejected setting has no evaluated RDF')
    for item in read(RESULTS / 'provenance/public_projection.json')['files']:
        require(sha(ROOT / item['public_path']) == item['public_sha256'], 'Public projection hash: ' + item['public_path'])
    figures = read(RESULTS / 'provenance/figures/public_dependency_map.json')
    require(len(figures['outputs']) == 8, 'Figure output coverage')
    for item in figures['inputs']:
        require(sha(ROOT / item['public_path']) == item['public_sha256'], 'Figure public dependency')
    for item in figures['outputs'] + [figures['figure4_editable_source']]:
        require(sha(ROOT / item['path']) == item['sha256'], 'Figure output/source integrity')
    e1 = ROOT / 'experiments/controlled_discontinuities/evidence'
    cases = read(e1 / 'continuity/E1_RESULTS.json')
    require(len(cases) == len({row['case_id'] for row in cases}) == 26, 'Continuity case coverage')
    require(sum(row['oracle_should_alarm'] for row in cases) == 22, 'Continuity fault denominator')
    require(sum(row['original_detected'] for row in cases) == 15, 'Original continuity detections')
    require(sum(row['extension_detected'] and not row['original_detected'] for row in cases) == 7, 'Supplementary-only detections')
    require(sum(row['detection_status'] == 'control_passed' for row in cases) == 4, 'Continuity controls')
    require(not any(row['false_positive'] or row['non_detection'] for row in cases), 'Continuity classification')
    shacl = read(e1 / 'shacl/summary_reclassified.json')
    require(shacl['counts'] == {'total_cases': 11, 'controls': 1, 'faults': 10, 'detected_by_original': 10, 'control_passed': 1, 'partial_detection': 0, 'not_detected': 0, 'false_positive': 0}, 'Independent SHACL cases')
    summary = pd.DataFrame(rows).groupby(['model', 'condition']).agg(macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'), accuracy_mean=('accuracy', 'mean'), seeds=('seed', 'count')).reset_index()
    pd.testing.assert_frame_equal(summary, pd.read_csv(RESULTS / 'provenance/campaign/main_summary.csv'), check_exact=False, rtol=0, atol=1e-12)
    return {'status': 'passed', 'release': 'v1.1.0', 'main_runs': 24, 'classical_ablation_runs': 126, 'cnn_ablation_runs': 42, 'predictions_recomputed': 62016, 'runtime_profiles_matched': fingerprints, 'sensitivity_admissible': 97, 'sensitivity_rejected': 28, 'training_performed': False}

if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
