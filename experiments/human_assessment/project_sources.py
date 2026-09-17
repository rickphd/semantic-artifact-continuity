"""Maintainer-only deterministic projection from authorized final private sources.

Public replay does not require this script or access to private source files.
Run with --workspace-root; writes only beside this script, refusing existing data.
"""
import argparse
from collections import Counter
from pathlib import Path
import re
from e2 import HERE, read, write, sha


def project(d, keys):
    return {k: d[k] for k in keys if k in d}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    root = parser.parse_args().workspace_root.resolve()
    rev = root / 'JOWS/revisions'
    sample = rev / 'e2_final_v22_20260916'
    review = rev / 'e2_final_v22_review_20260916'
    adj = rev / 'e2_final_v22_adjudication_20260916'
    evaluation = rev / 'e2_final_v22_evaluation_20260916'
    entries = []
    staged = {}

    def add(source, public, value=None, removed=(), transform='structured_allowlist_v1'):
        assert public not in staged
        data = source.read_bytes() if value is None else None
        staged[public] = (data, value)
        entries.append(dict(source_path=source.relative_to(root).as_posix(), source_sha256=sha(source),
                            public_path=public, transformation='exact_copy' if value is None else transform,
                            removed_fields=list(removed), equivalence='exact_bytes' if value is None else 'retained_source_fields_equal'))

    for n in 'AB':
        path = review / f'originals/E2_final_v22_{n}.json'
        original = read(path)
        data = project(original, ['protocol', 'sample_sha256', 'guide_sha256', 'annotator', 'complete', 'validation_issues'])
        assert data['annotator'] == n
        data['answers'] = {i: dict(language=a['language'], concepts={c: project(v, ['presence', 'polarity'])
                           for c, v in a['concepts'].items()}) for i, a in original['answers'].items()}
        add(path, f'annotations_{n}.json', data,
            ['exported_at', 'answers.*.notes', 'answers.*.concepts.*.evidence', 'answers.*.concepts.*.justification'])
    rows = []
    for r in read(adj / 'adjudications.json'):
        p = project(r, ['case_id', 'item_id', 'concept', 'dimension', 'status', 'adjudicated_presence',
                        'adjudicated_polarity', 'review_status', 'rationale_origin', 'adjudication_mode'])
        p['adjudicator_role'] = 'author; AI-assisted adjudication, not independent annotation'
        for n in 'AB':
            p[n] = project(r[n], ['presence', 'polarity'])
        if 'assistant_recommendation_rejected' in r:
            p['assistant_recommendation_rejected'] = project(r['assistant_recommendation_rejected'], ['presence', 'polarity'])
        rows.append(p)
    assert len(rows) == 77
    add(adj / 'adjudications.json', 'adjudications.json', rows,
        ['user_statement', 'history', 'adjudicator', 'date', 'coordinator_note', 'human_rationale',
         'A.evidence', 'A.justification', 'B.evidence', 'B.justification', 'assistant_recommendation_rejected.rationale'],
        'structured_allowlist_and_anonymous_author_role_v1')
    add(sample / 'items_blinded.json', 'items_blinded.json')
    add(sample / 'GUIA_E2_FINAL_V22.md', 'GUIA_E2_FINAL_V22.md')
    sm = read(sample / 'coordinator_only/sampling_manifest.json')
    public_sm = project(sm, ['protocol', 'seed', 'selection', 'records', 'sample_sha256', 'guide_sha256',
                             'same_participants', 'recall_possible', 'independent_new_sample'])
    add(sample / 'coordinator_only/sampling_manifest.json', 'sampling_manifest.json', public_sm,
        sorted(set(sm) - set(public_sm)))
    human = [project(r, ['item_id', 'previous_item_id', 'post_id', 'concept', 'presence', 'polarity',
                        'origin', 'case_id', 'rationale_origin']) for r in read(evaluation / 'human_reference.json')]
    add(evaluation / 'human_reference.json', 'human_reference.json', human,
        ['human_rationale', 'evidence_A', 'evidence_B', 'justification_A', 'justification_B'])
    for name in ['comparisons.json', 'summary.json']:
        add(evaluation / name, name)
    summary = read(review / 'summary.json')
    keep = ['presence', 'polarity_both_present', 'by_concept']
    add(review / 'summary.json', 'agreement_summary.json', project(summary, keep), sorted(set(summary) - set(keep)))
    verify = read(review / 'verification.json')
    keep = ['expected_units_per_annotator', 'disagreement_cases', 'shared_uncertainty_cases', 'distinct_review_units']
    add(review / 'verification.json', 'agreement_verification.json', project(verify, keep), sorted(set(verify) - set(keep)))
    ev = read(evaluation / 'verification.json')
    keep = ['reference_frozen_before_pipeline_read', 'gold_sha256', 'post_ids_texts_val_verified', 'units', 'adjudications', 'sklearn_metrics_crosscheck']
    add(evaluation / 'verification.json', 'historical_evaluation_verification.json', project(ev, keep), sorted(set(ev) - set(keep)))
    freeze = read(evaluation / 'reference_freeze.json')
    historical_sources = []
    for path, digest in freeze['source_hashes'].items():
        relative = 'JOWS/revisions/' + path.split('/revisions/', 1)[1]
        assert sha(root / relative) == digest
        target = next(e['public_path'] for e in entries if e['source_path'] == relative)
        historical_sources.append(dict(source_path=relative, source_sha256=digest, public_path=target))
    assert len(historical_sources) == 6
    public_freeze = project(freeze, ['counts', 'polarities', 'origins', 'constructed_without_reading_pipeline_values'])
    public_freeze['historical_reference_sha256'] = freeze['reference_sha256']
    public_freeze['historical_sources'] = historical_sources
    add(evaluation / 'reference_freeze.json', 'reference_freeze.json', public_freeze,
        ['source_hashes absolute keys'], 'relative_source_mapping_and_distinct_historical_hash_v1')
    # Reject contact/local-account strings in retained annotation metadata. Source
    # posts are published source text, not anonymous participant correspondence.
    private_values = {str(r.get('adjudicator', '')).strip() for r in read(adj / 'adjudications.json')}
    private_values = {v for v in private_values if v.lower() not in ['user', 'author', 'human', 'usuario', 'autor']}
    patterns = [r'/Users/[^/\s]+', r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}']
    import json
    for name, (data, value) in staged.items():
        if name == 'items_blinded.json':
            continue
        text = data.decode('utf-8') if data is not None else json.dumps(value, ensure_ascii=False)
        assert not any(re.search(p, text) for p in patterns), 'Privacy pattern in ' + name
        assert not any(v and len(v) > 2 and v in text for v in private_values), 'Identity match in ' + name
    for name in staged:
        assert not (HERE / name).exists(), 'Refusing to overwrite ' + name
    for name, (data, value) in staged.items():
        if data is not None:
            (HERE / name).write_bytes(data)
        else:
            write(HERE / name, value)
    public_freeze['public_reference_sha256'] = sha(HERE / 'human_reference.json')
    for source in public_freeze['historical_sources']:
        source['public_sha256'] = sha(HERE / source['public_path'])
    write(HERE / 'reference_freeze.json', public_freeze)
    for e in entries:
        e['public_sha256'] = sha(HERE / e['public_path'])
    gold = rev / 'f123_20260914/package/data/gold/gold_enriched_ontology.parquet'
    assert sha(gold) == ev['gold_sha256']
    write(HERE / 'provenance.json', dict(
        schema='E2-public-projection-v1', path_bases=dict(source='original workspace root (not a public dependency)', public='experiments/human_assessment'),
        gold_source_path=gold.relative_to(root).as_posix(),
        gold_public_path='data/gold/gold_enriched_ontology.parquet', gold_sha256=sha(gold), artifacts=entries,
        source_scripts=[dict(source_path=p.relative_to(root).as_posix(), source_sha256=sha(p), public_paths=names,
                             public_sha256={name: sha(HERE / name) for name in names},
                             transformation='portable_final_only_label_replay_v1')
                        for p, names in [(evaluation / 'evaluate.py', ['evaluate.py', 'e2.py']),
                                         (evaluation / 'build_reference.py', ['build_reference.py', 'e2.py']),
                                         (review / 'analyze.py', ['analyze.py', 'e2.py'])]],
        privacy=dict(free_annotation_prose='omitted, not replaced with empty strings',
                     retained_roles='anonymous A/B; author with AI assistance for all adjudications',
                     source_text='unchanged public-source text and post IDs; not anonymized',
                     checks='allowlisted fields; retained metadata checked for local paths, contacts and source adjudicator identity')))
    print('Projected', len(entries), 'source artifacts; preserved', len(human), 'reference units')


if __name__ == '__main__':
    main()
