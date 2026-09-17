"""Final E2 label-based replay; no private source dependencies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONCEPTS = ['InteligenciaArtificial', 'AprendizajeAutomatico', 'Tecnologia',
            'Futuro', 'Datos', 'Algoritmo', 'Robot', 'Automatizacion', 'Etica', 'Innovacion']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def output_dir(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gold', type=Path, default=HERE.parents[1] / 'data/gold/gold_enriched_ontology.parquet')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    # Require a new directory outside the checkout: never replace frozen evidence.
    repo = HERE.parents[1]
    if out == repo or repo in out.parents or out.exists():
        parser.error('--output-dir must be a NEW directory outside this checkout')
    out.mkdir(parents=True, exist_ok=False)
    args.output_dir = out
    return args


def verify_package():
    for line in (HERE / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(HERE / name) == digest, name
    freeze = read(HERE / 'reference_freeze.json')
    assert sha(HERE / 'human_reference.json') == freeze['public_reference_sha256']
    for n in 'AB':
        annotation = read(HERE / f'annotations_{n}.json')
        assert annotation['sample_sha256'] == sha(HERE / 'items_blinded.json')
        assert annotation['guide_sha256'] == sha(HERE / 'GUIA_E2_FINAL_V22.md')
        assert annotation['complete'] and annotation['validation_issues'] == []


def valid_label(row):
    pr, po = row['presence'], row['polarity']
    assert ((pr == 'absent' and po == 'not_applicable') or
            (pr == 'uncertain' and po == 'insufficient') or
            (pr == 'present' and po in ['positive', 'negative', 'neutral', 'mixed', 'insufficient']))


def reference():
    a, b = [read(HERE / f'annotations_{n}.json')['answers'] for n in 'AB']
    items = read(HERE / 'items_blinded.json')
    mapping = {r['item_id']: r for r in read(HERE / 'sampling_manifest.json')['records']}
    rows = read(HERE / 'adjudications.json')
    adj = {(r['item_id'], r['concept']): r for r in rows}
    assert len(adj) == len(rows) == 77
    assert len(items) == len(mapping) == 120 and set(a) == set(b) == set(mapping)
    assert {r['split'] for r in mapping.values()} == {'val'}
    result = []
    used = set()
    for item in items:
        i = item['item_id']
        assert set(a[i]['concepts']) == set(b[i]['concepts']) == set(CONCEPTS)
        for c in CONCEPTS:
            x, y = a[i]['concepts'][c], b[i]['concepts'][c]
            valid_label(x)
            valid_label(y)
            r = adj.get((i, c))
            if r:
                assert r['review_status'].startswith('closed')
                assert r['A'] == x and r['B'] == y
                pr, po = r['adjudicated_presence'], r['adjudicated_polarity']
                origin = 'author_assisted_adjudication'
                used.add((i, c))
            else:
                assert x == y
                pr, po = x['presence'], x['polarity']
                origin = 'independent_AB_agreement'
            row = dict(item_id=i, previous_item_id=mapping[i]['previous_item_id'],
                       post_id=mapping[i]['post_id'], concept=c, presence=pr, polarity=po,
                       origin=origin, case_id=r['case_id'] if r else None,
                       rationale_origin=r.get('rationale_origin') if r else None)
            valid_label(row)
            result.append(row)
    assert used == set(adj)
    assert len({(r['post_id'], r['concept']) for r in result}) == len(result) == 1200
    assert result == read(HERE / 'human_reference.json')
    return result


def agreement(pairs):
    a, b = zip(*pairs) if pairs else ([], [])
    ca, cb = Counter(a), Counter(b)
    n = len(a)
    same = sum(x == y for x, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2 if n else None
    k = None if not n or pe == 1 else (same / n - pe) / (1 - pe)
    return dict(n=n, agreements=same, disagreements=n-same,
                agreement=same/n if n else None, kappa=k, marginal_A=dict(ca), marginal_B=dict(cb),
                confusion=[dict(A=x, B=y, count=v) for (x, y), v in sorted(Counter(pairs).items())])


def analyze():
    a, b = [read(HERE / f'annotations_{n}.json')['answers'] for n in 'AB']
    items = read(HERE / 'items_blinded.json')
    prs, pol, families = [], [], {}
    for c in CONCEPTS:
        cp, cs = [], []
        for item in items:
            x, y = [d[item['item_id']]['concepts'][c] for d in (a, b)]
            pair = (x['presence'], y['presence'])
            prs.append(pair)
            cp.append(pair)
            if pair == ('present', 'present'):
                pair = (x['polarity'], y['polarity'])
                pol.append(pair)
                cs.append(pair)
        families[c] = dict(presence=agreement(cp), polarity_both_present=agreement(cs))
    result = dict(presence=agreement(prs), polarity_both_present=agreement(pol), by_concept=families)
    assert result == read(HERE / 'agreement_summary.json')
    return result


def metrics(rows):
    valid = [r for r in rows if r['human_presence'] != 'uncertain']
    tp = sum(r['human_presence'] == 'present' and r['automatic_presence'] for r in valid)
    fp = sum(r['human_presence'] == 'absent' and r['automatic_presence'] for r in valid)
    fn = sum(r['human_presence'] == 'present' and not r['automatic_presence'] for r in valid)
    tn = sum(r['human_presence'] == 'absent' and not r['automatic_presence'] for r in valid)
    pol = [r for r in rows if r['polarity_eligible']]
    exact = sum(r['exact_set_match'] for r in pol)
    detected = [r for r in pol if r['automatic_presence']]
    return dict(total=len(rows), human_counts=dict(Counter(r['human_presence'] for r in rows)),
                presence=dict(n=len(valid), excluded_uncertain=len(rows)-len(valid), TP=tp, FP=fp, FN=fn, TN=tn,
                              precision=tp/(tp+fp) if tp+fp else None,
                              recall=tp/(tp+fn) if tp+fn else None,
                              F1=2*tp/(2*tp+fp+fn) if tp+fn else None),
                polarity=dict(n=len(pol), excluded_absent=sum(r['human_presence'] == 'absent' for r in rows),
                              excluded_insufficient=sum(r['human_polarity'] == 'insufficient' for r in rows),
                              exact_matches=exact, exact_match_rate=exact/len(pol) if pol else None,
                              cases_with_extra_labels=sum(bool(r['extra_labels']) for r in pol),
                              cases_with_missing_labels=sum(bool(r['missing_labels']) for r in pol),
                              extra_label_count=sum(len(r['extra_labels']) for r in pol),
                              missing_label_count=sum(len(r['missing_labels']) for r in pol),
                              human_mixed=sum(r['human_polarity'] == 'mixed' for r in pol),
                              automatic_multilabel=sum(len(r['automatic_polarities']) > 1 for r in pol),
                              supplementary_detected_only_n=len(detected),
                              supplementary_detected_only_exact=sum(r['exact_set_match'] for r in detected)),
                error_types=dict(Counter(r['error_type'] for r in rows)))


def evaluate(human, goldpath):
    import pandas as pd
    expected = read(HERE / 'provenance.json')['gold_sha256']
    assert sha(goldpath) == expected, 'Gold bytes differ from the canonical F123 input'
    frame = pd.read_parquet(goldpath)
    assert frame.shape == (1614, 47)
    gold = frame.set_index('id')
    assert gold.index.is_unique
    items = {r['item_id']: r for r in read(HERE / 'items_blinded.json')}
    polmap = {'positive': 'Positivo', 'negative': 'Negativo', 'neutral': 'Neutro'}
    sets = {'positive': {'positive'}, 'negative': {'negative'}, 'neutral': {'neutral'}, 'mixed': {'positive', 'negative'}}
    comparisons = []
    for h in human:
        i, c = h['item_id'], h['concept']
        g = gold.loc[h['post_id']]
        assert g['split'] == 'val'
        assert g['titulo'] == items[i]['title'] and g['texto'] == items[i]['text']
        vals = {k: float(g[f'ont_{c}_{v}']) for k, v in polmap.items()}
        assert all(pd.notna(v) and v >= 0 for v in vals.values())
        auto = {k for k, v in vals.items() if v > 0}
        hs = sets.get(h['polarity']) if h['presence'] == 'present' else None
        typ = ('insufficient_context' if h['presence'] == 'uncertain' else
               'false_activation' if h['presence'] == 'absent' and auto else
               'correct_absence' if h['presence'] == 'absent' else
               'missed_concept' if not auto else 'polarity_insufficient' if hs is None else
               'exact_concept_polarity' if auto == hs else 'local_polarity_set_mismatch')
        comparisons.append(dict(item_id=i, post_id=h['post_id'], concept=c,
                                human_presence=h['presence'], human_polarity=h['polarity'],
                                automatic_values=vals, automatic_presence=bool(auto), automatic_polarities=sorted(auto),
                                human_polarities=sorted(hs) if hs is not None else None,
                                polarity_eligible=hs is not None, exact_set_match=(auto == hs) if hs is not None else None,
                                extra_labels=sorted(auto-hs) if hs is not None else None,
                                missing_labels=sorted(hs-auto) if hs is not None else None, error_type=typ))
    summary = dict(overall=metrics(comparisons),
                   by_concept={c: metrics([r for r in comparisons if r['concept'] == c]) for c in CONCEPTS},
                   human_reference_origin=dict(Counter(r['origin'] for r in human)),
                   protocol='E2-final-v22-reannotation-v1; unchanged F123 and set-matching protocol; polarity denominator is all human-present with resolvable polarity, including automatic misses')
    assert comparisons == read(HERE / 'comparisons.json')
    assert summary == read(HERE / 'summary.json')
    assert sha(goldpath) == expected
    return comparisons, summary
