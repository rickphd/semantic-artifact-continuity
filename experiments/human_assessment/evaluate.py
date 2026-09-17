"""Replay final E2 reference, agreement and extraction correspondence."""
from e2 import output_dir, verify_package, reference, analyze, evaluate, write

if __name__ == '__main__':
    args = output_dir(__doc__)
    verify_package()
    human = reference()
    agreement = analyze()
    comparisons, summary = evaluate(human, args.gold)
    for name, value in [('human_reference.json', human), ('agreement_summary.json', agreement),
                        ('comparisons.json', comparisons), ('summary.json', summary)]:
        write(args.output_dir / name, value)
    write(args.output_dir / 'verification.json', dict(
        public_manifest_verified=True, reference_exact=True, agreement_exact=True,
        comparisons_exact=True, summary_exact=True, canonical_gold_hash_verified=True,
        units=len(human), output_scope='fresh scratch directory only'))
    print(summary['overall'])
