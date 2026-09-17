"""Recompute pre-adjudication agreement, including per-concept results."""
from e2 import output_dir, verify_package, analyze, write

if __name__ == '__main__':
    args = output_dir(__doc__)
    verify_package()
    write(args.output_dir / 'agreement_summary.json', analyze())
