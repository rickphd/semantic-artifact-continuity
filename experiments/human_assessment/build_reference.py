"""Rebuild final labels without reading automatic pipeline values."""
from e2 import output_dir, verify_package, reference, write

if __name__ == '__main__':
    args = output_dir(__doc__)
    verify_package()
    write(args.output_dir / 'human_reference.json', reference())
