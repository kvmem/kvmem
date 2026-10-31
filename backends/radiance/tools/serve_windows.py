"""Compatibility entry point for the supervised Windows launcher."""
import sys
from serve import main

if __name__ == '__main__':
    raise SystemExit(main([*sys.argv[1:], '--platform', 'windows']))
