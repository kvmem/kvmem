#!/usr/bin/env python3
"""Check a structured launch against its actual model-free worker contract."""
import argparse
import json
from pathlib import Path
import sys

from launcher.worker import probe_worker, require_launch_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--backend', required=True)
    args = parser.parse_args()
    try:
        data = json.load(sys.stdin)
        if not isinstance(data, dict) or data.get('Backend') != args.backend:
            raise ValueError('Worker check requires its matching structured backend parameters')
        capabilities = probe_worker(args.worker, args.backend)
        require_launch_features(capabilities, data)
        print(json.dumps(capabilities, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
