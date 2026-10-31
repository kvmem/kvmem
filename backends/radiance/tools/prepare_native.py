"""Create an isolated, pinned ABI15 Radiance checkout and verify its patch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    meta = json.loads((root / 'upstream.json').read_text(encoding='utf-8'))
    patch = (root / meta['native_patch']).resolve()
    if hashlib.sha256(patch.read_bytes()).hexdigest() != meta['native_patch_sha256']:
        p.error('native patch checksum differs from upstream.json')
    dest = a.destination.resolve()
    if dest.exists():
        p.error('destination already exists; choose a new directory to preserve local work')
    subprocess.run(['git', 'init', str(dest)], check=True)
    def git(*args):
        subprocess.run(['git', '-C', str(dest), *args], check=True)
    git('remote', 'add', 'origin', meta['repository'])
    git('fetch', '--depth=1', 'origin', meta['revision'])
    git('checkout', '--detach', 'FETCH_HEAD')
    git('apply', '--check', str(patch))
    git('apply', str(patch))
    print(f'Prepared ABI15 source: {dest}')


if __name__ == '__main__':
    main()
