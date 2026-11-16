"""Create an isolated, pinned Radiance checkout and verify its patch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from patch_series import verified_patches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    meta, specs, patches = verified_patches(root)
    dest = a.destination.resolve()
    if dest.exists():
        p.error('destination already exists; choose a new directory to preserve local work')
    subprocess.run(['git', 'init', str(dest)], check=True)
    def git(*args):
        subprocess.run(['git', '-C', str(dest), *args], check=True)
    git('remote', 'add', 'origin', meta['repository'])
    git('fetch', '--depth=1', 'origin', meta['revision'])
    git('checkout', '--detach', 'FETCH_HEAD')
    for patch in patches:
        git('apply', '--check', str(patch))
        git('apply', str(patch))
    paths = subprocess.check_output(['git', '-C', str(dest), 'ls-files', '--cached', '--others', '--exclude-standard', '-z']).decode().split('\0')
    manifest = {'revision': meta['revision'], 'patches': [s['sha256'] for s in specs],
                'files': {name: hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in paths if name}}
    (dest/'.radiance-source-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"Prepared ABI{meta['abi']} source: {dest}")


if __name__ == '__main__':
    main()
