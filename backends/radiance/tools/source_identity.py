"""Validate either the pinned Git tree or a checksummed offline source archive."""
import hashlib
import json
import subprocess
from pathlib import Path
from patch_series import patch_specs


def source_revision(source):
    meta = json.loads((Path(__file__).resolve().parents[1]/'upstream.json').read_text(encoding='utf-8'))
    manifest = json.loads((source/'.radiance-source-manifest.json').read_text(encoding='utf-8'))
    expected = [spec['sha256'] for spec in patch_specs(meta)]
    if manifest.get('patches', [manifest.get('patch_sha256')]) != expected or manifest['revision'] != meta['revision']:
        raise ValueError('native source patch chain/revision differs from the adapter pin')
    if (source/'.git').exists():
        head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if head != manifest['revision']:
            raise ValueError('native Git HEAD differs from the source manifest')
    actual = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() and
              '.git' not in p.relative_to(source).parts and p.name != '.radiance-source-manifest.json'}
    if actual != set(manifest['files']):
        raise ValueError('native source file inventory differs from the source manifest')
    for name, expected in manifest['files'].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'offline source checksum mismatch: {name}')
    return manifest['revision']
