"""Validate either the pinned Git tree or a checksummed offline source archive."""
import hashlib
import json
import subprocess


def source_revision(source):
    if (source/'.git').exists():
        return subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = json.loads((source/'.radiance-source-manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'offline source checksum mismatch: {name}')
    return manifest['revision']
