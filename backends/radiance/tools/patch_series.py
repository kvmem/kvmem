"""The ordered, checksummed native patch series shared by build and packaging."""
import hashlib
import json
from pathlib import Path


def patch_specs(metadata):
    return [{'path': metadata['native_patch'], 'sha256': metadata['native_patch_sha256']},
            *metadata.get('additional_native_patches', [])]


def verified_patches(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    metadata = json.loads((root/'upstream.json').read_text(encoding='utf-8'))
    specs = patch_specs(metadata)
    paths = [(root/spec['path']).resolve() for spec in specs]
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate native patch in series')
    for spec, path in zip(specs, paths):
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('native patch checksum differs from upstream.json: '+spec['path'])
    return metadata, specs, paths
