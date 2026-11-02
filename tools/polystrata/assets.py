"""Prepare this GGUF's tokenizer in a model-specific local cache, without downloads."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

FILES = ('vocab.json', 'merges.txt', 'tokenizer.json', 'token_type.json', 'chat_template.jinja')

def tokenizer_files(target: Path) -> tuple[str, ...]:
    # Gemma's UTF-8 pieces can contain newlines; its exporter writes JSON.
    merge='merges.json' if (target/'merges.json').is_file() else 'merges.txt'
    return tuple(merge if name=='merges.txt' else name for name in FILES)


def model_identity(model: Path, source: Path) -> dict:
    stat = model.stat()
    with model.open('rb') as stream:
        header = hashlib.sha256(stream.read(1024 * 1024)).hexdigest()
    return {'path': str(model.resolve()), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'header_sha256': header,
            'exporter_sha256': hashlib.sha256(b''.join((source / 'tools' / name).read_bytes()
                for name in ('strata_tokenizer.py', 'gguf_reader.py'))).hexdigest()}


def tokenizer_path(model: Path, source: Path, *, prepare: bool, cache: Path | None = None) -> Path:
    identity = model_identity(model, source)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if cache is None:
        cache = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.cache'))) / 'KVMem' / 'polystrata-tokenizers'
    root = cache / key
    target = root / 'tokenizer'
    manifest = root / 'source.json'
    files=tokenizer_files(target)
    if manifest.is_file() and all((target / name).is_file() for name in files):
        record = json.loads(manifest.read_text(encoding='utf-8'))
        if record.get('model') == identity and set(record.get('files',{}))==set(files) and all(
                hashlib.sha256((target / name).read_bytes()).hexdigest() == record.get('files', {}).get(name)
                for name in files):
            return target
        raise ValueError(f'Cached tokenizer was modified: {root}; select an explicit tokenizer or remove that cache entry')
    if not prepare:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=key[:12] + '-', dir=cache) as directory:
        staging = Path(directory)
        sys.path.insert(0, str(source / 'tools'))
        try:
            from strata_tokenizer import extract
            extract(model, staging)
        finally:
            sys.path.pop(0)
        files=tokenizer_files(staging/'tokenizer')
        if not all((staging / 'tokenizer' / name).is_file() for name in files):
            raise ValueError('The GGUF must contain its own complete tokenizer and chat template')
        record = {'model': identity, 'files': {name: hashlib.sha256((staging / 'tokenizer' / name).read_bytes()).hexdigest() for name in files}}
        (staging / 'source.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        # Publish only a complete export; concurrent identical launches can share the winner.
        try:
            staging.rename(root)
        except FileExistsError:
            if not manifest.is_file() or json.loads(manifest.read_text(encoding='utf-8')) != record:
                raise ValueError(f'Incomplete or conflicting tokenizer cache: {root}') from None
    return target
