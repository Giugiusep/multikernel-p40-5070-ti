#!/usr/bin/env python3
"""Verify the two local IQ2_XS GGUF shards against their Hub LFS metadata."""

import hashlib
import json
from pathlib import Path


ROOT = Path('/home/shiba')
metadata = json.loads((ROOT / 'multikernel-experiment/research-qwen-files.json').read_text())
model_dir = ROOT / 'models/qwen3.8-flash-next-gsq-rco'
shards = sorted(
    (item for item in metadata if item['path'].startswith('IQ2_XS/') and item['path'].endswith('.gguf')),
    key=lambda item: item['path'],
)
assert len(shards) == 2, f'expected two IQ2_XS shards, found {len(shards)}'

for shard in shards:
    path = model_dir / shard['path']
    assert path.is_file(), f'missing {path}'
    expected_size = shard['lfs']['size']
    assert path.stat().st_size == expected_size, f'wrong size: {path}'
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 << 20), b''):
            digest.update(chunk)
    expected_hash = shard['lfs']['oid']
    observed_hash = digest.hexdigest()
    assert observed_hash == expected_hash, f'wrong SHA-256: {path}'
    print(f'{path.name}: {expected_size} bytes, SHA-256 {observed_hash}, OK')
