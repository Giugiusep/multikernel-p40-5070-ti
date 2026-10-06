"""Reconstruct the pinned Git tree over the official Linux base archive."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import time

BASE = Path('/home/shiba/multikernel-experiment')
REVISION = '3bdd35b64413da0b4e089ce931bfc2e8b031cbf7'


def fetch(item: dict) -> str:
    name = item['path']
    path = BASE / 'source' / name
    if path.is_file():
        data = path.read_bytes()
        if hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == item['sha']:
            return name + ' (already verified)'
    # Tag URL may be served from cache; Git blob hashes still pin the content.
    url = f'https://raw.githubusercontent.com/multikernel/linux/v7.0-mk2/{name}'
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)
    digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if digest != item['sha']:
        raise ValueError(f'Pinned Git blob mismatch: {name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if item['mode'] == '120000':
        raise ValueError(f'Unexpected changed symlink: {name}')
    path.write_bytes(data)
    path.chmod(0o755 if item['mode'] == '100755' else 0o644)
    return name


if __name__ == '__main__':
    delta = json.loads((BASE / 'logs/source-delta.json').read_text())
    assert not delta['extra'], 'Unexpected base-only files'
    failed = []
    for item in delta['missing']:
        try:
            print('Verified', fetch(item), flush=True)
        except Exception as error:
            failed.append(item['path'])
            print('FAILED', item['path'], str(error), flush=True)
    if failed:
        raise RuntimeError(f'Files still unavailable: {failed}')
    checked = 0
    for entry in (BASE / 'logs/pinned-tree.bin').read_bytes().split(b'\0'):
        if not entry:
            continue
        meta, name = entry.split(b'\t', 1)
        mode, kind, digest = meta.decode().split()
        path = BASE / 'source' / os.fsdecode(name)
        data = os.fsencode(os.readlink(path)) if mode == '120000' else path.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert actual == digest, os.fsdecode(name)
        assert (mode == '120000') == path.is_symlink(), os.fsdecode(name)
        if mode != '120000':
            assert bool(path.stat().st_mode & 0o111) == (mode == '100755'), os.fsdecode(name)
        checked += 1
    print(f'PASS: {checked} source files match the pinned Git tree contents and modes.', flush=True)
