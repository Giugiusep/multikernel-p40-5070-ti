#!/usr/bin/env python3
"""Resumable parallel HTTP-range fetch of Strata's published MTP inventory.

Only use after Strata tools/mtp_fetch.py inventory has produced
mtp-inventory.json and its serial fetch has stopped. Each completed range is
checkpointed; exact Content-Range and tensor sizes are checked before hashing.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import threading
import time

import requests


ROOT = pathlib.Path(__file__).resolve().parents[1]
MTP = ROOT / "research/Strata/mtp"
CHUNK = 64 << 20
STATE = MTP / "parallel-range-state.json"
LOCAL = threading.local()


def session() -> requests.Session:
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers["User-Agent"] = "shiba-strata-mtp-range-fetch"
    return LOCAL.session


def fetch_range(job: tuple[dict, int, int, int]) -> tuple[str, int, int]:
    row, offset, length, fd = job
    first = row["start"] + offset
    last = first + length - 1
    url = json.loads((MTP / "mtp-inventory.json").read_text())["repo"] + row["shard"]
    headers = {"Range": f"bytes={first}-{last}"}
    for attempt in range(5):
        try:
            response = session().get(url, headers=headers, timeout=(30, 180))
            response.raise_for_status()
            content_range = response.headers.get("Content-Range", "")
            found = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if response.status_code != 206 or not found or (int(found[1]), int(found[2])) != (first, last):
                raise IOError(f"wrong range: status={response.status_code} header={content_range}")
            if len(response.content) != length:
                raise IOError(f"short range: {len(response.content)} != {length}")
            if os.pwrite(fd, response.content, offset) != length:
                raise IOError("short pwrite")
            return row["name"], offset, length
        except (requests.RequestException, OSError) as error:
            if attempt == 4:
                raise RuntimeError(f"range {row['name']} {first}-{last}: {error}") from error
            time.sleep(min(2 ** attempt, 16))
    raise AssertionError("unreachable")


def save_state(completed: set[str]) -> None:
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(sorted(completed)))
    temporary.replace(STATE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    inventory = json.loads((MTP / "mtp-inventory.json").read_text())
    resuming_parallel = STATE.exists()
    completed = set(json.loads(STATE.read_text())) if resuming_parallel else set()
    tensor_dir = MTP / "tensors"
    tensor_dir.mkdir(exist_ok=True)
    files = {}
    jobs = []
    for row in inventory["tensors"]:
        path = tensor_dir / (row["name"] + ".bin")
        old_size = path.stat().st_size if path.exists() else 0
        if old_size > row["bytes"]:
            raise ValueError(f"oversized tensor: {path}")
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        files[row["name"]] = fd
        os.ftruncate(fd, row["bytes"])
        for offset in range(0, row["bytes"], CHUNK):
            key = f"{row['name']}:{offset}"
            length = min(CHUNK, row["bytes"] - offset)
            # A complete file from Strata's original serial downloader is
            # already checked there. A shorter file has a contiguous prefix.
            if not resuming_parallel and key not in completed and offset + length <= old_size:
                completed.add(key)
            if key not in completed:
                jobs.append((row, offset, length, fd))
    save_state(completed)
    print(f"{len(jobs)} ranges to fetch, {sum(job[2] for job in jobs) / 1e9:.3f} GB", flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_range, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                name, offset, length = future.result()
                os.fsync(files[name])
                completed.add(f"{name}:{offset}")
                save_state(completed)
                print(f"{len(completed)} ranges complete: {name} {offset + length}", flush=True)
        manifest = []
        for row in inventory["tensors"]:
            path = tensor_dir / (row["name"] + ".bin")
            assert path.stat().st_size == row["bytes"]
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(16 << 20), b""):
                    digest.update(block)
            manifest.append(dict(row, file=str(path.relative_to(MTP)), sha256=digest.hexdigest()))
        (MTP / "mtp-manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"complete: {len(manifest)} tensors, {(MTP / 'mtp-manifest.json')}", flush=True)
    finally:
        for fd in files.values():
            os.close(fd)


if __name__ == "__main__":
    main()
