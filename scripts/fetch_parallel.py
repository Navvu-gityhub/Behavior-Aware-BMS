"""Download one large file over many parallel HTTP range requests, resumably.

    python scripts/fetch_parallel.py URL OUTPUT [--parts 24]

For servers that throttle each connection (the Severson LFP batches on
data.matr.io deliver ~40 KB/s per connection). Each part is written to
OUTPUT.partNN and skipped on rerun if complete; parts are joined at the end.
"""

from __future__ import annotations

import argparse
import http.client
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def size_of(url: str) -> tuple[str, int]:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers["Content-Range"].split("/")[-1])
        return r.geturl(), total


def fetch_part(url: str, path: Path, start: int, end: int) -> str:
    want = end - start + 1
    have = path.stat().st_size if path.exists() else 0
    if have >= want:
        return f"{path.name} done"
    for _attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "ab") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    have += len(chunk)
            if have >= want:
                return f"{path.name} ok"
        except (OSError, http.client.HTTPException):
            have = path.stat().st_size if path.exists() else 0
    raise RuntimeError(f"{path.name}: gave up at {have}/{want} bytes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("output", type=Path)
    ap.add_argument("--parts", type=int, default=24)
    a = ap.parse_args()
    url, total = size_of(a.url)
    step = -(-total // a.parts)
    parts = [(a.output.with_suffix(a.output.suffix + f".part{i:02d}"), i * step, min((i + 1) * step, total) - 1)
             for i in range(a.parts) if i * step < total]
    with ThreadPoolExecutor(max_workers=len(parts)) as pool:
        for msg in pool.map(lambda p: fetch_part(url, *p), parts):
            print(msg, flush=True)
    with open(a.output, "wb") as out:
        for p, _, _ in parts:
            out.write(p.read_bytes())
    if a.output.stat().st_size != total:
        raise RuntimeError(f"joined size {a.output.stat().st_size} != {total}")
    for p, _, _ in parts:
        p.unlink()
    print(f"{a.output} {total} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
