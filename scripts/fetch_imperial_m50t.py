"""Fetch a fixed subset of the Imperial LG M50T cycle-ageing data (Expt 5).

    python scripts/fetch_imperial_m50t.py

Source: Kirkaldy et al., "Lithium-ion battery degradation: comprehensive cycle
ageing data and analysis for commercial 21700 cells", J. Power Sources (2024),
doi:10.1016/j.jpowsour.2024.234185; data doi:10.5281/zenodo.10637534 (CC-BY-4.0).

Expt 5: LG M50T (NMC811 / graphite-SiOx, 5 Ah, 21700), 0-100% SoC, 0.3C charge
and 1C discharge, at 10, 25 and 40 C - every ageing cycle is a full discharge
at the test temperature. The archive is 10 GB; the raw cycling is Biologic
.mpr. This reads only the members it needs through HTTP range requests.

SUBSET, FIXED BEFORE ANY DATA WAS READ: for each of the 8 cells, ageing sets
1, 3, 5, ..., 15; within a set, the file without "part" in its name (the
first segment), else the largest. Plus the per-cycle summary CSVs.
Written to data/raw/imperial_m50t/ (gitignored).
"""

from __future__ import annotations

import io
import re
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = ("https://zenodo.org/api/records/10637534/files/"
       "Expt%205%20-%20Standard%20Cycle%20Aging%20(Control).zip/content")
OUT = Path("data/raw/imperial_m50t")
SETS = set(range(1, 16, 2))


class _RangeFile(io.RawIOBase):
    """A seekable view of a remote file, read with HTTP Range requests."""

    def __init__(self, url: str):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.geturl()
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        rng = f"bytes={self.pos}-{self.pos + n - 1}"
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": rng})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 7:
                    raise
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _zip() -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(_RangeFile(URL), buffer_size=8 << 20))


def select(names: list[tuple[str, int]]) -> list[str]:
    chosen: dict[tuple[str, int], tuple[str, int]] = {}
    for name, size in names:
        m = re.search(r"Degradation Cycling/Set (\d+)/.*?(\d+)degC - cell (\w)", name)
        if not m or not name.endswith(".mpr") or int(m.group(1)) not in SETS:
            continue
        key = (m.group(3), int(m.group(1)))
        is_main = "part" not in name.rsplit("/", 1)[1].lower()
        best = chosen.get(key)
        rank = (is_main, size)
        if best is None or rank > (("part" not in best[0].rsplit("/", 1)[1].lower()), best[1]):
            chosen[key] = (name, size)
    return sorted(v[0] for v in chosen.values())


def fetch(name: str) -> str:
    target = OUT / "cycling" / name.rsplit("/", 1)[1]
    if target.exists():
        return f"have {target.name}"
    z = _zip()
    tmp = target.with_suffix(".part")
    with z.open(name) as src, open(tmp, "wb") as dst:
        while chunk := src.read(8 << 20):
            dst.write(chunk)
    tmp.rename(target)
    return f"got {target.name}"


def main() -> int:
    (OUT / "cycling").mkdir(parents=True, exist_ok=True)
    z = _zip()
    infos = [(i.filename, i.file_size) for i in z.infolist()]
    for name, _ in infos:
        if "Summary per Cycle" in name and name.endswith(".csv"):
            (OUT / name.rsplit("/", 1)[1]).write_bytes(z.read(name))
    picks = select(infos)
    print(f"{len(picks)} files, {sum(s for n, s in infos if n in set(picks)) / 1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for msg in pool.map(fetch, picks):
            print(msg, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
