"""Fetch the per-RPT performance summaries of Imperial LG M50T experiments.

    python scripts/fetch_imperial_summaries.py 4 5      # development experiments
    python scripts/fetch_imperial_summaries.py 1 2 3    # deciding set - only after pre-registration

Kirkaldy et al. 2024, doi:10.5281/zenodo.10637534 (CC-BY-4.0). Reads only the
small "Summary Data/Performance Summary/*.csv" members of each experiment's
zip through HTTP range requests. Writes data/raw/imperial_summaries/expt<N>/.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_imperial_m50t import _RangeFile  # noqa: E402

BASE = "https://zenodo.org/api/records/10637534/files/"
ZIPS = {
    1: "Expt%201%20-%20Si-based%20Degradation.zip",
    2: "Expt%202,2%20-%20C-based%20Degradation%202.zip",
    3: "Expt%203%20-%20Cathode%20Degradation%20and%20Li-Plating.zip",
    4: "Expt%204%20-%20Drive%20Cycle%20Aging%20(Control).zip",
    5: "Expt%205%20-%20Standard%20Cycle%20Aging%20(Control).zip",
}
OUT = Path("data/raw/imperial_summaries")


def fetch(n: int) -> int:
    out = OUT / f"expt{n}"
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BufferedReader(_RangeFile(BASE + ZIPS[n] + "/content"), buffer_size=1 << 20))
    names = [m for m in z.namelist() if "/Performance Summary/" in m and m.endswith(".csv")]
    for name in names:
        (out / name.rsplit("/", 1)[1]).write_bytes(z.read(name))
    return len(names)


def main() -> int:
    for arg in sys.argv[1:] or ["4", "5"]:
        print(f"expt {arg}: {fetch(int(arg))} files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
