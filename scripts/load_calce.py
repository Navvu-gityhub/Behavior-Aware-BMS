#!/usr/bin/env python3
"""
CLI wrapper for loading one CALCE battery data file.

Example:
    python scripts/load_calce.py --input data/interim/calce/sample.xlsx --out data/processed/calce/calce_sample_processed.csv --max-rows 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bms.io.load_calce import load_calce_file, save_processed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CALCE CSV/XLSX/XLS/TXT file")
    parser.add_argument("--out", default="data/processed/calce/calce_processed.csv")
    parser.add_argument("--sheet-name", default=None, help="Excel sheet name if needed")
    parser.add_argument("--max-rows", type=int, default=None, help="Small smoke-test row limit")
    args = parser.parse_args()

    try:
        df = load_calce_file(args.input, sheet_name=args.sheet_name, max_rows=args.max_rows)
        out = save_processed(df, args.out)
        print(f"CALCE load OK: rows={len(df)} cols={len(df.columns)} -> {out}")
        print("Columns:", ", ".join(df.columns[:20]))
        return 0
    except Exception as exc:
        print(f"CALCE load FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
