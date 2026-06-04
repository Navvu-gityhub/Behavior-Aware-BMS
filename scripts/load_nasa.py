#!/usr/bin/env python3
"""
CLI wrapper for loading one NASA battery data file.

Example:
    python scripts/load_nasa.py --input data/interim/nasa/sample.csv --out data/processed/nasa/nasa_sample_processed.csv --max-rows 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allows running from project root without installing package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bms.io.load_nasa import load_nasa_file, save_processed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NASA CSV/XLSX/XLS/TXT file")
    parser.add_argument("--out", default="data/processed/nasa/nasa_processed.csv")
    parser.add_argument("--sheet-name", default=None, help="Excel sheet name if needed")
    parser.add_argument("--max-rows", type=int, default=None, help="Small smoke-test row limit")
    args = parser.parse_args()

    try:
        df = load_nasa_file(args.input, sheet_name=args.sheet_name, max_rows=args.max_rows)
        out = save_processed(df, args.out)
        print(f"NASA load OK: rows={len(df)} cols={len(df.columns)} -> {out}")
        print("Columns:", ", ".join(df.columns[:20]))
        return 0
    except Exception as exc:
        print(f"NASA load FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
