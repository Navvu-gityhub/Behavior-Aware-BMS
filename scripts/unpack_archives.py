#!/usr/bin/env python3
"""
Day 3: Archive unpack + file discovery for BMS datasets.

What it does:
1. Scans a raw data folder recursively.
2. Extracts .zip/.tar/.tar.gz/.tgz archives into an interim folder.
3. Discovers loadable files: .csv, .xlsx, .xls, .txt.
4. Writes a manifest CSV so later loaders do not need manual clicking.
5. Logs failed archive/file operations to CSV.

Example:
    python scripts/unpack_archives.py --raw data/raw --out data/interim --manifest data/interim/discovered_files.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

SUPPORTED_DATA_EXTENSIONS = {".csv", ".xlsx", ".xls", ".txt"}
SUPPORTED_ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".tar.gz"}


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("unpack_archives")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def sha256_short(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}
        or name.endswith(".tar.gz")
    )


def safe_extract_zip(archive_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            target = extract_to / member.filename
            if not str(target.resolve()).startswith(str(extract_to.resolve())):
                raise ValueError(f"Unsafe zip path blocked: {member.filename}")
        zf.extractall(extract_to)


def safe_extract_tar(archive_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            target = extract_to / member.name
            if not str(target.resolve()).startswith(str(extract_to.resolve())):
                raise ValueError(f"Unsafe tar path blocked: {member.name}")
        tf.extractall(extract_to)


def extract_archive(archive_path: Path, out_root: Path, logger: logging.Logger) -> Path:
    stem = archive_path.name
    for suffix in [".tar.gz", ".zip", ".tar", ".tgz", ".gz"]:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    extract_to = out_root / "extracted" / stem
    if extract_to.exists():
        logger.info("Already extracted: %s", archive_path)
        return extract_to

    logger.info("Extracting: %s -> %s", archive_path, extract_to)
    if archive_path.suffix.lower() == ".zip":
        safe_extract_zip(archive_path, extract_to)
    else:
        safe_extract_tar(archive_path, extract_to)
    return extract_to


def discover_files(paths: Iterable[Path]) -> list[Path]:
    discovered: list[Path] = []
    for root in paths:
        if not root.exists():
            continue
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS:
                discovered.append(file_path)
    return sorted(set(discovered))


def guess_source(path: Path) -> str:
    text = str(path).lower()
    if "nasa" in text:
        return "nasa"
    if "calce" in text or "umd" in text:
        return "calce"
    if "stanford" in text:
        return "stanford"
    return "unknown"


def write_manifest(files: list[Path], manifest_path: Path, logger: logging.Logger) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "discovered_at",
        "source_guess",
        "file_type",
        "file_name",
        "file_path",
        "size_bytes",
        "sha256_16",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for path in files:
            try:
                writer.writerow(
                    {
                        "discovered_at": datetime.now().isoformat(timespec="seconds"),
                        "source_guess": guess_source(path),
                        "file_type": path.suffix.lower().lstrip("."),
                        "file_name": path.name,
                        "file_path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256_16": sha256_short(path),
                    }
                )
            except Exception as exc:
                logger.warning("Could not add to manifest: %s | %s", path, exc)


def write_failures(failures: list[dict], failure_log: Path) -> None:
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    fields = ["time", "stage", "file_path", "error"]
    with failure_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw", help="Raw dataset folder")
    parser.add_argument("--out", default="data/interim", help="Interim output folder")
    parser.add_argument("--manifest", default="data/interim/discovered_files.csv")
    parser.add_argument("--log-file", default="data/interim/unpack_archives.log")
    parser.add_argument("--failure-log", default="data/interim/failed_files.csv")
    parser.add_argument(
        "--copy-loose-files",
        action="store_true",
        help="Copy loose CSV/XLSX/XLS/TXT files from raw to interim/loose_files",
    )
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    logger = setup_logger(Path(args.log_file))
    failures: list[dict] = []

    if not raw.exists():
        logger.error("Raw folder does not exist: %s", raw)
        return 1

    extracted_roots: list[Path] = []
    for path in raw.rglob("*"):
        if path.is_file() and is_archive(path):
            try:
                extracted_roots.append(extract_archive(path, out, logger))
            except Exception as exc:
                logger.exception("Failed to extract archive: %s", path)
                failures.append(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "stage": "archive_extract",
                        "file_path": str(path),
                        "error": repr(exc),
                    }
                )

    if args.copy_loose_files:
        loose_out = out / "loose_files"
        loose_out.mkdir(parents=True, exist_ok=True)
        for path in discover_files([raw]):
            if "extracted" not in path.parts:
                try:
                    target = loose_out / path.name
                    if not target.exists():
                        shutil.copy2(path, target)
                except Exception as exc:
                    failures.append(
                        {
                            "time": datetime.now().isoformat(timespec="seconds"),
                            "stage": "copy_loose_file",
                            "file_path": str(path),
                            "error": repr(exc),
                        }
                    )

    search_roots = [raw, out] + extracted_roots
    files = discover_files(search_roots)
    write_manifest(files, Path(args.manifest), logger)
    write_failures(failures, Path(args.failure_log))

    logger.info("Discovered %d loadable files.", len(files))
    logger.info("Manifest written: %s", args.manifest)
    logger.info("Failure log written: %s", args.failure_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
