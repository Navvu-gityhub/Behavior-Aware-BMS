"""
Day 5 - BMS raw-data audit and quick review plots.

Run from the root of the Behavior-Aware-BMS repository:

    python scripts/day5_data_audit.py

Outputs:
    reports/weekly/week1_review/
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"

OUT_DIR = ROOT / "reports" / "weekly" / "week1_review"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

for folder in [OUT_DIR, FIG_DIR, TABLE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

STANDARD_COLUMNS = [
    "timestamp",
    "cycle",
    "voltage_v",
    "current_a",
    "temperature_c",
    "capacity_ah",
    "soc",
    "soh",
    "power_w",
    "mode_guess",
    "source",
    "source_file",
    "impedance_ohm",
    "resistance_ohm",
    "dcir_ohm",
    "acir_ohm",
]

NUMERIC_CANDIDATES = {
    "voltage_v": [
        "voltage_v",
        "voltage",
        "voltage_measured",
        "v",
    ],
    "current_a": [
        "current_a",
        "current",
        "current_measured",
        "i",
    ],
    "temperature_c": [
        "temperature_c",
        "temperature",
        "temp",
        "temp_c",
        "temperature_measured",
    ],
    "capacity_ah": [
        "capacity_ah",
        "capacity",
        "capacity_mah",
        "q",
        "discharge_capacity",
    ],
    "cycle": [
        "cycle",
        "cycle_index",
        "cycle_number",
    ],
    "soc": [
        "soc",
        "state_of_charge",
    ],
    "soh": [
        "soh",
        "state_of_health",
    ],
    "impedance_ohm": [
        "impedance_ohm",
        "impedance",
        "battery_impedance",
        "re",
        "rct",
    ],
    "resistance_ohm": [
        "resistance_ohm",
        "resistance",
        "internal_resistance",
        "r0",
        "dcir",
    ],
}


def scan_files() -> pd.DataFrame:
    """
    Scan all files inside data/ and create a file inventory.
    """

    rows = []

    if not DATA_DIR.exists():
        return pd.DataFrame(
            columns=["stage", "dataset", "file", "extension", "size_kb"]
        )

    for file in DATA_DIR.rglob("*"):
        if file.is_file() and file.name != ".gitkeep":
            rel = file.relative_to(DATA_DIR)
            parts = rel.parts

            stage = parts[0] if len(parts) > 0 else "unknown"
            dataset = parts[1] if len(parts) > 1 else "unknown"

            rows.append(
                {
                    "stage": stage,
                    "dataset": dataset,
                    "file": str(rel),
                    "extension": file.suffix.lower() or "no_ext",
                    "size_kb": round(file.stat().st_size / 1024, 2),
                }
            )

    return pd.DataFrame(rows)


def read_table(path: Path) -> pd.DataFrame | None:
    """
    Read common tabular battery data files.
    """

    try:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)

        if suffix == ".parquet":
            return pd.read_parquet(path)

        if suffix == ".json":
            return pd.read_json(path)

    except Exception as exc:
        print(f"Could not read {path}: {exc}")

    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns into a common style and map common aliases.
    """

    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in out.columns
    ]

    for canonical, aliases in NUMERIC_CANDIDATES.items():
        if canonical not in out.columns:
            for alias in aliases:
                if alias in out.columns:
                    out[canonical] = out[alias]
                    break

    for col in STANDARD_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="ignore")

    return out


def load_available_tables(file_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load all readable tables from data/.
    """

    frames = []
    readable_ext = {".csv", ".xlsx", ".xls", ".parquet", ".json"}

    for _, row in file_df.iterrows():
        path = DATA_DIR / row["file"]

        if path.suffix.lower() not in readable_ext:
            continue

        df = read_table(path)

        if df is None or df.empty:
            continue

        df = normalize_columns(df)

        if "source" not in df.columns:
            df["source"] = row["dataset"]

        if "source_file" not in df.columns:
            df["source_file"] = str(row["file"])

        df["_stage"] = row["stage"]
        df["_dataset"] = row["dataset"]

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def save_plot(fig, name: str) -> str:
    """
    Save one matplotlib figure.
    """

    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_file_coverage(file_df: pd.DataFrame) -> str | None:
    """
    Plot number of files available by data stage and dataset.
    """

    if file_df.empty:
        return None

    coverage = file_df.groupby(["stage", "dataset"]).size().unstack(fill_value=0)
    coverage.to_csv(TABLE_DIR / "file_coverage.csv")

    ax = coverage.plot(kind="bar", figsize=(9, 5))
    ax.set_title("File coverage by data stage and dataset")
    ax.set_xlabel("Data stage")
    ax.set_ylabel("Number of files")
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")

    return save_plot(ax.get_figure(), "01_file_coverage_by_stage.png")


def plot_missingness(df: pd.DataFrame) -> str | None:
    """
    Plot missing values percentage for important battery columns.
    """

    if df.empty:
        return None

    cols = [c for c in STANDARD_COLUMNS if c in df.columns]

    if not cols:
        cols = list(df.columns[: min(20, len(df.columns))])

    miss = df[cols].isna().mean().sort_values(ascending=False) * 100
    miss.rename("missing_percent").to_csv(TABLE_DIR / "missingness_percent.csv")

    fig, ax = plt.subplots(figsize=(9, max(4, len(cols) * 0.28)))
    ax.barh(miss.index.astype(str), miss.values)
    ax.set_title("Missingness by column")
    ax.set_xlabel("Missing values (%)")
    ax.invert_yaxis()

    return save_plot(fig, "02_missingness_by_column.png")


def plot_temperature_distribution(df: pd.DataFrame) -> str | None:
    """
    Plot battery temperature distribution.
    """

    col = "temperature_c"

    if df.empty or col not in df.columns:
        return None

    temp = pd.to_numeric(df[col], errors="coerce").dropna()

    if temp.empty:
        return None

    temp.describe().to_csv(TABLE_DIR / "temperature_summary.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(temp, bins=30)
    ax.set_title("Temperature distribution")
    ax.set_xlabel("Temperature (degree C)")
    ax.set_ylabel("Reading count")

    return save_plot(fig, "03_temperature_distribution.png")


def plot_capacity_traces(df: pd.DataFrame) -> str | None:
    """
    Plot capacity values over cycle/index.
    """

    if df.empty or "capacity_ah" not in df.columns:
        return None

    y = pd.to_numeric(df["capacity_ah"], errors="coerce")

    if y.dropna().empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))

    group_col = "source_file" if "source_file" in df.columns else "source"
    x_col = "cycle" if "cycle" in df.columns else None

    plotted = 0

    for name, g in df.assign(_cap=y).groupby(group_col):
        cap = pd.to_numeric(g["_cap"], errors="coerce")

        if cap.dropna().empty:
            continue

        if x_col:
            x = pd.to_numeric(g[x_col], errors="coerce")
        else:
            x = np.arange(len(g))

        ax.plot(
            x,
            cap,
            marker="o",
            linewidth=1,
            markersize=2,
            label=str(name)[:35],
        )

        plotted += 1

        if plotted >= 8:
            break

    ax.set_title("Capacity traces")
    ax.set_xlabel("Cycle" if x_col else "Row index")
    ax.set_ylabel("Capacity (Ah)")

    if plotted:
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

    return save_plot(fig, "04_capacity_traces.png")


def plot_impedance_availability(df: pd.DataFrame) -> str | None:
    """
    Plot availability of impedance/resistance related fields.
    """

    if df.empty:
        return None

    impedance_cols = [
        c
        for c in ["impedance_ohm", "resistance_ohm", "dcir_ohm", "acir_ohm"]
        if c in df.columns
    ]

    availability = {}

    for col in impedance_cols:
        availability[col] = float(
            pd.to_numeric(df[col], errors="coerce").notna().mean() * 100
        )

    if not availability:
        availability = {"impedance/resistance columns present": 0.0}

    pd.Series(
        availability,
        name="available_percent",
    ).to_csv(TABLE_DIR / "impedance_availability.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(availability.keys()), list(availability.values()))
    ax.set_title("Impedance / resistance availability")
    ax.set_ylabel("Available readings (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20)

    return save_plot(fig, "05_impedance_availability.png")


def plot_soc_soh(df: pd.DataFrame) -> str | None:
    """
    Plot SOC and SOH distribution if present.
    """

    cols = [c for c in ["soc", "soh"] if c in df.columns]

    if df.empty or not cols:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()

        if not vals.empty:
            ax.hist(vals, bins=25, alpha=0.5, label=col)

    ax.set_title("SOC / SOH quick distribution")
    ax.set_xlabel("Percent")
    ax.set_ylabel("Reading count")
    ax.legend()

    return save_plot(fig, "06_soc_soh_distribution.png")


def write_markdown(
    file_df: pd.DataFrame,
    df: pd.DataFrame,
    figures: list[str],
) -> Path:
    """
    Write final Day 5 review pack markdown.
    """

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files_found": int(len(file_df)),
        "rows_loaded": int(len(df)),
        "columns_loaded": int(len(df.columns)) if not df.empty else 0,
        "figures_generated": len(figures),
    }

    summary_path = OUT_DIR / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not df.empty:
        df.head(200).to_csv(
            TABLE_DIR / "loaded_preview_first_200_rows.csv",
            index=False,
        )

        schema_df = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(df[c].dtype) for c in df.columns],
            }
        )

        schema_df.to_csv(TABLE_DIR / "schema_detected.csv", index=False)

    md = [
        "# Week 1 Review Pack - Day 5 Data Audit",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## What was completed",
        "",
        "- Audited available raw, interim, and processed data files.",
        "- Checked column coverage and missing values.",
        "- Created quick descriptive plots for review only, not final modelling.",
        "- Prepared tables and figures under `reports/weekly/week1_review/`.",
        "",
        "## Audit summary",
        "",
        f"- Files found: {summary['files_found']}",
        f"- Rows loaded from readable tables: {summary['rows_loaded']}",
        f"- Columns loaded: {summary['columns_loaded']}",
        f"- Figures generated: {summary['figures_generated']}",
        "",
        "## Figures",
        "",
    ]

    if figures:
        for fig_path in figures:
            md.append(f"- `{Path(fig_path).name}`")
    else:
        md.append(
            "- No figures generated. Add readable CSV/XLSX/Parquet files under `data/` and rerun the script."
        )

    md += [
        "",
        "## Review notes for team",
        "",
        "This Day 5 work is a data-readiness checkpoint. It does not train a model yet. "
        "The goal is to confirm what data exists, what fields are missing, and whether "
        "the key battery patterns can be visualized before feature engineering.",
        "",
        "## Next step",
        "",
        "Use the audit output to decide which dataset is cleanest for Day 6 feature extraction: "
        "temperature stress, deep discharge, high SOC storage, fast charge events, and capacity fade indicators.",
    ]

    out = OUT_DIR / "week1_review_pack.md"
    out.write_text("\n".join(md), encoding="utf-8")

    return out


def main() -> None:
    print("Day 5 BMS data audit started")

    file_df = scan_files()
    file_df.to_csv(TABLE_DIR / "file_inventory.csv", index=False)

    df = load_available_tables(file_df)

    figures = []

    plot_functions = [
        plot_file_coverage,
        plot_missingness,
        plot_temperature_distribution,
        plot_capacity_traces,
        plot_impedance_availability,
        plot_soc_soh,
    ]

    for fn in plot_functions:
        try:
            if fn.__name__ == "plot_file_coverage":
                result = fn(file_df)
            else:
                result = fn(df)

            if result:
                figures.append(result)
                print("Saved:", result)

        except Exception as exc:
            print(f"Plot failed: {fn.__name__}: {exc}")

    md_path = write_markdown(file_df, df, figures)

    print("Saved:", md_path)
    print("DAY 5 DATA AUDIT COMPLETED")


if __name__ == "__main__":
    main()
