"""Render the §2 related-work survey from its extraction table.

The manuscript's premise is a claim about the literature: that holding out cells
is the field's de facto standard. §2.2 states that claim as a proportion, and a
proportion typed by hand into prose is exactly the kind of figure this project
has already caught itself getting wrong (see §6.4 of the manuscript). So the
proportions are computed here from `docs/manuscript/survey/extraction.csv` and
rendered, rather than written.

    python scripts/render_survey_table.py            # summary + table to stdout
    python scripts/render_survey_table.py --check    # exit 1 if not submittable

`--check` is the gate: it refuses while any row is unextracted, because a
proportion computed over a partial sample is not a proportion of the literature,
it is a proportion of whatever happened to be read first. Reporting it as the
former is the error the paper is about.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXTRACTION = REPO / "docs" / "manuscript" / "survey" / "extraction.csv"

#: Target sample size declared in §2.1. A survey that stops early is a different
#: survey, so the shortfall is reported rather than quietly accepted.
TARGET_N = 40

#: Rows counted as evidence. `abstract` is deliberately excluded: a split type
#: read off an abstract is a guess, and the whole point of the table is that
#: split types are frequently not stated where a reader would look.
COUNTABLE = ("full",)


def load(path: Path = EXTRACTION) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"no extraction table at {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarise(rows: list[dict[str, str]]) -> str:
    counted = [r for r in rows if r["verified"] in COUNTABLE]
    lines = [
        f"Rows in table:        {len(rows)}",
        f"Verified in full:     {len(counted)}",
        f"Abstract only:        {sum(1 for r in rows if r['verified'] == 'abstract')}",
        f"Pending retrieval:    {sum(1 for r in rows if r['verified'] == 'pending')}",
        f"Target sample (§2.1): {TARGET_N}",
        "",
    ]

    if not counted:
        lines.append("No fully verified rows: no proportion can be reported.")
        return "\n".join(lines)

    for field in ("split_type", "split_stated", "ceiling_reported",
                  "target_derivation", "uncertainty"):
        tally = Counter(r[field] for r in counted)
        lines.append(f"{field}:")
        for value, count in tally.most_common():
            pct = 100.0 * count / len(counted)
            lines.append(f"  {value:<24} {count:>3}  ({pct:.0f}% of {len(counted)})")
        lines.append("")

    lines.append(
        "Percentages are of fully verified rows only, and are NOT a description "
        "of the literature until the sample is complete."
    )
    return "\n".join(lines)


def markdown_table(rows: list[dict[str, str]]) -> str:
    header = (
        "| Ref | Year | Dataset | Split | Split stated | Ceiling | "
        "Target derivation | Uncertainty | Verified |\n"
        "|---|---:|---|---|---|---|---|---|---|\n"
    )
    body = "\n".join(
        f"| {r['key']} | {r['year']} | {r['dataset']} | {r['split_type']} | "
        f"{r['split_stated']} | {r['ceiling_reported']} | "
        f"{r['target_derivation']} | {r['uncertainty']} | {r['verified']} |"
        for r in sorted(rows, key=lambda r: (r["year"], r["key"]))
    )
    return header + body


def check(rows: list[dict[str, str]]) -> list[str]:
    """Reasons the survey is not yet submittable. Empty means it is."""
    problems: list[str] = []

    unverified = [r["key"] for r in rows if r["verified"] != "full"]
    if unverified:
        problems.append(
            f"{len(unverified)} row(s) not read in full: {', '.join(unverified)}. "
            f"A split type taken from an abstract is a guess, and §2.1 counts "
            f"only rows read against the paper's own methods section."
        )

    if len(rows) < TARGET_N:
        problems.append(
            f"{len(rows)} papers in the table against a declared target of "
            f"{TARGET_N} (§2.1). Either extract the remainder or amend the "
            f"protocol - a stated sampling plan that was not followed is worse "
            f"than a smaller stated plan."
        )

    for row in rows:
        if row["split_type"] == "unclear" and row["split_stated"] != "unclear":
            problems.append(
                f"{row['key']}: split_type is 'unclear' but split_stated is "
                f"'{row['split_stated']}' - these must agree."
            )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Exit 1 with reasons if the survey is not submittable.",
    )
    parser.add_argument(
        "--table", action="store_true", help="Print only the Markdown table."
    )
    args = parser.parse_args(argv)

    rows = load()

    if args.table:
        print(markdown_table(rows))
        return 0

    print(summarise(rows))
    print()
    print(markdown_table(rows))

    problems = check(rows)
    print()
    if problems:
        print("NOT SUBMITTABLE:")
        for problem in problems:
            print(f"  - {problem}")
        return 1 if args.check else 0

    print("Survey is complete and internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
