"""Self-contained interactive HTML dashboard.

Previously described in the README ("Interactive HTML dashboard, no server
required") but not implemented anywhere in the repository. This generates a
single dashboard.html with no external asset dependencies (charts are
embedded as base64 PNGs) so it can be opened directly in a browser.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_STATE_COLORS = {
    "HEALTHY": "#2e7d32",
    "WARNING": "#f9a825",
    "DEGRADED": "#ef6c00",
    "CRITICAL": "#c62828",
}
_RISK_COLORS = {
    "LOW": "#2e7d32",
    "MEDIUM": "#f9a825",
    "HIGH": "#ef6c00",
    "CRITICAL": "#c62828",
}


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _bar_chart(counts: pd.Series, colors: dict, title: str) -> str:
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = list(counts.index)
    values = list(counts.values)
    bar_colors = [colors.get(l, "#607d8b") for l in labels]
    ax.bar(labels, values, color=bar_colors)
    ax.set_title(title)
    ax.set_ylabel("Battery count")
    fig.tight_layout()
    return _fig_to_base64(fig)


def build_dashboard(battery: pd.DataFrame, output_path: str | Path = "dashboard.html") -> Path:
    """Render the final per-battery table (state, risk, RUL, guardian report) to HTML.

    Expects a dataframe containing at least: battery_id, health_index,
    battery_state, risk_score, risk_level, rul_cycles, replacement_policy,
    guardian_report.
    """
    required = {
        "battery_id", "health_index", "battery_state", "risk_score",
        "risk_level", "rul_cycles", "replacement_policy", "guardian_report",
    }
    missing = required - set(battery.columns)
    if missing:
        raise ValueError(f"build_dashboard: missing required columns {sorted(missing)}")

    state_counts = battery["battery_state"].value_counts().reindex(
        ["HEALTHY", "WARNING", "DEGRADED", "CRITICAL"]
    ).fillna(0).astype(int)
    risk_counts = battery["risk_level"].value_counts().reindex(
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ).fillna(0).astype(int)

    state_chart = _bar_chart(state_counts, _STATE_COLORS, "Battery Health State")
    risk_chart = _bar_chart(risk_counts, _RISK_COLORS, "Degradation Risk Level")

    display_cols = [
        "battery_id", "health_index", "battery_state", "risk_score",
        "risk_level", "rul_cycles", "replacement_policy",
    ]
    table_df = battery[display_cols].sort_values("risk_score", ascending=False).round(1)

    rows_html = []
    for _, row in table_df.iterrows():
        state_color = _STATE_COLORS.get(row["battery_state"], "#607d8b")
        risk_color = _RISK_COLORS.get(row["risk_level"], "#607d8b")
        rows_html.append(
            "<tr>"
            f"<td>{row['battery_id']}</td>"
            f"<td>{row['health_index']}</td>"
            f"<td><span class='pill' style='background:{state_color}'>{row['battery_state']}</span></td>"
            f"<td>{row['risk_score']}</td>"
            f"<td><span class='pill' style='background:{risk_color}'>{row['risk_level']}</span></td>"
            f"<td>{row['rul_cycles']}</td>"
            f"<td>{row['replacement_policy']}</td>"
            "</tr>"
        )

    reports_html = "".join(
        f"<div class='report'><strong>{r.battery_id}</strong><p>{r.guardian_report}</p></div>"
        for r in battery.sort_values("risk_score", ascending=False).itertuples()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Behavior-Aware BMS Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 24px; background: #f5f6f8; color: #1a1a1a; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 24px; font-size: 13px; }}
  .charts {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .card {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 13px; }}
  th {{ background: #fafafa; font-weight: 600; }}
  .pill {{ color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
  .report {{ background: white; border-radius: 8px; padding: 10px 16px; margin-bottom: 8px; font-size: 13px; }}
  .report strong {{ color: #333; }}
  .warning-note {{ background: #fff8e1; border-left: 3px solid #f9a825; padding: 10px 16px; margin-bottom: 20px; font-size: 12px; color: #665; }}
</style>
</head>
<body>
  <h1>Behavior-Aware BMS Dashboard</h1>
  <div class="subtitle">{len(battery)} batteries · generated by dashboard.build_dashboard</div>
  <div class="warning-note">
    Risk, health, and RUL scores are rule-based heuristics with hand-chosen weights,
    not validated against measured degradation outcomes. Treat as directional, not predictive.
  </div>
  <div class="charts">
    <div class="card"><img src="data:image/png;base64,{state_chart}" alt="state distribution"></div>
    <div class="card"><img src="data:image/png;base64,{risk_chart}" alt="risk distribution"></div>
  </div>
  <div class="card" style="margin-bottom:24px;">
    <table>
      <thead><tr><th>Battery</th><th>Health Index</th><th>State</th><th>Risk Score</th><th>Risk Level</th><th>RUL (cycles)</th><th>Policy</th></tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
  </div>
  <h2 style="font-size:16px;">Guardian Reports</h2>
  {reports_html}
</body>
</html>"""

    out_path = Path(output_path)
    out_path.write_text(html, encoding="utf-8")
    return out_path
