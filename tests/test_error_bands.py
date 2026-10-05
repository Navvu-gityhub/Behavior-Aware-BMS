"""The error-band constants must equal the artifact they were taken from.

If `scripts/build_error_bands.py` is re-run and a band moves, this fails until
`src/bms/health/error_bands.py` is updated to match - the same contract
`test_reported_numbers.py` enforces for quoted figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.error_bands import (  # noqa: E402
    RUL_BANDS,
    SOH_P90_ABS_FIXED,
    SOH_P90_ABS_LEARNED,
    describe_rul_band,
    rul_band,
    soh_band,
)

ARTIFACT = Path(__file__).resolve().parents[1] / "reports" / "metrics" / "error_bands.csv"


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT)


def test_soh_bands_match_the_artifact(table):
    fixed = table[table["estimator"] == "soh_fixed_window"].iloc[0]
    learned = table[table["estimator"] == "soh_learned_window"].iloc[0]
    assert SOH_P90_ABS_FIXED == pytest.approx(fixed["p90_abs"])
    assert SOH_P90_ABS_LEARNED == pytest.approx(learned["p90_abs"])


def test_rul_bands_match_the_artifact(table):
    rows = table[table["estimator"] == "rul_fade_extrapolation"].reset_index(drop=True)
    assert len(rows) == len(RUL_BANDS)
    for band, row in zip(RUL_BANDS, rows.itertuples(), strict=True):
        assert f"{band.low:.0f}-{band.high:.0f}" == row.bin
        assert band.later_q05 == pytest.approx(row.true_minus_pred_q05)
        assert band.later_q50 == pytest.approx(row.true_minus_pred_q50)
        assert band.later_q95 == pytest.approx(row.true_minus_pred_q95)
        assert band.share_at_least == pytest.approx(row.share_lasted_at_least_predicted)
        assert band.n == row.n and band.cells == row.cells


def test_band_lookup_covers_every_prediction():
    assert rul_band(0).low == 0
    assert rul_band(24.9).high == 25
    assert rul_band(5000).high >= 100_000


def test_learned_window_gets_the_wider_band():
    assert soh_band("learned") > soh_band("fixed")


def test_description_never_reports_a_negative_lower_bound():
    assert "between 0 and" in describe_rul_band(10) or "between 11" in describe_rul_band(10)
    text = describe_rul_band(30)          # q05 is -14, so 30 - 14 = 16
    assert "between 16 and 159" in text
