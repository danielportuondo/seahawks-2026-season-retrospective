import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from season_metrics import (
    percentile_rank,
    rolling_window_best,
    time_weighted_margin,
    time_weighted_share,
)

ADVANCED_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "team_season_advanced.csv"


# --- time_weighted_margin -------------------------------------------------


def test_lead_held_for_exactly_half_the_game():
    # +7 from the opening kickoff until the 1800s mark, 0 thereafter -> 3.5.
    seconds = [3600, 1800]
    margins = [7, 0]
    assert time_weighted_margin(seconds, margins) == pytest.approx(3.5)


def test_constant_margin_equals_that_margin():
    assert time_weighted_margin([3600, 2000, 500], [10, 10, 10]) == pytest.approx(10.0)


def test_late_lead_scores_lower_than_early_lead_at_equal_final_margin():
    early = time_weighted_margin([3600, 1800], [14, 14])
    late = time_weighted_margin([3600, 1800], [0, 14])
    assert early > late
    # Both finish +14; only the clock-weighting separates them.
    assert late == pytest.approx(7.0)


def test_play_order_does_not_matter():
    shuffled = time_weighted_margin([1800, 3600, 900], [7, 0, 14])
    ordered = time_weighted_margin([3600, 1800, 900], [0, 7, 14])
    assert shuffled == pytest.approx(ordered)


def test_empty_game_is_nan():
    assert np.isnan(time_weighted_margin([], []))


def test_zero_duration_is_nan():
    # Every play stamped at the same clock reading (the overtime case).
    assert np.isnan(time_weighted_margin([0, 0, 0], [3, 3, 3]))


# --- time_weighted_share --------------------------------------------------


def test_share_leading_for_final_quarter_only():
    assert time_weighted_share([3600, 900], [False, True]) == pytest.approx(0.25)


def test_share_is_one_when_always_true():
    assert time_weighted_share([3600, 1000], [True, True]) == pytest.approx(1.0)


# --- percentile_rank ------------------------------------------------------


def test_percentile_of_maximum():
    assert percentile_rank([1, 2, 3, 4], 4) == pytest.approx(87.5)  # 3 below + half of 1 tie


def test_percentile_of_minimum():
    assert percentile_rank([1, 2, 3, 4], 1) == pytest.approx(12.5)


def test_percentile_above_everything():
    assert percentile_rank([1, 2, 3], 99) == pytest.approx(100.0)


def test_ties_count_as_half():
    assert percentile_rank([5, 5, 5, 5], 5) == pytest.approx(50.0)


def test_nans_are_ignored():
    assert percentile_rank([1, np.nan, 3], 3) == pytest.approx(75.0)


# --- rolling_window_best --------------------------------------------------


def test_finds_lowest_window_and_its_start():
    series = [0.0, 0.0, -5.0, -5.0, 0.0]
    best = rolling_window_best(series, window=2, mode="min")
    assert best["value"] == pytest.approx(-5.0)
    assert best["start_index"] == 2


def test_max_mode_finds_highest_window():
    best = rolling_window_best([1.0, 9.0, 9.0, 1.0], window=2, mode="max")
    assert best["value"] == pytest.approx(9.0)
    assert best["start_index"] == 1


def test_window_longer_than_series_is_nan():
    best = rolling_window_best([1.0, 2.0], window=8)
    assert np.isnan(best["value"])
    assert best["start_index"] is None


# --- ground truth on the committed table ----------------------------------
# The one place these tests touch data: team_season_advanced.csv is the contract
# every downstream phase, the dashboard, and the narrative depend on, so its
# headline numbers are pinned to the published record.


@pytest.fixture(scope="module")
def sea_2025_reg():
    if not ADVANCED_CSV.exists():
        pytest.skip("run src/season_metrics.py first")
    df = pd.read_csv(ADVANCED_CSV)
    row = df[(df["team"] == "SEA") & (df["season"] == 2025) & (df["season_type"] == "REG")]
    assert len(row) == 1
    return row.iloc[0]


def test_sea_2025_record(sea_2025_reg):
    assert sea_2025_reg["wins"] == 14
    assert sea_2025_reg["losses"] == 3


def test_sea_2025_scoring(sea_2025_reg):
    assert sea_2025_reg["points_for"] == 483
    assert sea_2025_reg["point_diff"] == 191
    assert sea_2025_reg["points_against_per_game"] == pytest.approx(17.2, abs=0.05)


def test_regular_season_row_count():
    if not ADVANCED_CSV.exists():
        pytest.skip("run src/season_metrics.py first")
    df = pd.read_csv(ADVANCED_CSV)
    reg = df[df["season_type"] == "REG"]
    assert len(reg) == 3 * 31 + 24 * 32  # 31 teams 1999-2001, 32 from 2002
