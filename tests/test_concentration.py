"""Tests for the concentration kernels used by Phase 18's pass-rush analysis."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from season_metrics import herfindahl, top_share

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def test_top_share_single_contributor():
    assert top_share([9.0]) == pytest.approx(1.0)


def test_top_share_even_split():
    assert top_share([2.0, 2.0, 2.0, 2.0]) == pytest.approx(0.25)


def test_top_share_uses_the_max_not_the_last():
    assert top_share([1.0, 8.0, 1.0]) == pytest.approx(0.8)


def test_top_share_handles_halves():
    # Half-sack credits are the reason this kernel takes floats at all.
    assert top_share([0.5, 0.5, 1.0]) == pytest.approx(0.5)


def test_top_share_empty_and_zero_are_nan_not_zero():
    # A zero share would rank as "perfectly distributed" and quietly win the
    # leaderboard; NaN drops out of the reference set instead.
    assert np.isnan(top_share([]))
    assert np.isnan(top_share([0.0, 0.0]))


def test_top_share_ignores_nan():
    assert top_share([np.nan, 3.0, 1.0]) == pytest.approx(0.75)


def test_herfindahl_single_contributor_is_one():
    assert herfindahl([5.0]) == pytest.approx(1.0)


def test_herfindahl_even_split_is_one_over_n():
    assert herfindahl([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.25)
    assert herfindahl([3.0, 3.0]) == pytest.approx(0.5)


def test_herfindahl_is_scale_invariant():
    assert herfindahl([1.0, 3.0]) == pytest.approx(herfindahl([10.0, 30.0]))


def test_herfindahl_separates_distributions_that_share_a_top_share():
    # Both have a leader on 50%, but the tail differs -- which is the whole
    # reason Phase 18 reports this alongside top_share.
    two_way = herfindahl([5.0, 5.0])
    long_tail = herfindahl([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert two_way > long_tail


def test_herfindahl_empty_is_nan():
    assert np.isnan(herfindahl([]))


@pytest.mark.skipif(
    not (PROCESSED / "defender_season.csv").exists(),
    reason="defender_season.csv not built; run src/season_metrics.py",
)
def test_matches_project_reported_value():
    """SEA 2025 must reconcile to the published 47.0 sacks and 14.9% top share."""
    import pandas as pd

    d = pd.read_csv(PROCESSED / "defender_season.csv")
    sea = d[
        (d["season"] == 2025)
        & (d["season_type"] == "REG")
        & (d["team"] == "SEA")
        & (d["sacks"] > 0)
    ]
    assert sea["sacks"].sum() == pytest.approx(47.0)
    assert top_share(sea["sacks"]) == pytest.approx(0.149, abs=0.001)
    # Three players tied at the top is the detail the copy leans on.
    assert (sea["sacks"] == sea["sacks"].max()).sum() == 3
