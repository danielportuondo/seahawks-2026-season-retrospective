"""Fixture tests for the player-level frames behind Phases 17 and 18.

These guard the two failure modes that are silent rather than loud: the wrong
receiver key (which undercounts targets for six seasons) and the wrong pass-attempt
denominator (which deflates every target share in the file).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
RECEIVERS = PROCESSED / "receiver_season.csv"
PASSERS = PROCESSED / "passer_season.csv"

needs_receivers = pytest.mark.skipif(
    not RECEIVERS.exists(), reason="receiver_season.csv not built; run src/season_metrics.py"
)
needs_passers = pytest.mark.skipif(
    not PASSERS.exists(), reason="passer_season.csv not built; run src/season_metrics.py"
)


@needs_receivers
def test_smith_njigba_2025_matches_published_line():
    r = pd.read_csv(RECEIVERS)
    sea = r[(r["season"] == 2025) & (r["season_type"] == "REG") & (r["team"] == "SEA")]
    jsn = sea.sort_values("receiving_yards", ascending=False).iloc[0]
    assert jsn["receiver"] == "J.Smith-Njigba"
    assert int(jsn["targets"]) == 163
    assert int(jsn["receptions"]) == 119
    assert int(jsn["receiving_yards"]) == 1793
    assert int(jsn["receiving_tds"]) == 10


@needs_receivers
def test_team_pass_attempts_match_the_official_denominator():
    """nflverse pass_attempt includes sacks and two-point tries; official 2025 SEA is 481."""
    r = pd.read_csv(RECEIVERS)
    sea = r[(r["season"] == 2025) & (r["season_type"] == "REG") & (r["team"] == "SEA")]
    assert int(sea["team_pass_attempts"].iloc[0]) == 481
    assert sea["team_pass_attempts"].nunique() == 1


@needs_receivers
def test_target_share_reproduces_the_published_figure():
    r = pd.read_csv(RECEIVERS)
    sea = r[(r["season"] == 2025) & (r["season_type"] == "REG") & (r["team"] == "SEA")]
    jsn = sea.sort_values("receiving_yards", ascending=False).iloc[0]
    assert jsn["target_share"] == pytest.approx(0.339, abs=0.001)
    assert jsn["team_receiving_yards_share"] == pytest.approx(0.441, abs=0.001)


@needs_receivers
def test_receiver_key_did_not_collapse_in_the_2003_2008_window():
    """receiver_player_id is null on incompletions 2003-2008; receiver_id is not.

    If the wrong key is ever swapped in, catch rate in this window jumps towards
    100% while the rest of the file stays near 60%. This asserts it has not.
    """
    r = pd.read_csv(RECEIVERS)
    reg = r[(r["season_type"] == "REG") & (r["targets"] >= 50)].copy()
    reg["catch_rate"] = reg["receptions"] / reg["targets"]
    window = reg[reg["season"].between(2003, 2008)]["catch_rate"].mean()
    rest = reg[~reg["season"].between(2003, 2008)]["catch_rate"].mean()
    assert window < 0.75, f"catch rate {window:.3f} in 2003-2008 -- wrong receiver key?"
    assert abs(window - rest) < 0.08, f"2003-2008 catch rate {window:.3f} vs {rest:.3f} elsewhere"


@needs_passers
def test_darnold_2025_reproduces_phase6_ground_truth():
    p = pd.read_csv(PASSERS)
    d = p[(p["player_name"] == "S.Darnold") & (p["season"] == 2025)]
    reg = d[d["season_type"] == "REG"].iloc[0]
    assert int(reg["dropbacks"]) == 516
    assert int(reg["interceptions"]) == 14
    assert int(reg["fumbles_lost"]) == 6
    assert int(reg["turnovers"]) == 20

    post = d[d["season_type"] == "POST"].iloc[0]
    assert int(post["dropbacks"]) == 99
    assert int(post["turnovers"]) == 0


@needs_passers
def test_darnold_career_spans_every_team_he_played_for():
    """The frame is keyed by player, so following him across teams is free."""
    p = pd.read_csv(PASSERS)
    reg = p[(p["player_name"] == "S.Darnold") & (p["season_type"] == "REG")]
    assert set(reg["team"]) == {"NYJ", "CAR", "SF", "MIN", "SEA"}
    assert sorted(reg["season"]) == list(range(2018, 2026))


@needs_passers
def test_2025_is_darnolds_worst_turnover_rate():
    """The arc the Players tab describes: career-worst, then zero."""
    p = pd.read_csv(PASSERS)
    reg = p[(p["player_name"] == "S.Darnold") & (p["season_type"] == "REG")]
    worst = reg.sort_values("turnover_rate_per_dropback", ascending=False).iloc[0]
    assert int(worst["season"]) == 2025
