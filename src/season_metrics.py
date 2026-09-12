"""Shared advanced-metric builder for Phases 13-16: every team-season, 1999-2025.

Phases 2-7 only ever needed 2024 vs 2025 (or 2010-2025 for the Phase 5 defense
baseline). Phases 13-16 ask a different question -- "is this an all-time great
team?" -- which needs a denominator: every team-season nflverse publishes EPA and
win probability for. That is 1999 onward.

This module is the ONLY place raw play-by-play is read for those phases. It
writes two small committed CSVs that every downstream consumer (the four phase
scripts, the dashboard, the tests) reads instead of the multi-GB parquet cache,
so the repo stays reproducible without the download:

  data/processed/team_season_advanced.csv   one row per team-season-phase
  data/processed/team_game_control.csv      one row per SEA game, 2025 only

DEFINITIONAL CHOICES, spelled out because they are where this kind of analysis
goes quietly wrong:

- EPA/play and success rate reuse Phase 4's exact "core play" filter (down 1-4,
  play_type in {pass, run}, epa not null, win probability between 5% and 95%) so
  the numbers here are directly comparable to Phases 4, 5 and 7 rather than being
  a second, subtly different definition of the same statistic.
- Explosive plays are NOT garbage-time filtered and NOT restricted by down: a
  20-yard gain is a 20-yard gain. Thresholds are the public-analytics convention
  of 15+ air/receiving yards on a pass and 10+ on a rush.
- PRESSURE IS A PROXY. The cached pbp carries no participation or NGS data, so
  there is no pass-rusher count and no charted pressure. Phase 4 already proxies
  pressure as (sack OR qb_hit) over dropbacks; this module reuses that exact
  proxy rather than inventing a competing one. Consequence, stated loudly because
  it matters for Phase 15: this will NOT reproduce a charted pressure rate from
  PFF/SIS, and blitz rate is not computable from this data at all.
- Time-weighted margin integrates the score margin against the GAME CLOCK, not
  against plays -- see time_weighted_margin() below.
- Points per drive scores a drive by its RESULT (touchdown 7, field goal 3,
  opponent touchdown -7, safety -2), not by the actual extra-point outcome. This
  is the standard drive-efficiency convention and keeps the metric about the
  offense's work rather than its kicker's, but it means a team's points-per-drive
  times drives will not exactly equal its points scored.

TWO ERA DETAILS the 1999-2025 window forces, both already solved in Phase 5:

- Relocations: schedules.parquet keeps era-contemporary team codes (STL/SD/OAK)
  while pbp has been standardized to current codes. RELOCATION_MAP is reused from
  phase5_anomaly_detection so a franchise stays one entity across the window.
- Expansion: 31 teams in 1999-2001, 32 from 2002 (Houston). The regular-season
  row count is therefore 3*31 + 24*32 = 861, not 32*27.

WIN PROBABILITY uses nflverse's `home_wp`/`away_wp` (its own model) rather than
`vegas_wp`. Both are populated across the whole window, but vegas_wp folds the
pregame betting line into the estimate, which would let a team's *reputation*
inflate a "share of game spent in control" measure. The model-only version keeps
the metric a description of what happened on the field.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

SEASONS = list(range(1999, 2026))
FOCUS_TEAM = "SEA"
FOCUS_SEASON = 2025

# Kept in sync with phase5_anomaly_detection.RELOCATION_MAP.
RELOCATION_MAP = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

EXPLOSIVE_PASS_YARDS = 15
EXPLOSIVE_RUSH_YARDS = 10
CONTROL_WP_THRESHOLD = 0.75

PBP_COLS = [
    "season",
    "week",
    "season_type",
    "game_id",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "down",
    "play_type",
    "epa",
    "wp",
    "home_wp",
    "away_wp",
    "success",
    "yards_gained",
    "rush_attempt",
    "sack",
    "qb_hit",
    "qb_dropback",
    "interception",
    "fumble_lost",
    "fixed_drive",
    "fixed_drive_result",
    "drive_play_count",
    "game_seconds_remaining",
    "total_home_score",
    "total_away_score",
    "rusher_player_id",
    "rushing_yards",
    # Player-level columns (Phases 17-18). `receiver_id` rather than
    # `receiver_player_id`: see RECEIVER_ID_NOTE.
    "receiver_id",
    "receiver",
    "passer_player_id",
    "passer_player_name",
    "pass_attempt",
    "two_point_attempt",
    "complete_pass",
    "receiving_yards",
    "pass_touchdown",
    "rush_touchdown",
    "air_yards",
    "yards_after_catch",
    "cp",
    "xyac_mean_yardage",
    "sack_player_id",
    "sack_player_name",
    "half_sack_1_player_id",
    "half_sack_1_player_name",
    "half_sack_2_player_id",
    "half_sack_2_player_name",
    "interception_player_id",
    "interception_player_name",
    "fumbled_1_player_id",
    "solo_tackle_1_player_id",
    "solo_tackle_1_player_name",
    "rusher_player_name",
    "pass_defense_1_player_id",
    "pass_defense_1_player_name",
    "pass_defense_2_player_id",
    "pass_defense_2_player_name",
    "yardline_100",
    "run_location",
    "run_gap",
]

RECEIVER_ID_NOTE = (
    "nflverse ships two receiver keys and they are not interchangeable. "
    "`receiver_player_id` is null on INCOMPLETE passes for 2003-2008 -- 0.7% "
    "populated there against 80%+ for `receiver_id` -- so aggregating targets on "
    "it silently counts only catches for six seasons and inflates every catch "
    "rate in the window. This module uses `receiver_id` everywhere and asserts "
    "the incompletion coverage, because the failure is silent rather than loud."
)

# Air yards, YAC, completion probability and expected YAC are all zero-coverage
# before this season, so anything derived from them is a shorter analysis than
# the 1999 baseline and has to say so.
AIR_YARDS_FIRST_SEASON = 2006

SCORING_DRIVE_RESULTS = {"Touchdown", "Field goal"}
DRIVE_POINTS = {"Touchdown": 7, "Field goal": 3, "Opp touchdown": -7, "Safety": -2}


# --------------------------------------------------------------------------
# Pure kernels -- no I/O, unit-tested in tests/test_season_metrics.py
# --------------------------------------------------------------------------


def time_weighted_margin(seconds_remaining, margins) -> float:
    """Clock-time-weighted average score margin across one game.

    The plain final margin says a 24-17 win and a 24-17 win are the same game,
    even when one team led 24-0 at halftime and the other scored 24 unanswered in
    the fourth quarter. Weighting the margin by how long it was held separates
    them: sum(margin_i * duration_i) / total_duration.

    `margins` is the score margin in effect AFTER each play, from one team's
    perspective; `seconds_remaining` is the game clock at that play. Each play's
    margin is credited with the clock time until the next play -- a right-
    continuous step function over the game.

    Overtime contributes zero weight: nflverse reports game_seconds_remaining as 0
    throughout OT, so OT plays get zero duration. This makes the metric
    "regulation clock-weighted," which is the honest description of it.
    """
    sr = np.asarray(seconds_remaining, dtype=float)
    m = np.asarray(margins, dtype=float)
    if sr.size == 0:
        return float("nan")

    order = np.argsort(-sr, kind="stable")  # kickoff -> final whistle
    sr, m = sr[order], m[order]

    durations = np.clip(sr - np.append(sr[1:], 0.0), 0.0, None)
    total = durations.sum()
    if total <= 0:
        return float("nan")
    return float((m * durations).sum() / total)


def time_weighted_share(seconds_remaining, mask) -> float:
    """Share of game clock (0-1) during which `mask` held, same weighting as above."""
    sr = np.asarray(seconds_remaining, dtype=float)
    flag = np.asarray(mask, dtype=float)
    if sr.size == 0:
        return float("nan")

    order = np.argsort(-sr, kind="stable")
    sr, flag = sr[order], flag[order]

    durations = np.clip(sr - np.append(sr[1:], 0.0), 0.0, None)
    total = durations.sum()
    if total <= 0:
        return float("nan")
    return float((flag * durations).sum() / total)


def percentile_rank(values, x) -> float:
    """Percentile of `x` within `values`, 0-100, counting ties as half.

    Half-tie handling matters here because several of these metrics are integer-
    valued (wins, point differential), where a naive "fraction strictly below"
    understates a tied-for-best season and "fraction at or below" overstates it.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0 or np.isnan(x):
        return float("nan")
    below = np.sum(v < x)
    ties = np.sum(v == x)
    return float(100.0 * (below + 0.5 * ties) / v.size)


def top_share(values) -> float:
    """Share of a total held by its single largest contributor.

    Used to describe how concentrated a team's pass rush is. A team whose sacks
    all come from one edge rusher scores near 1.0; a team that spreads them over
    a rotation scores near 1/n.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    total = v.sum()
    if v.size == 0 or total <= 0:
        return float("nan")
    return float(v.max() / total)


def herfindahl(values) -> float:
    """Herfindahl concentration index of `values`, on 0-1.

    Complements `top_share`: top_share only sees the leader, while this sees the
    whole distribution, so a team with two co-leaders and a team with one leader
    plus a long tail can share a top_share but separate here.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    total = v.sum()
    if v.size == 0 or total <= 0:
        return float("nan")
    shares = v / total
    return float(np.sum(shares**2))


def rolling_window_best(series, window: int, mode: str = "min") -> dict:
    """Best rolling `window`-length mean in `series`.

    Used by Phase 15 to find a defense's strongest sustained stretch. `mode` is
    "min" for metrics where lower is better (EPA/play allowed) and "max"
    otherwise. Returns the value and the 0-indexed start position so the caller
    can report *which* stretch it was.
    """
    s = np.asarray(series, dtype=float)
    if s.size < window or window < 1:
        return {"value": float("nan"), "start_index": None, "window": window}

    means = np.convolve(s, np.ones(window) / window, mode="valid")
    idx = int(np.argmin(means)) if mode == "min" else int(np.argmax(means))
    return {"value": float(means[idx]), "start_index": idx, "window": window}


# --------------------------------------------------------------------------
# Play-by-play aggregation
# --------------------------------------------------------------------------


def load_pbp(season: int) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "pbp" / f"{season}.parquet", columns=PBP_COLS)


def _core_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Phase 4's core-play filter, verbatim, so EPA here matches Phases 4/5/7."""
    return pbp[
        pbp["down"].notna()
        & pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
        & pbp["wp"].between(0.05, 0.95)
    ]


def _two_sided(
    df: pd.DataFrame, keys: list[str], off_names: dict, def_names: dict
) -> pd.DataFrame:
    """Aggregate once from the offense's side and once from the defense's side.

    Every metric in this module has an offensive and a defensive reading of the
    same plays; doing it in one helper keeps the two from drifting apart.
    """
    off = df.groupby(keys + ["posteam"]).agg(**off_names).reset_index()
    off = off.rename(columns={"posteam": "team"})
    dfn = df.groupby(keys + ["defteam"]).agg(**def_names).reset_index()
    dfn = dfn.rename(columns={"defteam": "team"})
    return off.merge(dfn, on=keys + ["team"], how="outer")


def efficiency_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "season_type"]
    core = _core_plays(pbp)

    out = _two_sided(
        core,
        keys,
        {
            "off_epa_per_play": ("epa", "mean"),
            "off_success_rate": ("success", "mean"),
            "off_core_plays": ("epa", "size"),
        },
        {
            "def_epa_per_play_allowed": ("epa", "mean"),
            "def_success_rate_allowed": ("success", "mean"),
            "def_core_plays": ("epa", "size"),
        },
    )

    early = core[core["down"].isin([1, 2])]
    early_out = _two_sided(
        early,
        keys,
        {"off_early_down_epa": ("epa", "mean")},
        {"def_early_down_epa_allowed": ("epa", "mean")},
    )
    return out.merge(early_out, on=keys + ["team"], how="outer")


def explosive_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Explosive rate, deliberately unfiltered for game state (see module docstring)."""
    plays = pbp[
        pbp["down"].notna()
        & pbp["play_type"].isin(["pass", "run"])
        & pbp["yards_gained"].notna()
    ].copy()

    threshold = np.where(
        plays["rush_attempt"] == 1, EXPLOSIVE_RUSH_YARDS, EXPLOSIVE_PASS_YARDS
    )
    plays["is_explosive"] = (plays["yards_gained"] >= threshold).astype(float)

    return _two_sided(
        plays,
        ["season", "season_type"],
        {"off_explosive_rate": ("is_explosive", "mean")},
        {"def_explosive_rate_allowed": ("is_explosive", "mean")},
    )


def pressure_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Proxy pressure (sack OR qb_hit) over dropbacks -- NOT charted pressure."""
    db = pbp[(pbp["qb_dropback"] == 1) & pbp["down"].notna()].copy()
    db["is_pressured"] = ((db["sack"] == 1) | (db["qb_hit"] == 1)).astype(float)

    return _two_sided(
        db,
        ["season", "season_type"],
        {"pressure_rate_allowed": ("is_pressured", "mean"), "dropbacks": ("is_pressured", "size")},
        {"pressure_rate_created": ("is_pressured", "mean"), "opp_dropbacks": ("is_pressured", "size")},
    )


def turnover_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    to = pbp[(pbp["interception"] == 1) | (pbp["fumble_lost"] == 1)]
    out = _two_sided(
        to,
        ["season", "season_type"],
        {"giveaways": ("interception", "size")},
        {"takeaways": ("interception", "size")},
    )
    out[["giveaways", "takeaways"]] = out[["giveaways", "takeaways"]].fillna(0)
    out["turnover_margin"] = out["takeaways"] - out["giveaways"]
    return out


def run_defense_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    rushes = pbp[(pbp["rush_attempt"] == 1) & pbp["yards_gained"].notna()]
    ypc = (
        rushes.groupby(["season", "season_type", "defteam"])["yards_gained"]
        .mean()
        .reset_index()
        .rename(columns={"defteam": "team", "yards_gained": "def_yards_per_carry_allowed"})
    )

    # A "100-yard rusher allowed" is per individual back per game, which is why
    # this groups by rusher and game rather than by team and game.
    by_rusher = (
        rushes[rushes["rusher_player_id"].notna()]
        .groupby(["season", "season_type", "defteam", "game_id", "rusher_player_id"])["rushing_yards"]
        .sum()
        .reset_index()
    )
    hundred = (
        by_rusher[by_rusher["rushing_yards"] >= 100]
        .groupby(["season", "season_type", "defteam"])
        .size()
        .reset_index(name="opp_100yd_rushers")
        .rename(columns={"defteam": "team"})
    )

    out = ypc.merge(hundred, on=["season", "season_type", "team"], how="left")
    out["opp_100yd_rushers"] = out["opp_100yd_rushers"].fillna(0)
    return out


def drive_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Drive-level efficiency.

    A drive must contain a real scrimmage snap to count, matching Phase 5's
    reasoning -- otherwise kickoff-only and end-of-half fragments pad the
    denominator and deflate every rate.
    """
    snaps = pbp[pbp["down"].notna()].copy()
    # The punt that ENDS a three-and-out is itself a 4th-down snap, so counting
    # every snap makes a genuine three-and-out look like a four-play drive and
    # the rate collapses to zero. Count only the plays run to try to move the
    # chains.
    snaps["is_offensive_snap"] = (~snaps["play_type"].isin(["punt", "field_goal"])).astype(int)

    real_drives = snaps.groupby(["season", "season_type", "game_id", "posteam", "fixed_drive"]).agg(
        defteam=("defteam", "first"),
        result=("fixed_drive_result", "first"),
        plays=("is_offensive_snap", "sum"),
    ).reset_index()

    real_drives["points"] = real_drives["result"].map(DRIVE_POINTS).fillna(0.0)
    real_drives["is_scoring"] = real_drives["result"].isin(SCORING_DRIVE_RESULTS).astype(float)
    real_drives["is_three_and_out"] = (
        (real_drives["plays"] <= 3) & (real_drives["result"] == "Punt")
    ).astype(float)

    return _two_sided(
        real_drives,
        ["season", "season_type"],
        {
            "points_per_drive": ("points", "mean"),
            "scoring_drive_pct": ("is_scoring", "mean"),
            "three_and_out_rate": ("is_three_and_out", "mean"),
            "drives": ("points", "size"),
        },
        {
            "points_per_drive_allowed": ("points", "mean"),
            "scoring_drive_pct_allowed": ("is_scoring", "mean"),
            "three_and_out_rate_forced": ("is_three_and_out", "mean"),
            "drives_faced": ("points", "size"),
        },
    )


# --------------------------------------------------------------------------
# Game-level control (Phase 14's foundation)
# --------------------------------------------------------------------------


def game_control_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per game, per team: time-weighted margin, time leading, time in control."""
    df = pbp[pbp["game_seconds_remaining"].notna()].copy()
    df["home_margin"] = df["total_home_score"] - df["total_away_score"]

    rows = []
    for side, team_col, sign, wp_col in [
        ("home", "home_team", 1, "home_wp"),
        ("away", "away_team", -1, "away_wp"),
    ]:
        part = df[["season", "season_type", "week", "game_id", team_col, "game_seconds_remaining", "home_margin", wp_col]].copy()
        part = part.rename(columns={team_col: "team", wp_col: "wp_team"})
        part["margin"] = sign * part["home_margin"]
        part["side"] = side
        rows.append(part.drop(columns=["home_margin"]))

    stacked = pd.concat(rows, ignore_index=True)

    out = []
    for (season, season_type, week, game_id, team), g in stacked.groupby(
        ["season", "season_type", "week", "game_id", "team"], sort=False
    ):
        sr = g["game_seconds_remaining"].to_numpy()
        margin = g["margin"].to_numpy()
        wp = g["wp_team"].to_numpy()
        out.append(
            {
                "season": season,
                "season_type": season_type,
                "week": week,
                "game_id": game_id,
                "team": team,
                "time_weighted_margin": time_weighted_margin(sr, margin),
                "pct_game_time_leading": 100 * time_weighted_share(sr, margin > 0),
                "pct_game_time_wp_gt_75": 100 * time_weighted_share(sr, wp > CONTROL_WP_THRESHOLD),
                "never_trailed": float(np.nanmin(margin) >= 0),
                "final_margin": float(margin[np.argmin(sr)]),
            }
        )
    return pd.DataFrame(out)


def control_metrics(game_control: pd.DataFrame) -> pd.DataFrame:
    return (
        game_control.groupby(["season", "season_type", "team"])
        .agg(
            time_weighted_margin=("time_weighted_margin", "mean"),
            pct_game_time_leading=("pct_game_time_leading", "mean"),
            pct_game_time_wp_gt_75=("pct_game_time_wp_gt_75", "mean"),
            games_never_trailed=("never_trailed", "sum"),
        )
        .reset_index()
    )


# --------------------------------------------------------------------------
# Schedules-derived record and scoring
# --------------------------------------------------------------------------


def scoring_frame() -> pd.DataFrame:
    sched = pd.read_parquet(RAW_DIR / "schedules.parquet")
    sched = sched[sched["season"].isin(SEASONS)].dropna(subset=["home_score", "away_score"])
    sched = sched.copy()
    sched["season_type"] = np.where(sched["game_type"] == "REG", "REG", "POST")

    def half(team_col: str, own: str, opp: str) -> pd.DataFrame:
        out = sched[["season", "season_type", team_col, own, opp]].rename(
            columns={team_col: "team", own: "points_for", opp: "points_against"}
        )
        out["win"] = np.select(
            [out["points_for"] > out["points_against"], out["points_for"] == out["points_against"]],
            [1.0, 0.5],
            default=0.0,
        )
        return out

    stacked = pd.concat(
        [half("home_team", "home_score", "away_score"), half("away_team", "away_score", "home_score")],
        ignore_index=True,
    )
    stacked["team"] = stacked["team"].replace(RELOCATION_MAP)

    out = (
        stacked.groupby(["season", "season_type", "team"])
        .agg(
            points_for=("points_for", "sum"),
            points_against=("points_against", "sum"),
            wins=("win", "sum"),
            games=("win", "size"),
        )
        .reset_index()
    )
    out["losses"] = out["games"] - out["wins"]
    out["point_diff"] = out["points_for"] - out["points_against"]
    out["point_diff_per_game"] = out["point_diff"] / out["games"]
    out["points_for_per_game"] = out["points_for"] / out["games"]
    out["points_against_per_game"] = out["points_against"] / out["games"]
    return out


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def game_efficiency_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-game offensive and defensive EPA/play, for rolling-window analysis.

    Phase 15 needs to find a defense's best sustained STRETCH, which a
    season-level table cannot express. Both a garbage-time-filtered and an
    unfiltered version are emitted: the filtered one is comparable to the rest of
    this project, while published "best N-week defensive stretch" claims are
    usually computed unfiltered, and testing such a claim against a differently
    filtered number would not be a fair test of it.
    """
    keys = ["season", "season_type", "week", "game_id"]

    raw = pbp[
        pbp["down"].notna() & pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()
    ]
    core = _core_plays(pbp)

    unfiltered = _two_sided(
        raw,
        keys,
        {"off_epa_per_play_unfiltered": ("epa", "mean")},
        {"def_epa_per_play_allowed_unfiltered": ("epa", "mean"), "def_plays_unfiltered": ("epa", "size")},
    )
    filtered = _two_sided(
        core,
        keys,
        {"off_epa_per_play": ("epa", "mean"), "off_success_rate": ("success", "mean")},
        {"def_epa_per_play_allowed": ("epa", "mean"), "def_success_rate_allowed": ("success", "mean")},
    )
    out = unfiltered.merge(filtered, on=keys + ["team"], how="outer")

    # The individual-100-yard-rusher streak is a per-game fact about the leading
    # opposing back, which no season-level column can reconstruct afterwards.
    runs = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()]
    per_rusher = (
        runs.groupby(keys + ["defteam", "rusher_player_id"])["rushing_yards"].sum().reset_index()
    )
    opp_best = (
        per_rusher.groupby(keys + ["defteam"])["rushing_yards"]
        .max()
        .reset_index(name="opp_leading_rusher_yards")
        .rename(columns={"defteam": "team"})
    )
    return out.merge(opp_best, on=keys + ["team"], how="left")


def focus_wp_curve(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-play win-probability and score timeline for one team's games.

    Emitted so the dashboard can draw a real win-probability curve per game
    without loading play-by-play at runtime. Only the focus team's season is
    kept, which is what makes the file small enough to commit.
    """
    df = pbp[
        pbp["game_seconds_remaining"].notna()
        & ((pbp["home_team"] == FOCUS_TEAM) | (pbp["away_team"] == FOCUS_TEAM))
    ].copy()
    if df.empty:
        return pd.DataFrame()

    is_home = df["home_team"] == FOCUS_TEAM
    df["wp_sea"] = np.where(is_home, df["home_wp"], df["away_wp"])
    df["margin_sea"] = np.where(is_home, 1, -1) * (df["total_home_score"] - df["total_away_score"])
    df["opponent"] = np.where(is_home, df["away_team"], df["home_team"])
    df["at_home"] = is_home
    df["seconds_elapsed"] = 3600 - df["game_seconds_remaining"]

    cols = [
        "season", "season_type", "week", "game_id", "opponent", "at_home",
        "seconds_elapsed", "game_seconds_remaining", "wp_sea", "margin_sea",
    ]
    return df[cols].sort_values(["game_id", "seconds_elapsed"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Player-level frames (Phases 17-18)
# --------------------------------------------------------------------------


def receiver_season_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per receiver-season, plus the team denominator each share needs.

    A "target" is a pass attempt carrying an identified intended receiver.

    THE DENOMINATOR IS THE WHOLE BALLGAME for a target-share number, and nflverse's
    `pass_attempt` flag is not the official one: it also fires on sacks and on
    two-point conversion passes. Left alone it gives SEA 2025 510 attempts against
    the league's official 481, which would quietly deflate every share in this
    file. Removing both puts it on exactly 481 and puts Smith-Njigba's share on
    33.9%, reproducing the published figure rather than inventing a third one --
    three different denominators are already circulating in public coverage.
    """
    att = pbp[
        (pbp["pass_attempt"] == 1)
        & (pbp["sack"] != 1)
        & (pbp["two_point_attempt"] != 1)
        & pbp["posteam"].notna()
    ].copy()

    # See RECEIVER_ID_NOTE. Fails loudly if the wrong key is ever swapped back in.
    inc = att[att["complete_pass"] != 1]
    if len(inc):
        coverage = float(inc["receiver_id"].notna().mean())
        assert coverage > 0.5, (
            f"receiver_id covers only {coverage:.1%} of incompletions in "
            f"{int(att['season'].iloc[0])} -- wrong receiver key? {RECEIVER_ID_NOTE}"
        )

    team_totals = (
        att.groupby(["season", "season_type", "posteam"])
        .agg(team_pass_attempts=("pass_attempt", "size"),
             team_receiving_yards=("receiving_yards", "sum"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    tgt = att[att["receiver_id"].notna()].copy()
    tgt["expected_yards"] = tgt["cp"] * (
        tgt["air_yards"] + tgt["xyac_mean_yardage"].fillna(0.0)
    )

    per = (
        tgt.groupby(["season", "season_type", "posteam", "receiver_id", "receiver"])
        .agg(
            targets=("pass_attempt", "size"),
            receptions=("complete_pass", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            receiving_tds=("pass_touchdown", "sum"),
            air_yards=("air_yards", "sum"),
            yards_after_catch=("yards_after_catch", "sum"),
            expected_yards=("expected_yards", "sum"),
            games=("game_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    per = per.merge(team_totals, on=["season", "season_type", "team"], how="left")
    per["target_share"] = per["targets"] / per["team_pass_attempts"]
    per["yards_per_team_pass_attempt"] = per["receiving_yards"] / per["team_pass_attempts"]
    per["team_receiving_yards_share"] = per["receiving_yards"] / per["team_receiving_yards"]
    per["yards_per_target"] = per["receiving_yards"] / per["targets"]
    per["yards_over_expected"] = per["receiving_yards"] - per["expected_yards"]
    return per


def passer_season_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per passer-season, keyed on the dropback definition Phase 6 uses.

    Dropbacks are scramble-inclusive (the passer on a dropback, or the rusher when
    the dropback became a scramble), and lost fumbles are keyed on the FUMBLING
    player rather than the offense, so a running back's lost fumble is not charged
    to the quarterback. Both choices are Phase 6's.

    TURNOVERS ARE COUNTED OVER EVERY SNAP THE PLAYER TOUCHED, NOT ONLY DROPBACKS.
    Phase 6 found that one of Darnold's 20 giveaways in 2025 -- a week-10 aborted
    snap -- is coded as a run and therefore sits outside the dropback set.
    Restricting the numerator to dropbacks returns 19 and silently disagrees with
    every published total. The rate keeps dropbacks as its denominator (that is
    the exposure being modelled), so the rate is very slightly conservative by
    construction, which is the direction to err in.
    """
    off = pbp[pbp["posteam"].notna()].copy()
    off["player_id"] = off["passer_player_id"].fillna(off["rusher_player_id"])
    off["player_name"] = off["passer_player_name"].fillna(off["rusher_player_name"])
    off = off[off["player_id"].notna()]
    off["lost_own_fumble"] = (
        (off["fumble_lost"] == 1) & (off["fumbled_1_player_id"] == off["player_id"])
    ).astype(int)
    off["is_dropback"] = (off["qb_dropback"] == 1).astype(int)

    keys = ["season", "season_type", "posteam", "player_id", "player_name"]
    per = (
        off.groupby(keys)
        .agg(
            dropbacks=("is_dropback", "sum"),
            interceptions=("interception", "sum"),
            fumbles_lost=("lost_own_fumble", "sum"),
            sacks_taken=("sack", "sum"),
            games=("game_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    epa = (
        off[off["is_dropback"] == 1]
        .groupby(keys)["epa"]
        .mean()
        .reset_index(name="epa_per_dropback")
        .rename(columns={"posteam": "team"})
    )
    per = per.merge(epa, on=["season", "season_type", "team", "player_id", "player_name"], how="left")
    per = per[per["dropbacks"] > 0]
    per["turnovers"] = per["interceptions"] + per["fumbles_lost"]
    per["turnover_rate_per_dropback"] = per["turnovers"] / per["dropbacks"]
    return per


def defender_season_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per defender-season for the events the pbp actually attributes.

    Sacks are counted as full plus half credits, because `sack_player_id` alone
    misses roughly a tenth of sacks -- the shared ones -- and a pass rush measured
    without them understates exactly the rotational defenses this is meant to
    describe. Tackles, interceptions and forced fumbles are attributed reliably
    from 1999; QB hits are NOT, which is why no pressure statistic appears here
    (see phase15_defense.QB_HIT_FIRST_SEASON).
    """
    keys = ["season", "season_type", "defteam"]
    credits = []
    for id_col, name_col, weight in [
        ("sack_player_id", "sack_player_name", 1.0),
        ("half_sack_1_player_id", "half_sack_1_player_name", 0.5),
        ("half_sack_2_player_id", "half_sack_2_player_name", 0.5),
    ]:
        part = pbp[pbp[id_col].notna() & pbp["defteam"].notna()][keys + [id_col, name_col]]
        part = part.rename(columns={id_col: "player_id", name_col: "player_name"})
        part["sacks"] = weight
        credits.append(part)
    sacks = pd.concat(credits, ignore_index=True)
    sacks = (
        sacks.groupby(keys + ["player_id", "player_name"])["sacks"].sum().reset_index()
    )

    def _event(id_col: str, name_col: str, out: str) -> pd.DataFrame:
        part = pbp[pbp[id_col].notna() & pbp["defteam"].notna()][keys + [id_col, name_col]]
        part = part.rename(columns={id_col: "player_id", name_col: "player_name"})
        return part.groupby(keys + ["player_id", "player_name"]).size().reset_index(name=out)

    ints = _event("interception_player_id", "interception_player_name", "interceptions")
    solo = _event("solo_tackle_1_player_id", "solo_tackle_1_player_name", "solo_tackles")

    # Passes defensed measures ball disruption -- a defender physically getting a
    # hand to the throw. It is NOT coverage volume: it fires on a stable ~30% of
    # incompletions in every season from 1999, so it compares cleanly across eras,
    # but it says nothing about how often a defender was targeted.
    pds = []
    for id_col, name_col in [
        ("pass_defense_1_player_id", "pass_defense_1_player_name"),
        ("pass_defense_2_player_id", "pass_defense_2_player_name"),
    ]:
        part = pbp[pbp[id_col].notna() & pbp["defteam"].notna()][keys + [id_col, name_col]]
        pds.append(part.rename(columns={id_col: "player_id", name_col: "player_name"}))
    passes_defensed = (
        pd.concat(pds, ignore_index=True)
        .groupby(keys + ["player_id", "player_name"])
        .size()
        .reset_index(name="passes_defensed")
    )

    per = sacks.merge(ints, on=keys + ["player_id", "player_name"], how="outer")
    per = per.merge(solo, on=keys + ["player_id", "player_name"], how="outer")
    per = per.merge(passes_defensed, on=keys + ["player_id", "player_name"], how="outer")
    per = per.rename(columns={"defteam": "team"})
    for col in ("sacks", "interceptions", "solo_tackles", "passes_defensed"):
        per[col] = per[col].fillna(0)
    # Everyone who ever made a tackle would triple the committed file for no
    # analytical gain; the pass-rush and takeaway story needs the contributors.
    return per[(per["sacks"] > 0) | (per["interceptions"] > 0) | (per["passes_defensed"] > 0)]


def rushing_direction_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-season rushing split by direction, both run and defended.

    `run_location` is populated on ~96% of carries in every season since 1999,
    which makes it the one blocking-adjacent signal in this cache that survives a
    27-season comparison. It describes where a team's run game worked, not who
    blocked it -- the play-by-play names no blockers, so this cannot separate the
    line from the back and is not a line grade.
    """
    run = pbp[(pbp["rush_attempt"] == 1) & pbp["run_location"].notna()].copy()
    run["stuffed"] = (run["yards_gained"] <= 0).astype(int)
    keys = ["season", "season_type"]

    out = _two_sided(
        run,
        keys,
        {
            "off_rush_yards_per_carry": ("rushing_yards", "mean"),
            "off_rush_epa": ("epa", "mean"),
            "off_stuffed_rate": ("stuffed", "mean"),
            "off_carries": ("rush_attempt", "size"),
        },
        {
            "def_stuffed_rate_forced": ("stuffed", "mean"),
            "def_rush_epa_allowed": ("epa", "mean"),
        },
    )

    parts = [out]
    for loc in ("left", "middle", "right"):
        side = run[run["run_location"] == loc]
        parts.append(
            _two_sided(
                side,
                keys,
                {
                    f"off_rush_ypc_{loc}": ("rushing_yards", "mean"),
                    f"off_rush_share_{loc}": ("rush_attempt", "size"),
                },
                {f"def_rush_ypc_allowed_{loc}": ("rushing_yards", "mean")},
            )
        )
    frame = parts[0]
    for part in parts[1:]:
        frame = frame.merge(part, on=keys + ["team"], how="outer")

    for loc in ("left", "middle", "right"):
        frame[f"off_rush_share_{loc}"] = frame[f"off_rush_share_{loc}"] / frame["off_carries"]
    return frame


def rusher_season_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per rusher-season, including where on the field the carries came.

    Goal-line usage is carried explicitly because touchdown totals are mostly a
    story about opportunity: a back who gets the ball inside the five will outscore
    a better back who does not. Separating the two is the whole point of looking.

    Games started are NOT in the play-by-play, so any claim about starts is
    external context rather than something computed here.
    """
    run = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()].copy()
    run["inside_10"] = (run["yardline_100"] <= 10).astype(int)
    run["inside_5"] = (run["yardline_100"] <= 5).astype(int)
    run["td_inside_5"] = ((run["yardline_100"] <= 5) & (run["rush_touchdown"] == 1)).astype(int)
    run["stuffed"] = (run["yards_gained"] <= 0).astype(int)

    keys = ["season", "season_type", "posteam", "rusher_player_id", "rusher_player_name"]
    per = (
        run.groupby(keys)
        .agg(
            carries=("rush_attempt", "size"),
            rushing_yards=("rushing_yards", "sum"),
            rushing_tds=("rush_touchdown", "sum"),
            epa_per_rush=("epa", "mean"),
            games=("game_id", "nunique"),
            carries_inside_10=("inside_10", "sum"),
            carries_inside_5=("inside_5", "sum"),
            tds_inside_5=("td_inside_5", "sum"),
            stuffed_runs=("stuffed", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team", "rusher_player_id": "player_id",
                         "rusher_player_name": "player_name"})
    )
    team_totals = (
        run.groupby(["season", "season_type", "posteam"])
        .agg(team_carries=("rush_attempt", "size"),
             team_carries_inside_5=("inside_5", "sum"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    per = per.merge(team_totals, on=["season", "season_type", "team"], how="left")
    per["yards_per_carry"] = per["rushing_yards"] / per["carries"]
    per["tds_per_carry"] = per["rushing_tds"] / per["carries"]
    per["carry_share"] = per["carries"] / per["team_carries"]
    per["goal_line_carry_share"] = per["carries_inside_5"] / per["team_carries_inside_5"]
    per["stuffed_rate"] = per["stuffed_runs"] / per["carries"]
    return per


def build_season(season: int) -> dict[str, pd.DataFrame]:
    pbp = load_pbp(season)
    pbp = pbp[pbp["season_type"].isin(["REG", "POST"])]
    for col in ("posteam", "defteam", "home_team", "away_team"):
        # A handful of 2000 plays carry an empty-string team code rather than a
        # null, which notna() happily lets through and which then surfaces as a
        # phantom 32nd team in a 31-team season.
        pbp[col] = pbp[col].replace(RELOCATION_MAP).replace("", pd.NA)

    gc = game_control_frame(pbp)
    if season == FOCUS_SEASON:
        focus_wp_curve(pbp).to_csv(PROCESSED_DIR / "focus_wp_curve.csv", index=False)

    frame = efficiency_metrics(pbp)
    for build in (
        explosive_metrics,
        pressure_metrics,
        turnover_metrics,
        run_defense_metrics,
        drive_metrics,
        rushing_direction_frame,
    ):
        frame = frame.merge(build(pbp), on=["season", "season_type", "team"], how="outer")
    frame = frame.merge(control_metrics(gc), on=["season", "season_type", "team"], how="outer")

    return {
        "advanced": frame,
        "control": gc,
        "game_efficiency": game_efficiency_frame(pbp),
        "receiver": receiver_season_frame(pbp),
        "passer": passer_season_frame(pbp),
        "defender": defender_season_frame(pbp),
        "rusher": rusher_season_frame(pbp),
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parts: dict[str, list[pd.DataFrame]] = {}
    for season in SEASONS:
        built = build_season(season)
        for key, df in built.items():
            parts.setdefault(key, []).append(df)
        reg = built["advanced"][built["advanced"]["season_type"] == "REG"]
        print(f"{season}: {len(reg)} teams (REG), {len(built['advanced'])} team-season-phase rows")

    frames, controls, game_effs = parts["advanced"], parts["control"], parts["game_efficiency"]
    advanced = pd.concat(frames, ignore_index=True)
    advanced = advanced[advanced["team"].notna()].sort_values(["season", "season_type", "team"])
    advanced = advanced.merge(scoring_frame(), on=["season", "season_type", "team"], how="left")

    n_reg = int((advanced["season_type"] == "REG").sum())
    expected_reg = 3 * 31 + 24 * 32  # 31 teams 1999-2001, 32 from 2002 (Houston)
    assert n_reg == expected_reg, f"expected {expected_reg} REG team-seasons, got {n_reg}"
    assert advanced["points_for"].notna().all(), "some team-seasons failed to join to schedules"

    adv_path = PROCESSED_DIR / "team_season_advanced.csv"
    advanced.to_csv(adv_path, index=False)
    print(f"\nteam_season_advanced: {advanced.shape} -> {adv_path}")

    game_control = pd.concat(controls, ignore_index=True)
    sea = game_control[
        (game_control["team"] == FOCUS_TEAM) & (game_control["season"] == FOCUS_SEASON)
    ].sort_values(["season_type", "week"], ascending=[False, True])
    gc_path = PROCESSED_DIR / "team_game_control.csv"
    sea.to_csv(gc_path, index=False)
    print(f"team_game_control ({FOCUS_TEAM} {FOCUS_SEASON}): {sea.shape} -> {gc_path}")

    game_eff = pd.concat(game_effs, ignore_index=True)
    game_eff = game_eff[game_eff["team"].notna()].sort_values(["season", "season_type", "team", "week"])
    # 14.5k rows of full float64 repr is ~2MB of committed text for no analytical
    # gain -- EPA/play is never read past the third decimal.
    game_eff = game_eff.round(5)
    ge_path = PROCESSED_DIR / "team_game_efficiency.csv"
    game_eff.to_csv(ge_path, index=False)
    print(f"team_game_efficiency: {game_eff.shape} -> {ge_path}")

    curve = pd.read_csv(PROCESSED_DIR / "focus_wp_curve.csv")
    print(f"focus_wp_curve ({FOCUS_TEAM} {FOCUS_SEASON}): {curve.shape} -> {PROCESSED_DIR / 'focus_wp_curve.csv'}")

    for key, name, sort_keys in [
        ("receiver", "receiver_season.csv", ["season", "season_type", "team", "receiver_id"]),
        ("passer", "passer_season.csv", ["season", "season_type", "team", "player_id"]),
        ("defender", "defender_season.csv", ["season", "season_type", "team", "player_id"]),
        ("rusher", "rusher_season.csv", ["season", "season_type", "team", "player_id"]),
    ]:
        df = pd.concat(parts[key], ignore_index=True)
        df = df[df["team"].notna()].sort_values(sort_keys).round(5)
        path = PROCESSED_DIR / name
        df.to_csv(path, index=False)
        print(f"{name.removesuffix('.csv')}: {df.shape} -> {path}")

    sea_row = advanced[
        (advanced["team"] == FOCUS_TEAM)
        & (advanced["season"] == FOCUS_SEASON)
        & (advanced["season_type"] == "REG")
    ].iloc[0]
    print(
        f"\nSEA 2025 (REG): {sea_row['wins']:.0f}-{sea_row['losses']:.0f}, "
        f"{sea_row['points_for']:.0f} PF / {sea_row['points_against']:.0f} PA "
        f"({sea_row['point_diff']:+.0f}), {sea_row['points_against_per_game']:.1f} PPG allowed"
    )
    print(
        f"  def EPA/play {sea_row['def_epa_per_play_allowed']:+.3f} | "
        f"time-weighted margin {sea_row['time_weighted_margin']:+.2f} | "
        f"{sea_row['pct_game_time_leading']:.1f}% of clock leading"
    )


if __name__ == "__main__":
    main()
