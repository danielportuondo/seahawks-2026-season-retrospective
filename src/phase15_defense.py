"""Phase 15: the defense, measured against 27 seasons instead of adjectives.

Phase 5 established that the 2025 defense was a statistical outlier within
2010-2025. The published case goes further and makes a specific, falsifiable
claim: that Seattle's best eight-week stretch, at -0.34 EPA per play, is the best
eight-week run of defense in twenty-five years -- better than the 2013 Legion of
Boom, whose peak is quoted at -0.23.

That claim is testable, and testing it properly means computing the same rolling
window for EVERY team-season since 1999 and seeing where Seattle's best stretch
actually lands. Computing it for Seattle alone and repeating the "best in 25
years" label would be assuming the conclusion.

FAIRNESS TO THE CLAIM: filtered vs unfiltered. The rest of this project filters
garbage time (win probability between 5% and 95%) so that blowout snaps do not
flatter a team's efficiency. Published EPA-streak numbers are typically computed
WITHOUT that filter. Testing a claim against a differently-filtered statistic
would not be a test of the claim, so this phase computes the rolling window both
ways and reports both. The unfiltered number is the one compared to -0.34; the
filtered number is the one comparable to Phase 5 and the rest of this repo.

That distinction cuts against Seattle here, and is reported anyway: a dominant
defense spends a lot of its snaps in garbage time precisely BECAUSE it is
dominant, so filtering those out tends to make an elite defense look worse, not
better.

WINDOW LENGTH is eight games, matching the claim. Windows do not cross seasons --
a "stretch" that spans an offseason is not a stretch -- and postseason games are
excluded from the rolling analysis so that teams with deep playoff runs do not
get extra windows to draw from.

PRESSURE: the published "40.1% pressure rate on a 19.2% blitz rate" figures come
from charted data (SIS/PFF) that this project does not have. The pbp cache has no
participation or pass-rusher data at all, so blitz rate is NOT COMPUTABLE here,
and pressure is only available as Phase 4's (sack OR qb_hit) proxy, which
measures something narrower than a charted pressure and will read lower. Both
limitations are recorded in the output rather than papered over with a
similar-looking number.

PRESSURE'S REFERENCE WINDOW IS SHORTER THAN EVERY OTHER METRIC'S, and this is not
a stylistic choice. nflverse's qb_hit attribution is not stationary across the
cache: 1999-2002 carry roughly 1,000 qb_hits a season, 2003-2005 carry EXACTLY
ZERO, and 2006 onward carry 2,000-2,900. In the three empty seasons the proxy
silently degrades to a bare sack rate and parks those team-seasons at the bottom
of the distribution; in 1999-2002 it runs at about half its modern level. Ranking
2025 against all 861 team-seasons therefore flattered it by comparing against
years where half the metric did not exist. Pressure alone is ranked from 2006,
and the window is written into the output next to the number so the shorter
denominator travels with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chart_style import (
    ACTION_GREEN,
    ALERT_RED,
    AMBER,
    OFF_WHITE,
    WOLF_GREY,
    apply_scoreboard_style,
    fig_size,
)
from phase13_historical_baseline import rank_in_good_direction
from season_metrics import percentile_rank, rolling_window_best

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

FOCUS_TEAM = "SEA"
FOCUS_SEASON = 2025
COMPARISON_SEASON = 2013  # the Legion of Boom benchmark the claim names
WINDOW = 8

# 2003-2005 carry zero qb_hits and 1999-2002 about half the modern rate, so the
# (sack OR qb_hit) proxy is only comparable from here on.
QB_HIT_FIRST_SEASON = 2006

CLAIMED_SEA_2025_PEAK = -0.34
CLAIMED_SEA_2013_PEAK = -0.23


def load(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} missing -- run src/season_metrics.py first")
    return pd.read_csv(path)


def rolling_best_by_team_season(games: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Best `WINDOW`-game stretch of `metric` for every team-season."""
    rows = []
    for (season, team), g in games.groupby(["season", "team"], sort=False):
        g = g.sort_values("week")
        series = g[metric].to_numpy()
        if np.isnan(series).any() or len(series) < WINDOW:
            continue
        best = rolling_window_best(series, WINDOW, mode="min")
        start = best["start_index"]
        rows.append(
            {
                "season": int(season),
                "team": str(team),
                "best_window_value": best["value"],
                "start_week": int(g["week"].iloc[start]),
                "end_week": int(g["week"].iloc[start + WINDOW - 1]),
                "season_value": float(np.mean(series)),
            }
        )
    return pd.DataFrame(rows)


def streak_claim(best: pd.DataFrame, label: str, claimed_2025: float, claimed_2013: float) -> dict:
    ranked = best.sort_values("best_window_value")
    sea = ranked[(ranked["season"] == FOCUS_SEASON) & (ranked["team"] == FOCUS_TEAM)].iloc[0]
    sea13 = ranked[(ranked["season"] == COMPARISON_SEASON) & (ranked["team"] == FOCUS_TEAM)]

    rank = rank_in_good_direction(best["best_window_value"], sea["best_window_value"], higher_is_better=False)
    ahead = ranked[ranked["best_window_value"] < sea["best_window_value"]]

    return {
        "basis": label,
        "sea_2025_best_window": round(float(sea["best_window_value"]), 4),
        "sea_2025_window_weeks": [int(sea["start_week"]), int(sea["end_week"])],
        "sea_2025_rank_of_n": [rank, len(best)],
        "sea_2025_percentile": round(
            100 - percentile_rank(best["best_window_value"], float(sea["best_window_value"])), 1
        ),
        "sea_2013_best_window": round(float(sea13["best_window_value"].iloc[0]), 4) if len(sea13) else None,
        "beats_2013_seahawks": (
            bool(sea["best_window_value"] < sea13["best_window_value"].iloc[0]) if len(sea13) else None
        ),
        "team_seasons_ahead": [
            {"season": int(r.season), "team": str(r.team), "value": round(float(r.best_window_value), 4)}
            for r in ahead.itertuples()
        ],
        "published_figures": {"sea_2025": claimed_2025, "sea_2013": claimed_2013},
        "top10": [
            {
                "rank": i + 1,
                "season": int(r.season),
                "team": str(r.team),
                "value": round(float(r.best_window_value), 4),
                "weeks": [int(r.start_week), int(r.end_week)],
            }
            for i, r in enumerate(ranked.head(10).itertuples())
        ],
    }


def supporting_metrics(reg: pd.DataFrame) -> dict:
    sea = reg[(reg["team"] == FOCUS_TEAM) & (reg["season"] == FOCUS_SEASON)].iloc[0]
    out = {}
    for metric, label, higher_better, min_season in [
        ("def_epa_per_play_allowed", "Defensive EPA/play allowed", False, None),
        ("def_success_rate_allowed", "Success rate allowed", False, None),
        ("def_explosive_rate_allowed", "Explosive play rate allowed", False, None),
        ("def_yards_per_carry_allowed", "Yards per carry allowed", False, None),
        ("points_per_drive_allowed", "Points per drive allowed", False, None),
        ("scoring_drive_pct_allowed", "Opponent scoring-drive rate", False, None),
        ("three_and_out_rate_forced", "Three-and-outs forced", True, None),
        ("points_against_per_game", "Points allowed per game", False, None),
        ("takeaways", "Takeaways", True, None),
        ("pressure_rate_created", "Pressure rate created (proxy)", True, QB_HIT_FIRST_SEASON),
    ]:
        ref = reg if min_season is None else reg[reg["season"] >= min_season]
        value = float(sea[metric])
        pct = percentile_rank(ref[metric], value)
        if not higher_better:
            pct = 100 - pct
        out[metric] = {
            "label": label,
            "value": round(value, 4),
            "rank_of_n": [
                rank_in_good_direction(ref[metric], value, higher_better),
                int(ref[metric].notna().sum()),
            ],
            "percentile": round(pct, 1),
            "rank_in_2025": rank_in_good_direction(
                reg[reg["season"] == FOCUS_SEASON][metric], value, higher_better
            ),
            "reference_window": [
                int(ref["season"].min()),
                int(ref["season"].max()),
            ],
        }
    return out


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def rolling_chart(games: pd.DataFrame, metric: str, path: Path) -> None:
    """SEA 2025's rolling window against the full 1999-2025 cloud."""
    apply_scoreboard_style()
    fig, ax = plt.subplots(figsize=fig_size(5.8))

    for (season, team), g in games.groupby(["season", "team"], sort=False):
        if season == FOCUS_SEASON and team == FOCUS_TEAM:
            continue
        if season == COMPARISON_SEASON and team == FOCUS_TEAM:
            continue
        g = g.sort_values("week")
        s = g[metric].to_numpy()
        if np.isnan(s).any() or len(s) < WINDOW:
            continue
        means = np.convolve(s, np.ones(WINDOW) / WINDOW, mode="valid")
        ax.plot(np.arange(len(means)) + WINDOW, means, color=WOLF_GREY, lw=0.4, alpha=0.12, zorder=1)

    for season, color, lw, label in [
        (COMPARISON_SEASON, AMBER, 2.2, "SEA 2013 (Legion of Boom)"),
        (FOCUS_SEASON, ACTION_GREEN, 2.8, "SEA 2025"),
    ]:
        g = games[(games["season"] == season) & (games["team"] == FOCUS_TEAM)].sort_values("week")
        s = g[metric].to_numpy()
        means = np.convolve(s, np.ones(WINDOW) / WINDOW, mode="valid")
        ax.plot(np.arange(len(means)) + WINDOW, means, color=color, lw=lw, label=label, zorder=3)

    ax.axhline(CLAIMED_SEA_2025_PEAK, color=ALERT_RED, ls="--", lw=1.2,
               label=f"Published SEA 2025 peak ({CLAIMED_SEA_2025_PEAK})", zorder=2)
    ax.set_xlabel(f"Week the {WINDOW}-game window ends")
    ax.set_ylabel("Defensive EPA/play allowed (lower is better)")
    ax.set_title(
        f"Rolling {WINDOW}-game defensive EPA/play, every team-season 1999-2025\n"
        "Each faint line is one team's season; green is SEA 2025"
    )
    ax.invert_yaxis()
    ax.legend(fontsize=8, framealpha=0.2, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def drive_efficiency_chart(metrics: dict, path: Path) -> None:
    apply_scoreboard_style()
    items = [
        (v["label"], v["percentile"])
        for k, v in metrics.items()
        if k != "pressure_rate_created"
    ]
    items.sort(key=lambda kv: kv[1])
    labels = [a for a, _ in items]
    pcts = [b for _, b in items]
    colors = [ACTION_GREEN if p >= 90 else AMBER if p >= 70 else WOLF_GREY for p in pcts]

    fig, ax = plt.subplots(figsize=fig_size(5.4))
    bars = ax.barh(labels, pcts, color=colors)
    for bar, p in zip(bars, pcts):
        ax.text(p + 1, bar.get_y() + bar.get_height() / 2, f"{p:.0f}", va="center", fontsize=8)
    ax.axvline(50, color=OFF_WHITE, lw=0.7, alpha=0.5)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Percentile among 861 team-seasons, 1999-2025 (100 = best)")
    ax.set_title("SEA 2025 defense, metric by metric\nStrong nearly everywhere, ordinary at forcing three-and-outs")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    advanced = load("team_season_advanced.csv")
    reg = advanced[advanced["season_type"] == "REG"].reset_index(drop=True)

    games = load("team_game_efficiency.csv")
    games = games[games["season_type"] == "REG"].copy()

    filtered_best = rolling_best_by_team_season(games, "def_epa_per_play_allowed")
    unfiltered_best = rolling_best_by_team_season(games, "def_epa_per_play_allowed_unfiltered")

    claim_unfiltered = streak_claim(
        unfiltered_best, "unfiltered (all core plays, no garbage-time filter)",
        CLAIMED_SEA_2025_PEAK, CLAIMED_SEA_2013_PEAK,
    )
    claim_filtered = streak_claim(
        filtered_best, "garbage-time filtered (win probability 5%-95%), comparable to Phase 5",
        CLAIMED_SEA_2025_PEAK, CLAIMED_SEA_2013_PEAK,
    )

    support = supporting_metrics(reg)

    OUT_DIR.mkdir(exist_ok=True)
    results = {
        "methodology": {
            "window_games": WINDOW,
            "window_scope": "within a single regular season; windows never cross seasons, postseason excluded",
            "team_seasons_evaluated": len(unfiltered_best),
            "why_two_bases": (
                "Published EPA-streak figures are computed without a garbage-time filter, "
                "so the claim is tested unfiltered. The filtered version is reported "
                "alongside because it is what the rest of this project uses -- and it is "
                "the less flattering of the two for a dominant defense, which spends more "
                "snaps in garbage time precisely because it is dominant."
            ),
            "pressure_limitation": (
                "The pbp cache has no participation or pass-rusher data. Blitz rate is NOT "
                "computable; the published 19.2% figure cannot be checked here. Pressure is "
                "Phase 4's (sack OR qb_hit) proxy, which is narrower than a charted pressure "
                "and reads lower than the published 40.1%."
            ),
            "pressure_reference_window": (
                f"Pressure rate created is ranked from {QB_HIT_FIRST_SEASON} only, not 1999. "
                "nflverse qb_hit attribution is not stationary: 2003-2005 contain zero qb_hits "
                "and 1999-2002 about half the modern rate, so the proxy degrades to a bare sack "
                "rate in those years. Ranking 2025 against all 861 team-seasons overstated it. "
                "Every other metric on this page still uses the full 861."
            ),
        },
        "eight_week_streak_claim": {
            "claim": (
                f"SEA 2025's best {WINDOW}-week defensive stretch at {CLAIMED_SEA_2025_PEAK} EPA/play "
                f"is the best in 25 years; SEA 2013 peaked at {CLAIMED_SEA_2013_PEAK}"
            ),
            "source": "FTN / Sports Info Solutions",
            "unfiltered": claim_unfiltered,
            "garbage_time_filtered": claim_filtered,
        },
        "season_long_metrics": support,
    }
    with open(OUT_DIR / "defense_deep_dive.json", "w") as f:
        json.dump(results, f, indent=2)

    rolling_chart(games, "def_epa_per_play_allowed_unfiltered", OUT_DIR / "defense_rolling_epa.png")
    drive_efficiency_chart(support, OUT_DIR / "defense_drive_efficiency.png")

    for name, c in [("UNFILTERED", claim_unfiltered), ("FILTERED", claim_filtered)]:
        r, n = c["sea_2025_rank_of_n"]
        print(f"{name} best {WINDOW}-game defensive stretch (of {n} team-seasons since 1999):")
        print(f"  SEA 2025: {c['sea_2025_best_window']:+.3f} (weeks {c['sea_2025_window_weeks'][0]}-"
              f"{c['sea_2025_window_weeks'][1]}), rank {r}")
        print(f"  SEA 2013: {c['sea_2013_best_window']:+.3f}  -> SEA 2025 better? {c['beats_2013_seahawks']}")
        ahead = c["team_seasons_ahead"]
        print(f"  Ahead of SEA 2025: {[(a['season'], a['team']) for a in ahead[:5]] or 'none'}"
              f"{' ...' if len(ahead) > 5 else ''}\n")

    print("Season-long defensive metrics (percentile among all team-seasons in each window):")
    for v in sorted(support.values(), key=lambda d: -d["percentile"]):
        window = f"{v['reference_window'][0]}-{v['reference_window'][1]}"
        print(f"  {v['label']:<32} {v['value']:>8.3f}  {v['percentile']:>5.1f} pct  "
              f"({v['rank_of_n'][0]} of {v['rank_of_n'][1]}, {window}; #{v['rank_in_2025']} in 2025)")

    for name in ("defense_deep_dive.json", "defense_rolling_epa.png", "defense_drive_efficiency.png"):
        print(f"Wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
