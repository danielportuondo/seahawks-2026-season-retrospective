"""Phase 14: how they won, not just that they won -- testing the "one telling statistic."

A Sports Illustrated piece argued the 2025 Seahawks are historically dominant on
one measure: average end-of-game point differential, where they rank 2nd since
2000 behind only the 2013 Seahawks, both around +13. This phase reproduces that
claim exactly, and then does the thing the claim gestures at but does not
actually measure.

THE DISTINCTION THAT MOTIVATES THIS PHASE. "Average end-of-game margin" is just
mean final margin -- it is point differential per game with a more dramatic name.
It cannot tell a team that led 24-0 at halftime from a team that trailed and
scored 24 unanswered in the fourth quarter; both finish +24. The article's own
prose makes the stronger claim -- "the Seahawks get on top early, pile on, and
keep it up until the final whistle" -- which is a statement about the SHAPE of a
game, not its final score.

So this phase reports both:

  avg_final_margin        the article's actual statistic, reproduced as stated
  time_weighted_margin    the score margin integrated against the game clock

The second is the one that can distinguish the two seasons above, and it is
where an early-and-often team separates from a late-comeback team with the same
record. See season_metrics.time_weighted_margin for the construction.

THREE SUPPORTING CONTROL MEASURES, all clock-weighted rather than play-weighted
for the same reason:
  - share of game clock spent leading
  - share of game clock with win probability above 75%
  - games in which the team never trailed at any point

SCOPE. The SI claim is framed "since 2000" over Super Bowl champions, so that
exact reference set is reproduced for the headline comparison. Every other
ranking in this phase uses the project's standard 1999-2025 team-season window,
and the JSON labels which set each number came from -- mixing the two silently
would be the easiest way to manufacture a flattering rank.

INCLUDING PLAYOFFS. The published figure counts postseason games; a champion
plays three or four extra games against the league's best teams, which drags the
average down. Regular-season-only and combined numbers are both reported, since
quoting whichever is higher would be exactly the sin this project keeps warning
about.

WIN PROBABILITY is nflverse's model (`home_wp`/`away_wp`), not the Vegas-informed
variant -- see season_metrics' module docstring for why.
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
from phase13_historical_baseline import super_bowl_champions
from season_metrics import percentile_rank

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

FOCUS_TEAM = "SEA"
FOCUS_SEASON = 2025
SI_CLAIM_FLOOR = 2000  # the article's stated window


def load_advanced() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_season_advanced.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run src/season_metrics.py first")
    return pd.read_csv(path)


def combined_seasons(advanced: pd.DataFrame) -> pd.DataFrame:
    """Collapse REG + POST into one row per team-season.

    Time-weighted margin is a per-game mean, so combining the two phases means
    weighting each by its game count -- averaging the two averages would give a
    three-game postseason the same say as a seventeen-game regular season.
    """
    adv = advanced.copy()
    adv["twm_games"] = adv["time_weighted_margin"] * adv["games"]
    adv["lead_games"] = adv["pct_game_time_leading"] * adv["games"]
    adv["wp75_games"] = adv["pct_game_time_wp_gt_75"] * adv["games"]

    out = (
        adv.groupby(["season", "team"])
        .agg(
            points_for=("points_for", "sum"),
            points_against=("points_against", "sum"),
            games=("games", "sum"),
            wins=("wins", "sum"),
            games_never_trailed=("games_never_trailed", "sum"),
            twm_games=("twm_games", "sum"),
            lead_games=("lead_games", "sum"),
            wp75_games=("wp75_games", "sum"),
        )
        .reset_index()
    )
    out["avg_final_margin"] = (out["points_for"] - out["points_against"]) / out["games"]
    out["time_weighted_margin"] = out["twm_games"] / out["games"]
    out["pct_game_time_leading"] = out["lead_games"] / out["games"]
    out["pct_game_time_wp_gt_75"] = out["wp75_games"] / out["games"]
    out["pct_games_never_trailed"] = 100 * out["games_never_trailed"] / out["games"]
    return out.drop(columns=["twm_games", "lead_games", "wp75_games"])


def leaderboard(df: pd.DataFrame, metric: str, n: int = 10) -> list[dict]:
    top = df.sort_values(metric, ascending=False).head(n)
    return [
        {
            "rank": i + 1,
            "season": int(r.season),
            "team": str(r.team),
            "value": round(float(getattr(r, metric)), 3),
        }
        for i, r in enumerate(top.itertuples())
    ]


def rank_within(df: pd.DataFrame, metric: str, season: int, team: str) -> list[int]:
    value = float(df[(df["season"] == season) & (df["team"] == team)][metric].iloc[0])
    better = int((df[metric] > value).sum())
    return [better + 1, int(df[metric].notna().sum())]


def test_si_claim(combined: pd.DataFrame, champs: pd.DataFrame) -> dict:
    """Reproduce the SI ranking on its own terms, then on the stronger measure."""
    champ_rows = combined.merge(champs, on=["season", "team"], how="inner")
    champ_rows = champ_rows[champ_rows["season"] >= SI_CLAIM_FLOOR]

    final_rank = rank_within(champ_rows, "avg_final_margin", FOCUS_SEASON, FOCUS_TEAM)
    twm_rank = rank_within(champ_rows, "time_weighted_margin", FOCUS_SEASON, FOCUS_TEAM)
    sea = champ_rows[(champ_rows["season"] == FOCUS_SEASON) & (champ_rows["team"] == FOCUS_TEAM)].iloc[0]
    sea13 = champ_rows[(champ_rows["season"] == 2013) & (champ_rows["team"] == "SEA")]

    ahead = champ_rows[champ_rows["avg_final_margin"] > sea["avg_final_margin"]]

    return {
        "claim": (
            "2nd-highest average end-of-game point differential among Super Bowl "
            f"champions since {SI_CLAIM_FLOOR}, behind only the 2013 Seahawks, both around +13"
        ),
        "source": "Sports Illustrated / @AcccountStat",
        "reference_set": f"Super Bowl champions, {SI_CLAIM_FLOOR}-{FOCUS_SEASON}, regular season + playoffs",
        "n_champions": len(champ_rows),
        "sea_2025_avg_final_margin": round(float(sea["avg_final_margin"]), 2),
        "sea_2025_rank": final_rank,
        "champions_ahead": [
            {"season": int(r.season), "team": str(r.team), "avg_final_margin": round(float(r.avg_final_margin), 2)}
            for r in ahead.itertuples()
        ],
        "sea_2013_avg_final_margin": (
            round(float(sea13["avg_final_margin"].iloc[0]), 2) if len(sea13) else None
        ),
        "verdict": (
            "supported"
            if final_rank[0] == 2 and len(ahead) == 1 and set(ahead["season"]) == {2013}
            else "differs"
        ),
        "stronger_measure": {
            "metric": "time_weighted_margin",
            "why": (
                "Average final margin cannot distinguish a team that led wire-to-wire "
                "from one that won late by the same score. Weighting the margin by how "
                "long it was held can."
            ),
            "sea_2025_value": round(float(sea["time_weighted_margin"]), 2),
            "sea_2025_rank_among_champions": twm_rank,
            "sea_2013_value": (
                round(float(sea13["time_weighted_margin"].iloc[0]), 2) if len(sea13) else None
            ),
        },
    }


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def champions_chart(combined: pd.DataFrame, champs: pd.DataFrame, path: Path) -> None:
    apply_scoreboard_style()
    rows = combined.merge(champs, on=["season", "team"], how="inner")
    rows = rows[rows["season"] >= SI_CLAIM_FLOOR].sort_values("avg_final_margin")
    labels = [f"{int(r.season)} {r.team}" for r in rows.itertuples()]

    def colors_for(df: pd.DataFrame) -> list[str]:
        return [
            ACTION_GREEN
            if (r.season == FOCUS_SEASON and r.team == FOCUS_TEAM)
            else AMBER
            if (r.season == 2013 and r.team == "SEA")
            else WOLF_GREY
            for r in df.itertuples()
        ]

    fig, axes = plt.subplots(1, 2, figsize=fig_size(6.8), sharey=True)
    axes[0].barh(labels, rows["avg_final_margin"], color=colors_for(rows))
    axes[0].set_xlabel("Average final margin (pts/game)")
    axes[0].set_title("The published statistic\nAverage end-of-game differential")

    # Same row order on both panels so the eye reads the re-ranking directly.
    axes[1].barh(labels, rows["time_weighted_margin"], color=colors_for(rows))
    axes[1].set_xlabel("Clock-weighted margin (pts)")
    axes[1].set_title("The stronger version\nMargin weighted by how long it was held")

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(labelsize=8)
    fig.suptitle(
        f"Super Bowl champions since {SI_CLAIM_FLOOR} (regular season + playoffs)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def season_arc_chart(control: pd.DataFrame, path: Path) -> None:
    apply_scoreboard_style()
    df = control.sort_values(["season_type", "week"], ascending=[False, True]).reset_index(drop=True)
    x = np.arange(len(df))
    tick_labels = [
        f"W{int(r.week)}" if r.season_type == "REG" else "PO"
        for r in df.itertuples()
    ]
    colors = [ACTION_GREEN if v >= 0 else ALERT_RED for v in df["time_weighted_margin"]]

    fig, ax = plt.subplots(figsize=fig_size(5.0))
    ax.bar(x, df["time_weighted_margin"], color=colors)
    ax.plot(x, df["final_margin"], color=OFF_WHITE, lw=1.4, marker="o", ms=4, label="Final margin")
    ax.axhline(0, color=OFF_WHITE, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=0)
    ax.set_ylabel("Points")
    ax.set_title(
        "SEA 2025, game by game: clock-weighted margin (bars) vs final margin (line)\n"
        "A bar near its dot means the game was never close"
    )
    ax.legend(fontsize=8, framealpha=0.2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    advanced = load_advanced()
    combined = combined_seasons(advanced)
    champs = super_bowl_champions()
    reg = advanced[advanced["season_type"] == "REG"].reset_index(drop=True)

    control = pd.read_csv(PROCESSED_DIR / "team_game_control.csv")
    assert len(control) == 20, f"expected 20 SEA 2025 games, got {len(control)}"

    si = test_si_claim(combined, champs)

    sea_reg = reg[(reg["team"] == FOCUS_TEAM) & (reg["season"] == FOCUS_SEASON)].iloc[0]
    sea_all = combined[(combined["team"] == FOCUS_TEAM) & (combined["season"] == FOCUS_SEASON)].iloc[0]

    never_trailed = control[control["never_trailed"] == 1]
    wire_to_wire = {
        "games_never_trailed": int(sea_all["games_never_trailed"]),
        "of_games": int(sea_all["games"]),
        "pct": round(float(sea_all["pct_games_never_trailed"]), 1),
        "which": [
            f"{'W' + str(int(r.week)) if r.season_type == 'REG' else 'POST W' + str(int(r.week))}"
            for r in never_trailed.itertuples()
        ],
    }

    OUT_DIR.mkdir(exist_ok=True)
    results = {
        "methodology": {
            "window": "1999-2025 for all-team rankings; 2000-2025 champions for the SI claim",
            "n_team_seasons_reg": len(reg),
            "time_weighted_margin": (
                "sum(margin_after_play * clock_seconds_until_next_play) / total_seconds. "
                "Overtime carries zero weight because nflverse reports 0 seconds remaining "
                "throughout OT, making this a regulation-clock-weighted measure."
            ),
            "win_probability_source": "nflverse home_wp/away_wp (model only, not Vegas-informed)",
            "combining_reg_and_post": "per-game means combined weighted by game count, not averaged",
        },
        "si_claim": si,
        "sea_2025": {
            "regular_season": {
                "avg_final_margin": round(float(sea_reg["point_diff_per_game"]), 2),
                "time_weighted_margin": round(float(sea_reg["time_weighted_margin"]), 2),
                "pct_game_time_leading": round(float(sea_reg["pct_game_time_leading"]), 1),
                "pct_game_time_wp_gt_75": round(float(sea_reg["pct_game_time_wp_gt_75"]), 1),
                "games_never_trailed": int(sea_reg["games_never_trailed"]),
            },
            "including_playoffs": {
                "avg_final_margin": round(float(sea_all["avg_final_margin"]), 2),
                "time_weighted_margin": round(float(sea_all["time_weighted_margin"]), 2),
                "pct_game_time_leading": round(float(sea_all["pct_game_time_leading"]), 1),
                "pct_game_time_wp_gt_75": round(float(sea_all["pct_game_time_wp_gt_75"]), 1),
            },
            "wire_to_wire": wire_to_wire,
            "ranks_among_all_team_seasons_1999_2025": {
                "time_weighted_margin": rank_within(reg, "time_weighted_margin", FOCUS_SEASON, FOCUS_TEAM),
                "pct_game_time_leading": rank_within(reg, "pct_game_time_leading", FOCUS_SEASON, FOCUS_TEAM),
                "pct_game_time_wp_gt_75": rank_within(reg, "pct_game_time_wp_gt_75", FOCUS_SEASON, FOCUS_TEAM),
            },
            "percentiles_among_all_team_seasons_1999_2025": {
                m: round(percentile_rank(reg[m], float(sea_reg[m])), 1)
                for m in ("time_weighted_margin", "pct_game_time_leading", "pct_game_time_wp_gt_75")
            },
        },
        "leaderboards_1999_2025_regular_season": {
            "time_weighted_margin": leaderboard(reg, "time_weighted_margin"),
            "pct_game_time_leading": leaderboard(reg, "pct_game_time_leading"),
        },
        "per_game": control.to_dict(orient="records"),
    }
    with open(OUT_DIR / "game_control.json", "w") as f:
        json.dump(results, f, indent=2)

    champions_chart(combined, champs, OUT_DIR / "game_control_champions.png")
    season_arc_chart(control, OUT_DIR / "game_control_season_arc.png")

    print(f"SI claim: {si['verdict']}")
    print(f"  SEA 2025 avg final margin (incl. playoffs): {si['sea_2025_avg_final_margin']:+.2f}, "
          f"rank {si['sea_2025_rank'][0]} of {si['sea_2025_rank'][1]} champions since {SI_CLAIM_FLOOR}")
    print(f"  2013 SEA: {si['sea_2013_avg_final_margin']:+.2f}")
    print(f"  Champions ahead: {[c['season'] for c in si['champions_ahead']] or 'none'}")

    st = si["stronger_measure"]
    print(f"\nClock-weighted margin: SEA 2025 {st['sea_2025_value']:+.2f} "
          f"(rank {st['sea_2025_rank_among_champions'][0]} of {st['sea_2025_rank_among_champions'][1]}), "
          f"2013 SEA {st['sea_2013_value']:+.2f}")

    r = results["sea_2025"]["ranks_among_all_team_seasons_1999_2025"]
    print("\nAmong all 861 regular seasons since 1999:")
    print(f"  clock-weighted margin   {r['time_weighted_margin'][0]} of {r['time_weighted_margin'][1]}")
    print(f"  % of clock leading      {r['pct_game_time_leading'][0]} of {r['pct_game_time_leading'][1]}")
    print(f"  % of clock with WP>75%  {r['pct_game_time_wp_gt_75'][0]} of {r['pct_game_time_wp_gt_75'][1]}")
    print(f"\nNever trailed in {wire_to_wire['games_never_trailed']} of {wire_to_wire['of_games']} games "
          f"({wire_to_wire['pct']}%)")

    for name in ("game_control.json", "game_control_champions.png", "game_control_season_arc.png"):
        print(f"Wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
