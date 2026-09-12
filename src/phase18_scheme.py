"""Phase 18: the Dark Side defense, and what can honestly be said about a scheme.

Mike Macdonald's defense is described publicly in terms of disguise, simulated
pressures and a famously low blitz rate. Almost none of that is checkable here,
and this phase is largely an exercise in being clear about the difference between
what the play-by-play knows and what the coverage says.

WHAT IS NOT COMPUTABLE, AND WHY IT IS NOT ATTEMPTED. The cache carries no
participation or tracking data: no personnel groupings, no defenders in the box,
no pass-rusher counts, no coverage shell, no pre- or post-snap safety rotation,
no time to throw. That rules out blitz rate, simulated-pressure rate, man/zone
splits and every form of coverage attribution. Most importantly it rules out
DISGUISE ITSELF, which is the single thing Macdonald is best known for. There is
no honest proxy for it either: outcome variance is sometimes offered as one, but
game-to-game variance is dominated by opponent quality and sample size, and
Seattle's 2025 dispersion is high mostly because it played a hard schedule in
seventeen games. Presenting that as evidence of deception would be the most
over-claimy thing available in this dataset, so it is not presented at all.

WHAT IS COMPUTABLE IS THE SHAPE OF THE PASS RUSH, and it happens to be the
observable counterpart of the thing people mean when they say pressure came from
everywhere. Sacks are attributed reliably from 1999, so how a team's sacks are
spread across its roster can be ranked against all 861 team-seasons. Seattle 2025
is an extreme case: its sack total is unremarkable and its best individual season
is nowhere near the top of the all-time list, yet it fielded the league's best
scoring defense. That is a description, not a mechanism, and the output says so.

SACKS ARE COUNTED WITH HALF-CREDITS. `sack_player_id` alone misses roughly a
tenth of sacks -- the shared ones -- and a rotation-heavy defense is exactly the
kind that gets undercounted by dropping them. Including halves makes Seattle's
2025 total reconcile to the published 47.0 exactly.

QB HITS ARE DELIBERATELY ABSENT from every per-player number here. nflverse's
qb_hit attribution is not stationary (zero events in 2003-2005), which is why
Phase 15 now restricts its pressure percentile to 2006 onward. Rather than carry
a metric with two different denominators through a player-level analysis, this
phase uses sacks only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chart_style import (
    ACTION_GREEN,
    AMBER,
    OFF_WHITE,
    WOLF_GREY,
    apply_scoreboard_style,
    fig_size,
)
from phase13_historical_baseline import rank_in_good_direction
from season_metrics import herfindahl, percentile_rank, top_share

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

FOCUS_TEAM = "SEA"
FOCUS_SEASON = 2025
PRIOR_SEASON = 2024
HUNDRED_YARD_RUSHER = 100

GROUND_TRUTH = {"sacks": 47.0, "points_against_per_game": 17.2}


def load(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} missing -- run src/season_metrics.py first")
    return pd.read_csv(path)


def sack_distribution(defenders: pd.DataFrame) -> pd.DataFrame:
    """One row per team-season describing how its sacks were spread."""
    reg = defenders[(defenders["season_type"] == "REG") & (defenders["sacks"] > 0)]
    rows = []
    for (season, team), g in reg.groupby(["season", "team"]):
        sacks = g["sacks"].to_numpy()
        rows.append(
            {
                "season": int(season),
                "team": str(team),
                "team_sacks": float(sacks.sum()),
                "distinct_sackers": len(sacks),
                "top_sacker_share": top_share(sacks),
                "herfindahl": herfindahl(sacks),
                "best_individual": float(sacks.max()),
            }
        )
    return pd.DataFrame(rows)


def distribution_rankings(dist: pd.DataFrame, defenders: pd.DataFrame) -> dict:
    sea = dist[(dist["season"] == FOCUS_SEASON) & (dist["team"] == FOCUS_TEAM)].iloc[0]
    assert abs(sea["team_sacks"] - GROUND_TRUTH["sacks"]) < 0.01, (
        f"SEA 2025 sacks {sea['team_sacks']} != {GROUND_TRUTH['sacks']}"
    )

    out = {}
    for metric, label, higher_better in [
        ("top_sacker_share", "Share of sacks by the team's leading sacker", False),
        ("herfindahl", "Sack concentration (Herfindahl)", False),
        ("distinct_sackers", "Players with at least a half-sack", True),
        ("team_sacks", "Team sacks", True),
    ]:
        values = dist[metric].dropna()
        value = float(sea[metric])
        pct = percentile_rank(values, value)
        out[metric] = {
            "label": label,
            "value": round(value, 4),
            "rank_of_n": [rank_in_good_direction(values, value, higher_better), int(values.size)],
            "percentile": round(pct if higher_better else 100 - pct, 1),
            "league_mean_2025": round(
                float(dist[dist["season"] == FOCUS_SEASON][metric].mean()), 4
            ),
        }

    # Where the best individual Seahawk season sits among every sacker-season --
    # the point being that it is nowhere near the top, and it did not matter.
    reg = defenders[(defenders["season_type"] == "REG") & (defenders["sacks"] > 0)]
    best = float(sea["best_individual"])
    out["best_individual"] = {
        "label": "Best individual sack total on the team",
        "value": best,
        "rank_of_n": [
            rank_in_good_direction(reg["sacks"], best, True),
            int(reg["sacks"].notna().sum()),
        ],
        "note": "Ranked among every player-season with at least a half-sack since 1999.",
    }
    return out


def sea_pass_rush_roster(defenders: pd.DataFrame) -> list[dict]:
    sea = defenders[
        (defenders["season"] == FOCUS_SEASON)
        & (defenders["season_type"] == "REG")
        & (defenders["team"] == FOCUS_TEAM)
        & (defenders["sacks"] > 0)
    ].sort_values("sacks", ascending=False)
    return [
        {
            "player": str(r.player_name),
            "sacks": float(r.sacks),
            "interceptions": int(r.interceptions),
            "solo_tackles": int(r.solo_tackles),
        }
        for r in sea.itertuples()
    ]


def takeaway_leaders(defenders: pd.DataFrame) -> list[dict]:
    sea = defenders[
        (defenders["season"] == FOCUS_SEASON)
        & (defenders["season_type"] == "REG")
        & (defenders["team"] == FOCUS_TEAM)
        & (defenders["interceptions"] > 0)
    ].sort_values("interceptions", ascending=False)
    return [
        {
            "player": str(r.player_name),
            "interceptions": int(r.interceptions),
            "sacks": float(r.sacks),
            "solo_tackles": int(r.solo_tackles),
        }
        for r in sea.itertuples()
    ]


def rushing_wall(game_eff: pd.DataFrame) -> dict:
    """The individual-100-yard-rusher drought, counted the way a streak is counted.

    Games are ordered by season then week across seasons, because the streak the
    public figure describes runs through an offseason.
    """
    sea = game_eff[(game_eff["team"] == FOCUS_TEAM) & game_eff["opp_leading_rusher_yards"].notna()]
    sea = sea.sort_values(["season", "season_type", "week"], ascending=[True, False, True])
    allowed = sea["opp_leading_rusher_yards"].to_numpy()

    streak = 0
    for value in allowed[::-1]:
        if value >= HUNDRED_YARD_RUSHER:
            break
        streak += 1

    last = sea[sea["opp_leading_rusher_yards"] >= HUNDRED_YARD_RUSHER].tail(1)
    return {
        "games_without_allowing_a_100_yard_rusher": int(streak),
        "threshold_yards": HUNDRED_YARD_RUSHER,
        "last_allowed": (
            {
                "season": int(last.iloc[0]["season"]),
                "week": int(last.iloc[0]["week"]),
                "yards": float(last.iloc[0]["opp_leading_rusher_yards"]),
            }
            if len(last)
            else None
        ),
        "note": (
            "Counted over regular and postseason games in play-by-play order, so the "
            "streak runs across the offseason the way the published figure does."
        ),
    }


def year_over_year(advanced: pd.DataFrame) -> dict:
    """The same unit, the same head coach, one year apart."""
    reg = advanced[advanced["season_type"] == "REG"]
    out = {}
    for metric, label, higher_better in [
        ("def_epa_per_play_allowed", "Defensive EPA/play allowed", False),
        ("points_against_per_game", "Points allowed per game", False),
        ("def_yards_per_carry_allowed", "Yards per carry allowed", False),
        ("def_explosive_rate_allowed", "Explosive play rate allowed", False),
        ("def_success_rate_allowed", "Success rate allowed", False),
    ]:
        row = {}
        for season in (PRIOR_SEASON, FOCUS_SEASON):
            pool = reg[reg["season"] == season]
            value = float(pool[pool["team"] == FOCUS_TEAM].iloc[0][metric])
            row[str(season)] = {
                "value": round(value, 4),
                "rank_in_season": rank_in_good_direction(pool[metric], value, higher_better),
                "rank_all_time": [
                    rank_in_good_direction(reg[metric], value, higher_better),
                    int(reg[metric].notna().sum()),
                ],
            }
        row["label"] = label
        row["change"] = round(
            row[str(FOCUS_SEASON)]["value"] - row[str(PRIOR_SEASON)]["value"], 4
        )
        out[metric] = row
    return out


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def concentration_chart(dist: pd.DataFrame, path: Path) -> None:
    apply_scoreboard_style()
    sea = dist[(dist["season"] == FOCUS_SEASON) & (dist["team"] == FOCUS_TEAM)].iloc[0]
    fig, ax = plt.subplots(figsize=fig_size(5.2))
    ax.hist(dist["top_sacker_share"].dropna(), bins=55, color=WOLF_GREY, edgecolor=OFF_WHITE, linewidth=0.3)
    ax.axvline(float(sea["top_sacker_share"]), color=ACTION_GREEN, linewidth=2.5)
    league = float(dist[dist["season"] == FOCUS_SEASON]["top_sacker_share"].mean())
    ax.axvline(league, color=AMBER, linewidth=1.5, linestyle="--")
    ax.annotate(
        f"SEA 2025\n{sea['top_sacker_share']:.1%}",
        xy=(float(sea["top_sacker_share"]), ax.get_ylim()[1] * 0.72),
        xytext=(8, 0), textcoords="offset points",
        color=ACTION_GREEN, fontweight="bold",
    )
    ax.annotate(
        f"2025 league average {league:.1%}",
        xy=(league, ax.get_ylim()[1] * 0.92),
        xytext=(8, 0), textcoords="offset points",
        color=AMBER, fontsize=9,
    )
    ax.set_xlabel("Share of team sacks recorded by its leading sacker")
    ax.set_ylabel("Team-seasons")
    ax.set_title(
        "How much of a pass rush comes from one player, 1999-2025\n"
        "further left means the sacks were spread more widely"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def roster_chart(roster: list[dict], path: Path) -> None:
    apply_scoreboard_style()
    fig, ax = plt.subplots(figsize=fig_size(5.4))
    names = [r["player"] for r in roster][::-1]
    sacks = [r["sacks"] for r in roster][::-1]
    top = max(sacks)
    colors = [ACTION_GREEN if s == top else WOLF_GREY for s in sacks]
    y = np.arange(len(names))
    ax.barh(y, sacks, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    for i, v in enumerate(sacks):
        ax.text(v + 0.08, i, f"{v:g}", va="center", fontsize=8.5)
    ax.set_xlabel("Sacks (half-sacks counted as 0.5)")
    ax.set_title(
        f"{len(roster)} Seahawks recorded a sack in 2025, and none reached eight\n"
        f"{sum(sacks):g} sacks, three players tied at the top"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    defenders = load("defender_season.csv")
    advanced = load("team_season_advanced.csv")
    game_eff = load("team_game_efficiency.csv")

    dist = sack_distribution(defenders)
    rankings = distribution_rankings(dist, defenders)
    roster = sea_pass_rush_roster(defenders)
    takeaways = takeaway_leaders(defenders)
    wall = rushing_wall(game_eff)
    yoy = year_over_year(advanced)

    results = {
        "methodology": {
            "team_seasons": len(dist),
            "window": [1999, 2025],
            "sack_credit": (
                "Full sacks plus half-sack credits. sack_player_id alone misses roughly a "
                "tenth of sacks -- the shared ones -- and a rotational pass rush is exactly "
                "what that undercounts. With halves included SEA 2025 reconciles to the "
                "published 47.0 exactly."
            ),
            "distinct_sackers_definition": (
                "Players credited with at least a half-sack. A player with only a shared "
                "sack counts, which is why this can exceed a published count of whole-sack "
                "contributors."
            ),
            "not_computable": [
                "Blitz rate and pass-rusher counts -- no participation data in the cache",
                "Simulated pressures, stunts, coverage shells, man vs zone",
                "Pre-snap or post-snap safety rotation, i.e. disguise itself",
                "Snap counts, coverage snaps, targets or completions allowed per defender",
                "Individual offensive-line grades -- the play-by-play names no blockers",
            ],
            "no_disguise_proxy": (
                "No proxy for disguise is offered. Game-to-game outcome variance is sometimes "
                "used as one, but it is dominated by opponent quality and by having only "
                "seventeen games, so it would measure schedule and sample size rather than "
                "deception."
            ),
            "qb_hits_excluded": (
                "No QB-hit or pressure statistic appears at player level. nflverse qb_hit "
                "attribution is absent entirely for 2003-2005, so it cannot carry a single "
                "denominator across this window (see Phase 15)."
            ),
            "external_context_not_computed": [
                "First head coach to win a Super Bowl while calling his own defensive plays",
                "Coach of the Year voting; Macdonald finished third behind Mike Vrabel",
                "Baltimore's 2023 league lead in scoring defense, sacks and takeaways",
                "Blitz rate (~19-22%) and charted pressure rate, both from SIS/PFF",
            ],
        },
        "pass_rush_distribution": rankings,
        "sea_2025_pass_rush": roster,
        "sea_2025_takeaway_leaders": takeaways,
        "rushing_wall": wall,
        "year_over_year": yoy,
    }

    with open(OUT_DIR / "scheme_deep_dive.json", "w") as f:
        json.dump(results, f, indent=2)

    concentration_chart(dist, OUT_DIR / "scheme_sack_concentration.png")
    roster_chart(roster, OUT_DIR / "scheme_sack_roster.png")

    print(f"Pass-rush distribution, SEA {FOCUS_SEASON} among {len(dist):,} team-seasons:")
    for v in rankings.values():
        r, n = v["rank_of_n"]
        print(f"  {v['label']:<46} {v['value']:>8.3f}  rank {r} of {n:,}")

    print(f"\n{len(roster)} Seahawks recorded a sack; top {roster[0]['sacks']:g} "
          f"({sum(r['sacks'] for r in roster):g} total)")
    print(f"Games without allowing a 100-yard rusher: "
          f"{wall['games_without_allowing_a_100_yard_rusher']}")

    print("\n2024 -> 2025, same unit:")
    for v in yoy.values():
        a, b = v[str(PRIOR_SEASON)], v[str(FOCUS_SEASON)]
        print(f"  {v['label']:<34} {a['value']:>8.3f} (#{a['rank_in_season']}) -> "
              f"{b['value']:>8.3f} (#{b['rank_in_season']})  "
              f"[{b['rank_all_time'][0]} of {b['rank_all_time'][1]:,} all-time]")

    for name in ("scheme_deep_dive.json", "scheme_sack_concentration.png", "scheme_sack_roster.png"):
        print(f"Wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
