"""Phase 13: the missing denominator -- where SEA 2025 sits among all team-seasons since 1999.

Phases 2-7 answered "what changed from 2024 to 2025." That question has a
two-season denominator, so it can say the defense improved but not whether the
result was historically unusual. The public case for this team is stronger than
that -- "one of the all-time greats" -- and testing it needs every team-season
nflverse supports, which is 1999 onward: 861 regular seasons, 27 champions.

This phase ranks SEA 2025 on every metric season_metrics.py builds, against two
reference sets, and checks the specific published claims about the team.

TWO PERCENTILES PER METRIC, and the reason matters. The NFL's scoring
environment is not stationary across 1999-2025 -- rules changes moved passing
efficiency substantially -- so a raw percentile quietly rewards modern offenses
and modern-era defenses get punished for facing them. Phase 5 hit this same
problem and reported both a naive and a within-season rank
(robustness.era_naive_vs_within_season_rank); this phase follows that precedent:

  raw percentile        SEA 2025's value against all 861 raw values
  era-adjusted          SEA 2025's within-season z-score against all 861
                        within-season z-scores

Where the two disagree, the era-adjusted number is the honest one and the
narrative leads with whichever is LESS flattering. A metric that looks top-1%
raw and top-10% era-adjusted is a top-10% metric.

TIES: percentile_rank counts ties as half (see season_metrics), which matters
because several of these are integer-valued -- wins, point differential -- where
"fraction strictly below" would understate a season tied for the best.

RANK vs PERCENTILE: both are reported. Percentile is the comparable summary
across metrics; the "Nth of 861" rank is what a reader actually wants to hear,
and it is computed in the metric's own good direction (rank 1 = best), not
mechanically descending.

NO DVOA. It is proprietary and cannot be reproduced from play-by-play; where the
published claims cite it, this phase says so rather than substituting a
look-alike and implying equivalence.
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
)
from season_metrics import RELOCATION_MAP, percentile_rank

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

FOCUS_TEAM = "SEA"
FOCUS_SEASON = 2025

# metric -> (label, higher_is_better)
METRICS = {
    "point_diff_per_game": ("Point differential / game", True),
    "time_weighted_margin": ("Clock-weighted lead", True),
    "off_epa_per_play": ("Offensive EPA / play", True),
    "def_epa_per_play_allowed": ("Defensive EPA / play allowed", False),
    "off_success_rate": ("Offensive success rate", True),
    "def_success_rate_allowed": ("Defensive success rate allowed", False),
    "off_explosive_rate": ("Explosive play rate", True),
    "def_explosive_rate_allowed": ("Explosive plays allowed", False),
    "points_per_drive": ("Points per drive", True),
    "points_per_drive_allowed": ("Points per drive allowed", False),
    "three_and_out_rate_forced": ("Three-and-outs forced", True),
    "def_yards_per_carry_allowed": ("Yards per carry allowed", False),
    "turnover_margin": ("Turnover margin", True),
    "pct_game_time_leading": ("% of clock leading", True),
}

GROUND_TRUTH = {"wins": 14, "losses": 3, "points_for": 483, "point_diff": 191}


def load_advanced() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_season_advanced.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run src/season_metrics.py first")
    return pd.read_csv(path)


def super_bowl_champions() -> pd.DataFrame:
    """Champion per season, derived from the schedule rather than hardcoded."""
    sched = pd.read_parquet(RAW_DIR / "schedules.parquet")
    sb = sched[(sched["game_type"] == "SB") & (sched["season"] >= 1999)].dropna(
        subset=["home_score", "away_score"]
    )
    champ = np.where(sb["home_score"] > sb["away_score"], sb["home_team"], sb["away_team"])
    out = pd.DataFrame({"season": sb["season"].to_numpy(), "team": champ})
    out["team"] = out["team"].replace(RELOCATION_MAP)
    return out.sort_values("season").reset_index(drop=True)


def add_era_adjustment(reg: pd.DataFrame) -> pd.DataFrame:
    """Within-season z-score for every metric: the era-neutral view."""
    out = reg.copy()
    for metric in METRICS:
        grp = out.groupby("season")[metric]
        out[f"z_{metric}"] = (out[metric] - grp.transform("mean")) / grp.transform("std", ddof=1)
    return out


def rank_in_good_direction(values: pd.Series, x: float, higher_is_better: bool) -> int:
    """1 = best. Ties share the better rank, as standings conventionally do."""
    v = values.dropna()
    better = (v > x).sum() if higher_is_better else (v < x).sum()
    return int(better + 1)


def score_metric(reg: pd.DataFrame, metric: str, sea: pd.Series, champs: pd.DataFrame) -> dict:
    label, higher_is_better = METRICS[metric]
    value = float(sea[metric])
    z_col = f"z_{metric}"

    raw_pct = percentile_rank(reg[metric], value)
    z_pct = percentile_rank(reg[z_col], float(sea[z_col]))
    # Percentile is computed in the raw scale; flip it when low is good so that
    # "99th percentile" always means "99% of team-seasons were worse."
    if not higher_is_better:
        raw_pct = 100 - raw_pct
        z_pct = 100 - z_pct

    champ_rows = reg.merge(champs, on=["season", "team"], how="inner")

    return {
        "label": label,
        "higher_is_better": higher_is_better,
        "sea_2025_value": round(value, 4),
        "raw_percentile": round(raw_pct, 1),
        "era_adjusted_percentile": round(z_pct, 1),
        "headline_percentile": round(min(raw_pct, z_pct), 1),
        "within_season_z": round(float(sea[z_col]), 2),
        "rank_of_n": [rank_in_good_direction(reg[metric], value, higher_is_better), int(reg[metric].notna().sum())],
        "champion_rank_of_n": [
            rank_in_good_direction(champ_rows[metric], value, higher_is_better),
            int(champ_rows[metric].notna().sum()),
        ],
        "league_mean_2025": round(float(reg[reg["season"] == FOCUS_SEASON][metric].mean()), 4),
        "all_time_best": _extreme(reg, metric, higher_is_better),
    }


def _extreme(reg: pd.DataFrame, metric: str, higher_is_better: bool) -> dict:
    idx = reg[metric].idxmax() if higher_is_better else reg[metric].idxmin()
    row = reg.loc[idx]
    return {"season": int(row["season"]), "team": str(row["team"]), "value": round(float(row[metric]), 4)}


def check_published_claims(reg: pd.DataFrame, full: pd.DataFrame, champs: pd.DataFrame) -> list[dict]:
    """Each published claim, the computed value, and whether they agree."""
    claims = []

    sea_reg = reg[(reg["team"] == FOCUS_TEAM) & (reg["season"] == FOCUS_SEASON)].iloc[0]
    claims.append(
        {
            "claim": "+191 regular-season point differential, best in the NFL and in franchise history",
            "source": "seahawks.com",
            "computed": int(sea_reg["point_diff"]),
            "nfl_rank_2025": rank_in_good_direction(
                reg[reg["season"] == FOCUS_SEASON]["point_diff"], float(sea_reg["point_diff"]), True
            ),
            "verdict": "supported" if int(sea_reg["point_diff"]) == 191 else "differs",
        }
    )

    # Regular season + playoffs combined, which is how the +246 figure is quoted.
    combined = (
        full.groupby(["season", "team"])
        .agg(points_for=("points_for", "sum"), points_against=("points_against", "sum"))
        .reset_index()
    )
    combined["point_diff"] = combined["points_for"] - combined["points_against"]
    sea_all = combined[(combined["team"] == FOCUS_TEAM) & (combined["season"] == FOCUS_SEASON)].iloc[0]

    champ_combined = combined.merge(champs, on=["season", "team"], how="inner").sort_values(
        "point_diff", ascending=False
    )
    better_champs = champ_combined[champ_combined["point_diff"] > sea_all["point_diff"]]
    claims.append(
        {
            "claim": "+246 including playoffs, best by a Super Bowl champion since the 1999 Rams",
            "source": "seahawks.com",
            "computed": int(sea_all["point_diff"]),
            "champions_since_1999_with_better": [
                {"season": int(r["season"]), "team": str(r["team"]), "point_diff": int(r["point_diff"])}
                for _, r in better_champs.iterrows()
            ],
            "verdict": (
                "supported"
                if int(sea_all["point_diff"]) == 246
                and set(better_champs["season"]).issubset({1999})
                else "differs"
            ),
        }
    )

    ppg_allowed = float(sea_reg["points_against_per_game"])
    rank_2025 = rank_in_good_direction(
        reg[reg["season"] == FOCUS_SEASON]["points_against_per_game"], ppg_allowed, False
    )
    claims.append(
        {
            "claim": "17.2 points per game allowed, the NFL's No. 1 scoring defense",
            "source": "seahawks.com",
            "computed": round(ppg_allowed, 1),
            "nfl_rank_2025": rank_2025,
            "verdict": "supported" if round(ppg_allowed, 1) == 17.2 and rank_2025 == 1 else "differs",
        }
    )

    ypc = float(sea_reg["def_yards_per_carry_allowed"])
    ypc_rank = rank_in_good_direction(
        reg[reg["season"] == FOCUS_SEASON]["def_yards_per_carry_allowed"], ypc, False
    )
    claims.append(
        {
            "claim": "3.7 yards per carry allowed, a league low",
            "source": "seahawks.com",
            "computed": round(ypc, 2),
            "nfl_rank_2025": ypc_rank,
            "note": (
                "Computed from play-by-play over all rush attempts. Published rushing "
                "splits sometimes exclude quarterback kneels and scrambles, so a small "
                "difference from 3.7 is a definitional gap, not a contradiction."
            ),
            "verdict": "supported" if ypc_rank == 1 else "differs",
        }
    )

    claims.append(
        {
            "claim": "No. 1 defense by DVOA, 8th best since 1978",
            "source": "seahawks.com / FTN",
            "computed": None,
            "verdict": "not reproducible",
            "note": "DVOA is proprietary. Cited as external context only; not recomputed here.",
        }
    )

    return claims


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def scorecard_chart(scores: dict, path: Path) -> None:
    apply_scoreboard_style()
    items = sorted(scores.items(), key=lambda kv: kv[1]["headline_percentile"])
    labels = [v["label"] for _, v in items]
    raw = [v["raw_percentile"] for _, v in items]
    era = [v["era_adjusted_percentile"] for _, v in items]

    fig, ax = plt.subplots(figsize=(9.5, 7))
    y = np.arange(len(labels))

    for yi, r, e in zip(y, raw, era):
        ax.plot([min(r, e), max(r, e)], [yi, yi], color=WOLF_GREY, lw=1.5, zorder=1)
    ax.scatter(raw, y, s=70, color=WOLF_GREY, zorder=2, label="Raw percentile")
    ax.scatter(era, y, s=70, color=ACTION_GREEN, zorder=3, label="Era-adjusted (within-season)")

    ax.axvline(50, color=OFF_WHITE, lw=0.7, alpha=0.5)
    ax.axvline(99, color=AMBER, lw=0.7, ls="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 104)
    ax.set_xlabel("Percentile among 861 team-seasons, 1999-2025 (100 = best)")
    ax.set_title("SEA 2025 against every team-season since 1999\nGrey = raw, green = era-adjusted; the gap is the rules-era effect")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.2)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def distribution_chart(reg: pd.DataFrame, scores: dict, path: Path) -> None:
    apply_scoreboard_style()
    picks = [
        "point_diff_per_game",
        "time_weighted_margin",
        "def_epa_per_play_allowed",
        "off_epa_per_play",
        "points_per_drive_allowed",
        "def_explosive_rate_allowed",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.ravel(), picks):
        vals = reg[metric].dropna()
        sea_val = scores[metric]["sea_2025_value"]
        ax.hist(vals, bins=40, color=WOLF_GREY, alpha=0.75, edgecolor="none")
        ax.axvline(sea_val, color=ACTION_GREEN, lw=2.2)
        ax.set_title(
            f"{scores[metric]['label']}\nSEA 2025 = {sea_val:.3g} "
            f"({scores[metric]['headline_percentile']:.0f}th pct)",
            fontsize=9,
        )
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Where SEA 2025 falls in the 1999-2025 distribution (861 team-seasons)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def champion_chart(reg: pd.DataFrame, champs: pd.DataFrame, path: Path) -> None:
    apply_scoreboard_style()
    champ_rows = reg.merge(champs, on=["season", "team"], how="inner").sort_values(
        "point_diff_per_game"
    )
    labels = [f"{int(r.season)} {r.team}" for r in champ_rows.itertuples()]
    vals = champ_rows["point_diff_per_game"].to_numpy()
    colors = [
        ACTION_GREEN
        if (r.season == FOCUS_SEASON and r.team == FOCUS_TEAM)
        else AMBER
        if (r.season == 2013 and r.team == "SEA")
        else WOLF_GREY
        for r in champ_rows.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Regular-season point differential per game")
    ax.set_title("Every Super Bowl champion since 1999, by regular-season margin")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    advanced = load_advanced()
    reg = advanced[advanced["season_type"] == "REG"].reset_index(drop=True)
    assert len(reg) == 861, f"expected 861 REG team-seasons, got {len(reg)}"

    reg = add_era_adjustment(reg)
    champs = super_bowl_champions()

    sea = reg[(reg["team"] == FOCUS_TEAM) & (reg["season"] == FOCUS_SEASON)].iloc[0]
    for key, expected in GROUND_TRUTH.items():
        assert int(sea[key]) == expected, f"SEA 2025 {key}={sea[key]}, expected {expected}"

    scores = {m: score_metric(reg, m, sea, champs) for m in METRICS}
    claims = check_published_claims(reg, advanced, champs)

    top1_metrics = [v["label"] for v in scores.values() if v["rank_of_n"][0] == 1]
    era_gaps = {
        m: round(v["raw_percentile"] - v["era_adjusted_percentile"], 1)
        for m, v in scores.items()
        if abs(v["raw_percentile"] - v["era_adjusted_percentile"]) >= 2
    }

    OUT_DIR.mkdir(exist_ok=True)
    results = {
        "methodology": {
            "window": "1999-2025",
            "n_team_seasons_reg": len(reg),
            "n_champions": len(champs),
            "percentile_convention": "100 = best; ties count as half; low-is-better metrics are flipped so higher percentile is always better",
            "era_adjustment": "within-season z-score, then percentile of that z across all 861 team-seasons",
            "headline_percentile": "min(raw, era-adjusted) -- the less flattering of the two, by design",
            "rank_convention": "1 = best in the metric's own good direction; ties share the better rank",
            "excluded": "DVOA (proprietary); charted pressure rate and blitz rate (no participation data in the pbp cache)",
        },
        "sea_2025_summary": {
            "record": f"{int(sea['wins'])}-{int(sea['losses'])}",
            "points_for": int(sea["points_for"]),
            "points_against": int(sea["points_against"]),
            "point_diff": int(sea["point_diff"]),
            "metrics_ranked_first_all_time": top1_metrics,
            "era_adjustment_gaps": era_gaps,
        },
        "metrics": scores,
        "published_claims": claims,
        "champions": [
            {"season": int(r.season), "team": r.team} for r in champs.itertuples()
        ],
    }
    with open(OUT_DIR / "historical_percentiles.json", "w") as f:
        json.dump(results, f, indent=2)

    scorecard_chart(scores, OUT_DIR / "historical_scorecard.png")
    distribution_chart(reg, scores, OUT_DIR / "historical_distributions.png")
    champion_chart(reg, champs, OUT_DIR / "historical_champions.png")

    print(f"Reference set: {len(reg)} team-seasons, {len(champs)} champions (1999-2025)\n")
    print(f"{'Metric':<34} {'value':>9} {'raw':>6} {'era':>6}  rank")
    for _, v in sorted(scores.items(), key=lambda kv: -kv[1]["headline_percentile"]):
        r, n = v["rank_of_n"]
        print(
            f"{v['label']:<34} {v['sea_2025_value']:>9.3f} "
            f"{v['raw_percentile']:>6.1f} {v['era_adjusted_percentile']:>6.1f}  {r} of {n}"
        )

    print("\nPublished claims:")
    for c in claims:
        print(f"  [{c['verdict']:<16}] {c['claim']}  -> computed: {c['computed']}")

    for name in ("historical_percentiles.json", "historical_scorecard.png",
                 "historical_distributions.png", "historical_champions.png"):
        print(f"Wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
