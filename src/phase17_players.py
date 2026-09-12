"""Phase 17: two players, measured against everyone who came before them.

Jaxon Smith-Njigba led the NFL in receiving yards and won Offensive Player of the
Year on an offense that threw less than almost anybody. Sam Darnold led the NFL
in turnovers and then committed none at all in January. Both stories are about a
number being surprising *given* something else, which is exactly where it is
easiest to talk nonsense, so both are handled deliberately here.

WHY THERE IS NO PROBABILITY ATTACHED TO THE RECEIVING SEASON. The tempting
framing -- "how unlikely is 1,793 yards on only 481 pass attempts" -- is not a
question this data can answer, and the reason is not fussiness:

  1. The conditioning set is endogenous and plausibly a collider. Seattle ran the
     ball partly because it was ahead (Phase 14), and it was ahead partly because
     Smith-Njigba was producing. Pass volume is not a constraint handed down from
     outside the season; it is a jointly determined outcome. P(yards | volume)
     would be treating one half of a feedback loop as fixed.
  2. There is no defensible null to draw from. Receiving yards are heavy-tailed,
     serially correlated within games and correlated with game script. A binomial
     or Poisson resample of targets assumes independent exchangeable trials, and
     that assumption fails hardest in the tail -- precisely where the answer would
     live. Phase 6 can use a Poisson for turnovers because turnovers really are
     rare, discrete and near-exchangeable. Yardage is a different object, and
     reusing that machinery here would borrow credibility it has not earned.
  3. The surprise is definitional rather than statistical. Yards per team pass
     attempt is identically target share times yards per target. Smith-Njigba is
     first all-time on the product and sixth on the share, but only around the
     96th percentile on efficiency per target. The rarity is concentration of
     opportunity, which is a coaching decision, not a coin that came up heads.

So this phase ranks instead of simulating. Every number is a position in a stated
reference set, which is the same thing Phase 13 does for teams and is a claim that
cannot be wrong in the way a fabricated probability can be.

THE OPPORTUNITY DENOMINATOR IS A PROXY AND IS LABELLED AS ONE. The honest
denominator for a receiver's opportunity is routes run, and routes are not in the
play-by-play at any price -- there is no participation data in this cache. Target
share over team pass attempts therefore conflates "ran a route" with "was on the
field", and it will slightly flatter any receiver who never leaves the field.
That is a real limitation, not a footnote, and it is recorded in the output.

YARDS OVER EXPECTED IS A SHORTER ANALYSIS THAN EVERYTHING ELSE HERE. It needs air
yards, completion probability and expected YAC, none of which exist before 2006,
so it is ranked over 2006-2025 and says so. The underlying models are nflverse's,
not this project's.

DARNOLD'S ARC IS THE ONE PLACE A PROBABILITY IS DEFENSIBLE, and Phase 6 already
built it. This phase only widens the baseline: Phase 6 compared his clean
postseason against his own 2025 rate, and 2025 turns out to be the worst turnover
season of his career, which sharpens rather than softens the story.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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
from season_metrics import AIR_YARDS_FIRST_SEASON, percentile_rank

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

FOCUS_TEAM = "SEA"
FOCUS_SEASON = 2025

# A receiver-season needs enough volume for a rate to mean anything. 50 targets
# is roughly three a game over a full season and keeps 3,504 seasons in the set;
# raising it to 100 would drop the set to ~1,100 and quietly exclude the
# low-volume seasons that make the concentration comparison interesting.
MIN_TARGETS = 50

JSN_NAME = "J.Smith-Njigba"
DARNOLD_NAME = "S.Darnold"
MURPHY_NAME = "B.Murphy"
PRIOR_SEASON = 2024

# Enough carries for a rate to mean anything without excluding a committed
# change-of-pace back, which is precisely the role this phase is looking at.
MIN_CARRIES = 50

# Ground truth, asserted before anything downstream runs. Every figure here is
# independently published; if the pipeline stops reproducing them, fail loudly.
JSN_TRUTH = {"targets": 163, "receptions": 119, "receiving_yards": 1793, "receiving_tds": 10}
JSN_TEAM_PASS_ATTEMPTS = 481  # official NFL figure for SEA 2025
DARNOLD_2025_TRUTH = {"dropbacks": 516, "interceptions": 14, "fumbles_lost": 6, "turnovers": 20}
DARNOLD_POST_TRUTH = {"dropbacks": 99, "turnovers": 0}


def load(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} missing -- run src/season_metrics.py first")
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Receiver
# --------------------------------------------------------------------------


def receiver_reference(receivers: pd.DataFrame) -> pd.DataFrame:
    reg = receivers[(receivers["season_type"] == "REG") & (receivers["targets"] >= MIN_TARGETS)]
    return reg.copy()


def focus_receiver(reference: pd.DataFrame) -> pd.Series:
    sea = reference[(reference["season"] == FOCUS_SEASON) & (reference["team"] == FOCUS_TEAM)]
    row = sea.sort_values("receiving_yards", ascending=False).iloc[0]
    assert row["receiver"] == JSN_NAME, f"expected {JSN_NAME}, got {row['receiver']}"
    for key, expected in JSN_TRUTH.items():
        assert int(row[key]) == expected, f"{key}: got {row[key]}, expected {expected}"
    assert int(row["team_pass_attempts"]) == JSN_TEAM_PASS_ATTEMPTS, (
        f"team pass attempts {row['team_pass_attempts']} != official {JSN_TEAM_PASS_ATTEMPTS}"
    )
    return row


def receiver_rankings(reference: pd.DataFrame, jsn: pd.Series) -> dict:
    """Rank Smith-Njigba on each concentration and production measure."""
    out = {}
    for metric, label, window in [
        ("yards_per_team_pass_attempt", "Receiving yards per team pass attempt", None),
        ("target_share", "Target share (of team pass attempts)", None),
        ("receiving_yards", "Receiving yards", None),
        ("team_receiving_yards_share", "Share of team receiving yards", None),
        ("yards_per_target", "Yards per target", None),
        ("yards_over_expected", "Receiving yards over expected", AIR_YARDS_FIRST_SEASON),
    ]:
        ref = reference if window is None else reference[reference["season"] >= window]
        values = ref[metric].dropna()
        value = float(jsn[metric])
        out[metric] = {
            "label": label,
            "value": round(value, 4),
            "rank_of_n": [rank_in_good_direction(values, value, True), int(values.size)],
            "percentile": round(percentile_rank(values, value), 2),
            "reference_window": [int(ref["season"].min()), int(ref["season"].max())],
            "top5": [
                {
                    "season": int(r.season),
                    "team": str(r.team),
                    "receiver": str(r.receiver),
                    "value": round(float(getattr(r, metric)), 4),
                }
                for r in ref.nlargest(5, metric).itertuples()
            ],
        }
    return out


def volume_barrier(reference: pd.DataFrame, jsn: pd.Series) -> dict:
    """Test the published claim about how much volume the seasons above him had.

    The public framing is that every receiver with a bigger season worked behind a
    far busier offense. That is checkable directly rather than repeatable on faith.
    """
    above = reference[reference["receiving_yards"] > jsn["receiving_yards"]]
    above = above.sort_values("receiving_yards", ascending=False)
    return {
        "claim": (
            "Every receiver above Smith-Njigba on the single-season receiving list "
            "played on an offense that attempted at least 566 passes; Seattle threw 481."
        ),
        "source": "Seahawks.com, 2026-01-21",
        "computed": {
            "sea_2025_team_pass_attempts": int(jsn["team_pass_attempts"]),
            "n_receiver_seasons_above": len(above),
            "min_team_pass_attempts_above": int(above["team_pass_attempts"].min()),
            "seasons_above": [
                {
                    "season": int(r.season),
                    "team": str(r.team),
                    "receiver": str(r.receiver),
                    "receiving_yards": int(r.receiving_yards),
                    "team_pass_attempts": int(r.team_pass_attempts),
                }
                for r in above.itertuples()
            ],
        },
        "verdict": "supported" if int(above["team_pass_attempts"].min()) >= 566 else "differs",
    }


# --------------------------------------------------------------------------
# Quarterback
# --------------------------------------------------------------------------


def darnold_career(passers: pd.DataFrame) -> pd.DataFrame:
    car = passers[passers["player_name"] == DARNOLD_NAME].copy()
    car = car.sort_values(["season", "season_type"], ascending=[True, False])
    reg25 = car[(car["season"] == FOCUS_SEASON) & (car["season_type"] == "REG")].iloc[0]
    for key, expected in DARNOLD_2025_TRUTH.items():
        assert int(reg25[key]) == expected, f"2025 {key}: got {reg25[key]}, expected {expected}"
    post25 = car[(car["season"] == FOCUS_SEASON) & (car["season_type"] == "POST")].iloc[0]
    for key, expected in DARNOLD_POST_TRUTH.items():
        assert int(post25[key]) == expected, f"2025 POST {key}: got {post25[key]}, expected {expected}"
    return car


def clean_postseason_probability(career: pd.DataFrame) -> dict:
    """P(zero turnovers across the playoff run) at his own regular-season rate.

    Phase 6 established the method; the only change here is that the rate being
    tested against is now visible next to every other season he has played.
    """
    reg = career[(career["season"] == FOCUS_SEASON) & (career["season_type"] == "REG")].iloc[0]
    post = career[(career["season"] == FOCUS_SEASON) & (career["season_type"] == "POST")].iloc[0]
    rate = float(reg["turnover_rate_per_dropback"])
    n = int(post["dropbacks"])
    expected = rate * n
    return {
        "regular_season_rate_per_dropback": round(rate, 5),
        "playoff_dropbacks": n,
        "expected_turnovers": round(expected, 3),
        "p_zero_poisson": round(float(stats.poisson.pmf(0, expected)), 5),
        "p_zero_binomial": round(float(stats.binom.pmf(0, n, rate)), 5),
        "odds_against_1_in": round(1 / float(stats.poisson.pmf(0, expected)), 1),
        "caveat": (
            "The playoff opponent set is not a random draw from the regular season and "
            "Seattle led for most of all three games, so this is an upper bound on how "
            "surprising the run was, not a point estimate."
        ),
    }


def career_turnover_ranking(career: pd.DataFrame) -> dict:
    """Where 2025 sits among his own seasons. The arc is the finding."""
    reg = career[career["season_type"] == "REG"].copy()
    worst = reg.sort_values("turnover_rate_per_dropback", ascending=False).iloc[0]
    return {
        "worst_rate_season": int(worst["season"]),
        "worst_rate_team": str(worst["team"]),
        "worst_rate_per_dropback": round(float(worst["turnover_rate_per_dropback"]), 5),
        "focus_season_is_career_worst": bool(int(worst["season"]) == FOCUS_SEASON),
        "note": (
            "2025 is the worst turnover rate of his career by this measure -- worse than "
            "his rookie year -- which is the opposite of a player who gradually cleaned it "
            "up. The clean postseason is a break from his own form, not the end of a trend."
        ),
    }


# --------------------------------------------------------------------------
# Running backs and the year-two leap
# --------------------------------------------------------------------------


def backfield_split(rushers: pd.DataFrame) -> dict:
    """Two backs, one backfield, and a touchdown split that looks inexplicable.

    Charbonnet scored more than twice Walker's rushing touchdowns on fewer carries
    and a lower average. The tempting read is that he was simply better near the
    goal line. The honest one is visible the moment goal-line carries are counted
    separately: he was given most of them. Opportunity, not finishing.
    """
    reg = rushers[(rushers["season_type"] == "REG") & (rushers["carries"] >= MIN_CARRIES)]
    sea = reg[(reg["season"] == FOCUS_SEASON) & (reg["team"] == FOCUS_TEAM)]
    sea = sea.nlargest(2, "carries")

    backs = []
    for r in sea.itertuples():
        backs.append(
            {
                "player": str(r.player_name),
                "carries": int(r.carries),
                "rushing_yards": int(r.rushing_yards),
                "yards_per_carry": round(float(r.yards_per_carry), 2),
                "epa_per_rush": round(float(r.epa_per_rush), 4),
                "rushing_tds": int(r.rushing_tds),
                "carries_inside_5": int(r.carries_inside_5),
                "tds_inside_5": int(r.tds_inside_5),
                "goal_line_carry_share": round(float(r.goal_line_carry_share), 4),
                "carry_share": round(float(r.carry_share), 4),
            }
        )

    scorer = max(backs, key=lambda b: b["rushing_tds"])
    td_values = reg["rushing_tds"].dropna()
    return {
        "backs": backs,
        "touchdown_leader": scorer["player"],
        "touchdown_rank_of_n": [
            rank_in_good_direction(td_values, scorer["rushing_tds"], True),
            int(td_values.size),
        ],
        "reading": (
            "The touchdown gap is a goal-line usage gap. Charbonnet took "
            f"{scorer['goal_line_carry_share']:.0%} of Seattle's carries inside the five "
            "while carrying the ball less often everywhere else. Touchdown totals are "
            "mostly a story about who gets handed the ball on the two-yard line."
        ),
    }


def year_two_leap(defenders: pd.DataFrame) -> dict:
    """Rank Byron Murphy's sack jump against every back-to-back defender season.

    A raw sack delta flatters players who simply played more, so the comparison
    set is every pair of consecutive seasons by the same player for the same team
    since 1999 -- which is the widest honest denominator this cache supports.
    """
    reg = defenders[defenders["season_type"] == "REG"][
        ["season", "team", "player_id", "player_name", "sacks"]
    ].copy()
    prior = reg.copy()
    prior["season"] = prior["season"] + 1
    pairs = reg.merge(
        prior,
        on=["season", "team", "player_id", "player_name"],
        suffixes=("", "_prior"),
        how="inner",
    )
    pairs["jump"] = pairs["sacks"] - pairs["sacks_prior"]

    murphy = pairs[
        (pairs["season"] == FOCUS_SEASON)
        & (pairs["team"] == FOCUS_TEAM)
        & (pairs["player_name"] == MURPHY_NAME)
    ]
    if murphy.empty:
        raise SystemExit(f"{MURPHY_NAME} has no {PRIOR_SEASON}->{FOCUS_SEASON} pair")
    row = murphy.iloc[0]

    return {
        "player": MURPHY_NAME,
        "prior_season": PRIOR_SEASON,
        "prior_sacks": float(row["sacks_prior"]),
        "season": FOCUS_SEASON,
        "sacks": float(row["sacks"]),
        "jump": float(row["jump"]),
        "rank_of_n": [
            rank_in_good_direction(pairs["jump"], float(row["jump"]), True),
            len(pairs),
        ],
        "percentile": round(percentile_rank(pairs["jump"], float(row["jump"])), 2),
        "comparison_set": (
            "Every pair of consecutive regular seasons by the same player for the same "
            "team since 1999, counting half-sacks."
        ),
    }


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def concentration_chart(reference: pd.DataFrame, jsn: pd.Series, path: Path) -> None:
    apply_scoreboard_style()
    fig, ax = plt.subplots(figsize=fig_size(5.4))
    values = reference["yards_per_team_pass_attempt"].dropna()
    ax.hist(values, bins=60, color=WOLF_GREY, edgecolor=OFF_WHITE, linewidth=0.3)
    ax.axvline(float(jsn["yards_per_team_pass_attempt"]), color=ACTION_GREEN, linewidth=2.5)
    ax.annotate(
        f"{JSN_NAME} 2025\n{jsn['yards_per_team_pass_attempt']:.2f} -- 1st of {len(reference):,}",
        xy=(float(jsn["yards_per_team_pass_attempt"]), ax.get_ylim()[1] * 0.62),
        xytext=(-12, 0),
        textcoords="offset points",
        color=ACTION_GREEN,
        fontweight="bold",
        ha="right",
    )
    ax.set_xlabel("Receiving yards per team pass attempt")
    ax.set_ylabel("Receiver-seasons")
    ax.set_title(
        "Receiving yards per team pass attempt, 1999-2025\n"
        f"every receiver-season with {MIN_TARGETS}+ targets"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def volume_barrier_chart(barrier: dict, jsn: pd.Series, path: Path) -> None:
    """The seasons above him, and how much more they had to work with."""
    apply_scoreboard_style()
    rows = barrier["computed"]["seasons_above"]
    labels = [f"{r['season']} {r['receiver']}" for r in rows] + [f"2025 {JSN_NAME}"]
    attempts = [r["team_pass_attempts"] for r in rows] + [int(jsn["team_pass_attempts"])]
    yards = [r["receiving_yards"] for r in rows] + [int(jsn["receiving_yards"])]
    colors = [WOLF_GREY] * len(rows) + [ACTION_GREEN]

    fig, axes = plt.subplots(1, 2, figsize=fig_size(4.6), sharey=True)
    y = np.arange(len(labels))
    axes[0].barh(y, yards, color=colors)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Receiving yards")
    axes[0].set_title("What they gained", fontsize=11.5)
    for i, v in enumerate(yards):
        axes[0].text(v - 40, i, f"{v:,}", va="center", ha="right", fontsize=8.5)

    axes[1].barh(y, attempts, color=colors)
    axes[1].axvline(566, color=AMBER, linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Team pass attempts")
    axes[1].set_title("What they had to work with", fontsize=11.5)
    for i, v in enumerate(attempts):
        axes[1].text(v - 15, i, f"{v}", va="center", ha="right", fontsize=8.5)

    fig.suptitle(
        "Six receiver-seasons since 1999 beat 1,793 yards.\n"
        "None had fewer than 566 team pass attempts; Seattle threw 481."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def darnold_career_chart(career: pd.DataFrame, path: Path) -> None:
    apply_scoreboard_style()
    reg = career[career["season_type"] == "REG"].sort_values("season")
    post = career[
        (career["season"] == FOCUS_SEASON) & (career["season_type"] == "POST")
    ].iloc[0]

    fig, ax = plt.subplots(figsize=fig_size(5.2))
    x = np.arange(len(reg))
    rates = 100 * reg["turnover_rate_per_dropback"].to_numpy()
    colors = [ACTION_GREEN if s == FOCUS_SEASON else WOLF_GREY for s in reg["season"]]
    ax.bar(x, rates, color=colors)
    for i, (rate, row) in enumerate(zip(rates, reg.itertuples())):
        ax.text(i, rate + 0.06, f"{rate:.2f}", ha="center", fontsize=8.5)
        ax.text(i, -0.22, f"{row.turnovers:.0f} TO", ha="center", fontsize=8, color=WOLF_GREY)

    ax.bar([len(reg)], [0], color=ALERT_RED)
    # A zero bar cannot carry its own label, so the annotation sits above the
    # axis and is right-aligned to keep it inside the figure.
    ax.text(
        len(reg), 0.12,
        f"0.00\n0 turnovers in\n{post['dropbacks']:.0f} dropbacks",
        ha="right", color=ALERT_RED, fontweight="bold", fontsize=9,
    )
    ax.set_xlim(-0.7, len(reg) + 0.35)
    ax.set_xticks(list(x) + [len(reg)])
    ax.set_xticklabels(
        [f"{r.season}\n{r.team}" for r in reg.itertuples()] + ["2025\nPOST"], fontsize=8.5
    )
    ax.set_ylabel("Turnovers per 100 dropbacks")
    ax.set_ylim(-0.45, max(rates) * 1.28)
    ax.set_title(
        "Sam Darnold's turnover rate, every season\n"
        "2025 was the worst of his career -- and then he stopped entirely"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def backfield_chart(split: dict, path: Path) -> None:
    """Four paired bars: the touchdown gap next to the opportunity that caused it."""
    apply_scoreboard_style()
    backs = split["backs"]
    names = [b["player"] for b in backs]
    fig, axes = plt.subplots(1, 4, figsize=fig_size(4.0))

    panels = [
        ("carries", "Carries", "{:.0f}"),
        ("yards_per_carry", "Yards per carry", "{:.2f}"),
        ("rushing_tds", "Rushing touchdowns", "{:.0f}"),
        ("carries_inside_5", "Carries inside the 5", "{:.0f}"),
    ]
    colors = [WOLF_GREY, ACTION_GREEN]
    for ax, (key, title, fmt) in zip(axes, panels):
        values = [b[key] for b in backs]
        ax.bar(names, values, color=colors)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=8.5)
        for i, v in enumerate(values):
            ax.text(i, v * 0.5, fmt.format(v), ha="center", va="center",
                    fontweight="bold", fontsize=10)

    fig.suptitle(
        "Seattle's two backs in 2025\n"
        "the touchdown gap is a goal-line usage gap, not a finishing gap"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    receivers = load("receiver_season.csv")
    passers = load("passer_season.csv")
    rushers = load("rusher_season.csv")
    defenders = load("defender_season.csv")

    reference = receiver_reference(receivers)
    jsn = focus_receiver(reference)
    rankings = receiver_rankings(reference, jsn)
    barrier = volume_barrier(reference, jsn)

    career = darnold_career(passers)
    clean = clean_postseason_probability(career)
    arc = career_turnover_ranking(career)
    split = backfield_split(rushers)
    leap = year_two_leap(defenders)

    results = {
        "methodology": {
            "receiver_reference_set": len(reference),
            "receiver_reference_window": [1999, 2025],
            "min_targets": MIN_TARGETS,
            "season_type": "REG for all reference sets; postseason reported separately",
            "target_denominator": (
                "Team pass attempts, excluding sacks and two-point conversions. nflverse's "
                "pass_attempt flag includes both; leaving them in gives SEA 2025 510 attempts "
                "against the official 481 and deflates every share in the file."
            ),
            "routes_limitation": (
                "The honest denominator for opportunity is routes run, which is NOT in the "
                "play-by-play -- this cache carries no participation data. Target share over "
                "pass attempts conflates running a route with being on the field, and "
                "flatters a receiver who never leaves it."
            ),
            "no_probability_on_yardage": (
                "No probability is attached to the receiving season. Pass volume is jointly "
                "determined with the receiving production it would be conditioning on, and "
                "receiving yards violate the independence a count model would need. Rankings "
                "against a stated reference set are reported instead."
            ),
            "yards_over_expected_window": (
                f"Yards over expected uses air yards, completion probability and expected YAC, "
                f"none of which exist before {AIR_YARDS_FIRST_SEASON}, so it is ranked over "
                f"{AIR_YARDS_FIRST_SEASON}-2025 only. The underlying models are nflverse's."
            ),
            "external_context_not_computed": [
                "AP Offensive Player of the Year voting (272 points, 14 first-place votes)",
                "All-Pro and Pro Bowl selections",
                "Darnold's oblique injury and contract terms",
            ],
        },
        "smith_njigba": {
            "season": {
                "targets": int(jsn["targets"]),
                "receptions": int(jsn["receptions"]),
                "receiving_yards": int(jsn["receiving_yards"]),
                "receiving_tds": int(jsn["receiving_tds"]),
                "team_pass_attempts": int(jsn["team_pass_attempts"]),
                "games": int(jsn["games"]),
            },
            "rankings": rankings,
            "volume_barrier_claim": barrier,
        },
        "darnold": {
            "career": [
                {
                    "season": int(r.season),
                    "season_type": str(r.season_type),
                    "team": str(r.team),
                    "games": int(r.games),
                    "dropbacks": int(r.dropbacks),
                    "interceptions": int(r.interceptions),
                    "fumbles_lost": int(r.fumbles_lost),
                    "turnovers": int(r.turnovers),
                    "turnover_rate_per_dropback": round(float(r.turnover_rate_per_dropback), 5),
                }
                for r in career.itertuples()
            ],
            "clean_postseason": clean,
            "career_arc": arc,
        },
        "backfield": split,
        "year_two_leap": leap,
    }

    with open(OUT_DIR / "players_deep_dive.json", "w") as f:
        json.dump(results, f, indent=2)

    concentration_chart(reference, jsn, OUT_DIR / "players_jsn_concentration.png")
    volume_barrier_chart(barrier, jsn, OUT_DIR / "players_jsn_volume_barrier.png")
    darnold_career_chart(career, OUT_DIR / "players_darnold_career.png")
    backfield_chart(split, OUT_DIR / "players_backfield_split.png")

    print(f"Smith-Njigba 2025 vs {len(reference):,} receiver-seasons ({MIN_TARGETS}+ targets, 1999-2025):")
    for v in rankings.values():
        r, n = v["rank_of_n"]
        w = v["reference_window"]
        print(f"  {v['label']:<40} {v['value']:>9.3f}  rank {r:>4} of {n:,} ({w[0]}-{w[1]})")

    print(f"\nVolume barrier: {barrier['verdict']}")
    print(f"  {barrier['computed']['n_receiver_seasons_above']} seasons gained more; "
          f"fewest team pass attempts among them = "
          f"{barrier['computed']['min_team_pass_attempts_above']} vs SEA's "
          f"{barrier['computed']['sea_2025_team_pass_attempts']}")

    print("\nDarnold career turnover rate (per 100 dropbacks):")
    for r in career.itertuples():
        print(f"  {r.season} {r.season_type:<4} {r.team:<4} {r.dropbacks:>4.0f} db  "
              f"{r.turnovers:>3.0f} TO  {100 * r.turnover_rate_per_dropback:>5.2f}")
    print(f"  career-worst rate: {arc['worst_rate_season']} ({arc['worst_rate_team']}) "
          f"-> 2025 is career worst? {arc['focus_season_is_career_worst']}")
    print(f"  P(zero turnovers in {clean['playoff_dropbacks']} playoff dropbacks) = "
          f"{clean['p_zero_poisson']:.2%} (~1 in {clean['odds_against_1_in']:.0f})")

    print("\nSeattle's backfield:")
    for b in split["backs"]:
        print(f"  {b['player']:<14} {b['carries']:>4} car  {b['yards_per_carry']:.2f} ypc  "
              f"{b['rushing_tds']:>2} TD  |  inside the 5: {b['carries_inside_5']:>2} carries "
              f"({b['goal_line_carry_share']:.0%} of team), {b['tds_inside_5']} TD")
    r, n = split["touchdown_rank_of_n"]
    print(f"  {split['touchdown_leader']}'s {max(b['rushing_tds'] for b in split['backs'])} "
          f"rushing TDs rank {r} of {n:,} rusher-seasons")

    print(f"\nYear-two leap: {leap['player']} {leap['prior_sacks']:g} -> {leap['sacks']:g} sacks "
          f"({leap['jump']:+g}), rank {leap['rank_of_n'][0]} of {leap['rank_of_n'][1]:,} "
          f"consecutive-season pairs")

    for name in (
        "players_deep_dive.json",
        "players_jsn_concentration.png",
        "players_jsn_volume_barrier.png",
        "players_darnold_career.png",
        "players_backfield_split.png",
    ):
        print(f"Wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
