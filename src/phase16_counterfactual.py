"""Phase 16: how much of 14-3 was the team, and how much was the bounce?

The three 2025 losses came by four, three and two points -- nine points across an
entire season. That is the single most quoted fact about this team, and it is
usually offered as evidence of greatness: they were never really beaten. It is
equally evidence of the opposite: a team whose losses are all one-score games is
a team whose RECORD sat on a knife edge, and a couple of bounces the other way in
games they won would have produced a very different narrative.

This phase refuses to pick a side and instead quantifies the spread. Three
independent angles:

1. PYTHAGOREAN. Reuses src/pythagorean.py (Phase 2's model, NFL exponent 2.37).
   Points scored and allowed imply an expected win total; the gap between that
   and 14 is the part of the record that scoring margin does not explain.

2. MONTE CARLO SEASON. A simple, well-established margin model --
   expected margin = own rating - opponent rating + home-field advantage, where
   a team's rating is its season point differential per game -- is fit across all
   of 2025, its residual standard deviation is measured, and Seattle's actual
   17-game schedule is then replayed 20,000 times. The result is a distribution
   of records this exact team, playing this exact schedule, would plausibly
   produce.

   WHAT THIS MODEL IS NOT: it uses full-season ratings, so it is retrospective
   rather than predictive, and it treats each game as independent, ignoring
   injuries, weather and midseason improvement. Its job is to bound how much
   variance a 17-game sample carries, not to forecast anything. It recovers a
   residual SD of about 11.5 points, slightly tighter than the ~13 usually quoted
   for NFL game margins -- expected, because the ratings are fit on the same
   season they are used to explain, so some of the noise has been absorbed. That
   makes the win range below, if anything, a little too narrow.

3. TURNOVER LUCK. Fumble RECOVERY is close to a coin flip -- which team falls on
   a loose ball is largely not a skill -- while forcing the fumble is not. So
   Seattle's fumble-recovery share is compared to the league rate and resampled
   at 50/50 to bound how much of its turnover margin was bounce.

   Note the finding this sits next to: Seattle's full-season turnover margin was
   -1 (25 takeaways, 26 giveaways). The "they fixed the turnovers" story is a
   story about the END of the season, not the whole of it, and Phase 6 already
   showed the within-season trend. A team does not need a positive turnover
   margin to win 14 games, and this one did not have one.

HONEST FRAMING. None of this argues the 2025 Seahawks were lucky. A +191 point
differential is not luck, and the Monte Carlo is built from that differential. The
question is narrower: given how good they were, how wide was the range of records
that goodness could have produced?
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
)
from pythagorean import NFL_EXPONENT, expected_wins
from season_metrics import RELOCATION_MAP

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

FOCUS_TEAM = "SEA"
FOCUS_SEASON = 2025
N_SIMS = 20_000
RANDOM_STATE = 42
GROUND_TRUTH_WINS = 14


def load_advanced() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_season_advanced.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run src/season_metrics.py first")
    return pd.read_csv(path)


def load_games() -> pd.DataFrame:
    sched = pd.read_parquet(RAW_DIR / "schedules.parquet")
    g = sched[(sched["season"] == FOCUS_SEASON) & (sched["game_type"] == "REG")].dropna(
        subset=["home_score", "away_score"]
    ).copy()
    for col in ("home_team", "away_team"):
        g[col] = g[col].replace(RELOCATION_MAP)
    g["home_margin"] = g["home_score"] - g["away_score"]
    return g


def fit_margin_model(games: pd.DataFrame, ratings: dict[str, float]) -> dict:
    """expected home margin = home rating - away rating + HFA; measure the residual."""
    exp_margin = (
        games["home_team"].map(ratings).to_numpy()
        - games["away_team"].map(ratings).to_numpy()
    )
    actual = games["home_margin"].to_numpy()
    hfa = float(np.mean(actual - exp_margin))
    resid = actual - (exp_margin + hfa)
    return {
        "home_field_advantage": round(hfa, 3),
        "residual_sd": round(float(np.std(resid, ddof=1)), 3),
        "n_games": len(games),
        "mean_abs_error": round(float(np.mean(np.abs(resid))), 3),
    }


def simulate_season(
    sea_games: pd.DataFrame, ratings: dict[str, float], model: dict, rng: np.random.Generator
) -> np.ndarray:
    """Replay SEA's schedule N_SIMS times, returning the win total each time."""
    exp_margin = []
    for r in sea_games.itertuples():
        own, opp = ratings[FOCUS_TEAM], ratings[r.opponent]
        hfa = model["home_field_advantage"] * (1 if r.at_home else -1)
        exp_margin.append(own - opp + hfa)
    exp_margin = np.array(exp_margin)

    draws = rng.normal(
        loc=exp_margin[None, :], scale=model["residual_sd"], size=(N_SIMS, len(exp_margin))
    )
    # A simulated margin of exactly 0 is a tie, worth half a win. It is vanishingly
    # rare on a continuous draw but handled rather than silently counted as a loss.
    return (draws > 0).sum(axis=1) + 0.5 * (draws == 0).sum(axis=1)


def sea_schedule(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in games.itertuples():
        if r.home_team == FOCUS_TEAM:
            rows.append({"week": r.week, "opponent": r.away_team, "at_home": True, "margin": r.home_margin})
        elif r.away_team == FOCUS_TEAM:
            rows.append({"week": r.week, "opponent": r.home_team, "at_home": False, "margin": -r.home_margin})
    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)


def one_score_analysis(sched: pd.DataFrame) -> dict:
    one_score = sched[sched["margin"].abs() <= 8]
    losses = sched[sched["margin"] < 0]
    return {
        "losses": [
            {"week": int(r.week), "opponent": r.opponent, "margin": int(r.margin)}
            for r in losses.itertuples()
        ],
        "total_points_lost_by": int(-losses["margin"].sum()),
        "one_score_games": len(one_score),
        "one_score_record": f"{int((one_score['margin'] > 0).sum())}-{int((one_score['margin'] < 0).sum())}",
        "blowout_wins_3plus_scores": int((sched["margin"] >= 17).sum()),
        "note": (
            "A one-score game is decided by 8 points or fewer. Going well in these is "
            "weakly predictive year over year, which is why a strong one-score record "
            "is usually read as partly variance."
        ),
    }


def turnover_luck(advanced: pd.DataFrame) -> dict:
    """Bound the fumble-recovery component of the turnover margin."""
    pbp = pd.read_parquet(
        RAW_DIR / "pbp" / f"{FOCUS_SEASON}.parquet",
        columns=["season_type", "posteam", "defteam", "fumble", "fumble_lost", "interception"],
    )
    reg = pbp[pbp["season_type"] == "REG"]
    for col in ("posteam", "defteam"):
        reg = reg.assign(**{col: reg[col].replace(RELOCATION_MAP)})

    league_fumbles = int((reg["fumble"] == 1).sum())
    league_lost = int((reg["fumble_lost"] == 1).sum())
    league_recovery_rate = league_lost / league_fumbles if league_fumbles else float("nan")

    sea_off = reg[reg["posteam"] == FOCUS_TEAM]
    sea_def = reg[reg["defteam"] == FOCUS_TEAM]
    own_fumbles = int((sea_off["fumble"] == 1).sum())
    own_lost = int((sea_off["fumble_lost"] == 1).sum())
    forced_fumbles = int((sea_def["fumble"] == 1).sum())
    recovered = int((sea_def["fumble_lost"] == 1).sum())

    sea = advanced[
        (advanced["team"] == FOCUS_TEAM)
        & (advanced["season"] == FOCUS_SEASON)
        & (advanced["season_type"] == "REG")
    ].iloc[0]

    # What the turnover margin would have been if every loose ball had gone at the
    # league's own observed rate instead of the way it actually bounced.
    neutral_own_lost = own_fumbles * league_recovery_rate
    neutral_recovered = forced_fumbles * league_recovery_rate
    neutral_margin = (
        (int(sea["takeaways"]) - recovered + neutral_recovered)
        - (int(sea["giveaways"]) - own_lost + neutral_own_lost)
    )

    return {
        "league_fumble_recovery_rate_by_defense": round(league_recovery_rate, 3),
        "sea_own_fumbles": own_fumbles,
        "sea_own_fumbles_lost": own_lost,
        "sea_own_fumble_loss_rate": round(own_lost / own_fumbles, 3) if own_fumbles else None,
        "sea_forced_fumbles": forced_fumbles,
        "sea_forced_fumbles_recovered": recovered,
        "sea_forced_fumble_recovery_rate": round(recovered / forced_fumbles, 3) if forced_fumbles else None,
        "actual_turnover_margin": int(sea["turnover_margin"]),
        "turnover_margin_at_league_recovery_rate": round(float(neutral_margin), 2),
        "bounce_component": round(float(int(sea["turnover_margin"]) - neutral_margin), 2),
        "note": (
            "Forcing a fumble is a skill; recovering one is close to a coin flip. This "
            "holds the recovery share at the league rate and leaves the forcing alone."
        ),
    }


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def record_distribution_chart(sim_wins: np.ndarray, path: Path) -> None:
    apply_scoreboard_style()
    counts = pd.Series(sim_wins).value_counts().sort_index()
    pct = 100 * counts / counts.sum()
    colors = [ACTION_GREEN if w == GROUND_TRUTH_WINS else WOLF_GREY for w in counts.index]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(counts.index, pct, color=colors, width=0.75)
    ax.axvline(np.mean(sim_wins), color=AMBER, ls="--", lw=1.6,
               label=f"Simulated mean {np.mean(sim_wins):.1f} wins")
    ax.axvline(GROUND_TRUTH_WINS, color=ACTION_GREEN, lw=1.6, label="Actual: 14 wins")
    ax.set_xlabel("Regular-season wins")
    ax.set_ylabel("% of simulated seasons")
    ax.set_title(
        f"Replaying SEA's 2025 schedule {N_SIMS:,} times\n"
        "Same team, same opponents, only the bounces re-rolled"
    )
    ax.legend(fontsize=8, framealpha=0.2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def close_games_chart(sched: pd.DataFrame, path: Path) -> None:
    apply_scoreboard_style()
    df = sched.sort_values("week")
    colors = [ACTION_GREEN if m > 0 else ALERT_RED for m in df["margin"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(df["week"], df["margin"], color=colors)
    ax.axhspan(-8, 8, color=AMBER, alpha=0.12, zorder=0)
    ax.axhline(0, color=OFF_WHITE, lw=0.9)

    for bar, r in zip(bars, df.itertuples()):
        if r.margin < 0:
            ax.text(bar.get_x() + bar.get_width() / 2, r.margin - 1.6,
                    f"{r.opponent}\n{int(r.margin)}", ha="center", va="top", fontsize=7.5)

    ax.set_xlabel("Week")
    ax.set_ylabel("Final margin")
    ax.set_title(
        "SEA 2025 game margins -- the shaded band is one-score territory\n"
        "All three losses were one-score games, nine points in total"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    advanced = load_advanced()
    reg = advanced[advanced["season_type"] == "REG"]
    games = load_games()
    sched = sea_schedule(games)
    assert len(sched) == 17, f"expected 17 SEA regular-season games, got {len(sched)}"
    assert int((sched["margin"] > 0).sum()) == GROUND_TRUTH_WINS, "SEA 2025 win count mismatch"

    sea = reg[(reg["team"] == FOCUS_TEAM) & (reg["season"] == FOCUS_SEASON)].iloc[0]

    pyth = expected_wins(float(sea["points_for"]), float(sea["points_against"]), 17)
    ratings = (
        reg[reg["season"] == FOCUS_SEASON].set_index("team")["point_diff_per_game"].to_dict()
    )
    model = fit_margin_model(games, ratings)

    rng = np.random.default_rng(RANDOM_STATE)
    sim_wins = simulate_season(sched, ratings, model, rng)

    pct_at_least_14 = float(100 * np.mean(sim_wins >= GROUND_TRUTH_WINS))
    lo, hi = np.percentile(sim_wins, [5, 95])

    OUT_DIR.mkdir(exist_ok=True)
    results = {
        "methodology": {
            "pythagorean_exponent": NFL_EXPONENT,
            "n_simulations": N_SIMS,
            "random_state": RANDOM_STATE,
            "margin_model": "expected margin = own season point differential/game - opponent's + home-field advantage",
            "margin_model_fit": model,
            "what_this_is_not": (
                "Retrospective, not predictive: ratings are full-season, so the model already "
                "knows how the season went. Games are treated as independent, ignoring injuries, "
                "weather and in-season improvement. It bounds the variance in a 17-game sample; "
                "it does not forecast."
            ),
        },
        "pythagorean": {
            "points_for": int(sea["points_for"]),
            "points_against": int(sea["points_against"]),
            "expected_wins": round(float(pyth), 2),
            "actual_wins": GROUND_TRUTH_WINS,
            "wins_over_expectation": round(GROUND_TRUTH_WINS - float(pyth), 2),
        },
        "monte_carlo": {
            "mean_wins": round(float(np.mean(sim_wins)), 2),
            "median_wins": float(np.median(sim_wins)),
            "sd_wins": round(float(np.std(sim_wins, ddof=1)), 2),
            "p05_wins": float(lo),
            "p95_wins": float(hi),
            "pct_of_seasons_at_least_14_wins": round(pct_at_least_14, 1),
            "pct_of_seasons_at_most_11_wins": round(float(100 * np.mean(sim_wins <= 11)), 1),
            "win_distribution": {
                str(int(w)): round(float(100 * np.mean(sim_wins == w)), 2)
                for w in range(18)
                if np.any(sim_wins == w)
            },
        },
        "close_games": one_score_analysis(sched),
        "turnover_luck": turnover_luck(advanced),
    }
    with open(OUT_DIR / "counterfactual.json", "w") as f:
        json.dump(results, f, indent=2)

    record_distribution_chart(sim_wins, OUT_DIR / "counterfactual_record_distribution.png")
    close_games_chart(sched, OUT_DIR / "counterfactual_close_games.png")

    p = results["pythagorean"]
    print(f"Pythagorean: {p['expected_wins']:.2f} expected wins vs {p['actual_wins']} actual "
          f"({p['wins_over_expectation']:+.2f})")
    print(f"Margin model: HFA {model['home_field_advantage']:+.2f} pts, "
          f"residual SD {model['residual_sd']:.2f} pts over {model['n_games']} games")

    mc = results["monte_carlo"]
    print(f"\nMonte Carlo ({N_SIMS:,} seasons): mean {mc['mean_wins']:.2f} wins, "
          f"90% range {mc['p05_wins']:.0f}-{mc['p95_wins']:.0f}")
    print(f"  {mc['pct_of_seasons_at_least_14_wins']}% of simulated seasons reach 14+ wins")
    print(f"  {mc['pct_of_seasons_at_most_11_wins']}% finish at 11 wins or fewer")

    cg = results["close_games"]
    print(f"\nOne-score games: {cg['one_score_record']} in {cg['one_score_games']} games; "
          f"all {len(cg['losses'])} losses by {cg['total_points_lost_by']} points combined")

    tl = results["turnover_luck"]
    print(f"\nTurnover margin {tl['actual_turnover_margin']:+d} actual vs "
          f"{tl['turnover_margin_at_league_recovery_rate']:+.1f} at league fumble-recovery rate "
          f"(bounce component {tl['bounce_component']:+.1f})")

    for name in ("counterfactual.json", "counterfactual_record_distribution.png",
                 "counterfactual_close_games.png"):
        print(f"Wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
