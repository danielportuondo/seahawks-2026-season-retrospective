"""Phase 6: how surprising was Sam Darnold's zero-turnover 2025-26 playoff run?

Darnold led the NFL with 20 turnovers in the 2025 regular season (14 INT + 6
lost fumbles over 17 games), then committed zero across three playoff wins
(Divisional 41-6 over SF, NFC Championship 31-27 over LA, Super Bowl LX 29-13
over NE). This phase puts a probability on that.

FRAMING: THIS IS A DISCRETE RATE-MODEL PROBLEM, NOT EXTREME VALUE THEORY.
We are counting a small number of discrete events (turnovers) over a bounded
number of trials (dropbacks / drives), so the right tool is a count
distribution -- Poisson or negative binomial -- fit to an observed rate. EVT
(GEV / generalized Pareto / block maxima) models the tail of a CONTINUOUS
magnitude distribution: "what is the largest single-game passing yardage we
would ever expect to see." Turnovers have no magnitude tail to model; the
outcome of interest is literally the count k=0. Reaching for EVT here would
be a category error, and a zero is a minimum, not a maximum, so even the
block-maxima intuition points the wrong way.

ATTRIBUTION (the part most likely to go subtly wrong):
- Interceptions: `interception == 1` on rows where Darnold is the passer. This
  is clean and reproduces the ground-truth 14.
- Lost fumbles: `fumbled_1_player_id == <Darnold>` AND `fumble_lost == 1`.
  Filtering on `fumble_lost == 1` across his dropback rows instead returns 8,
  because three of those are receivers fumbling AFTER a completed catch
  (Smith-Njigba wk 1, Arroyo wk 7, Kupp wk 16) -- the QB's row is charged with
  the play's fumble flag even though the ball security failure was not his.
  Keying on the fumbler's ID returns exactly the ground-truth 6 (5 sack
  fumbles + 1 aborted snap), same "right in aggregate, wrong per-row" trap
  Phase 4 hit.
- `validate_weekly()` asserts the weekly breakdown re-sums to 14 / 6 / 20
  against `player_stats_2024_2025_reg.parquet` before anything downstream
  runs. If attribution drifts, the script fails loudly instead of modelling
  a wrong rate.

DROPBACK DENOMINATOR:
`passer_player_name == 'S.Darnold'` matches 506 regular-season rows, but that
is passer rows, not dropbacks: 502 are real dropbacks (passes + sacks) and 4
are clock-stopping spikes, which nflverse flags `qb_dropback == 0`. Scrambles
are the mirror-image problem -- they ARE dropbacks (14 of them) but carry a
null passer and name Darnold as the rusher. The honest QB dropback count is
therefore 502 + 14 = 516, and we key on `passer_player_id OR rusher_player_id`
with `qb_dropback == 1`. The 506 figure is preserved in the output JSON as a
reconciliation field so the discrepancy is documented rather than buried.

RATE DEFINITIONS:
- Per dropback: 20 / 516. One of the 20 (the wk-10 aborted snap) is coded as a
  run and sits outside the dropback set, so a strict "turnovers occurring on
  dropbacks" rate of 19/516 is reported alongside as a sensitivity. We keep
  all 20 in the headline because a botched exchange is a genuine QB
  ball-security event, not a third party's error.
- Per drive: 20 / 193 SEA regular-season offensive drives (`fixed_drive`).
  Darnold started all 17 games; the handful of snaps taken by the backup are
  absorbed here, which is a rounding-level approximation, not a distortion.

POISSON vs NEGATIVE BINOMIAL:
Checked rather than assumed. We report the raw variance-to-mean ratio of the
17 weekly counts, the exposure-adjusted Pearson dispersion statistic (which
is the one that matters, since weekly dropback volume varies a lot -- 13 in
wk 10 vs 45 in wk 11), and an NB2 fit by maximum likelihood. Poisson is used
when the dispersion test does not reject and the NB2 likelihood-ratio test
(boundary-corrected, since alpha=0 sits on the edge of the parameter space)
does not either. The NB result is still carried into the output as a
conservative bound, because overdispersion always raises P(zero).

PLAYOFF EXPOSURE SIZING:
Derived from Darnold's own data, not picked: three games at his own
regular-season per-game average. Note this makes the answer invariant to the
choice of exposure unit -- lambda = rate_per_unit * (units_per_game * 3)
collapses to 3 * (20/17) = 3.53 expected turnovers whether the unit is a
dropback or a drive, because the denominator cancels. His ACTUAL playoff
volume (99 dropbacks, 34 drives) is reported as a sensitivity; it was higher
than his season average, which makes the zero MORE surprising, not less.

STATIONARITY:
The narrative that Darnold "got safer with the ball down the stretch" is
tested, not assumed, via a log-linear Poisson trend model with a dropback
offset (LRT against intercept-only) and a first-half/second-half split test.
The full-season rate is used for the headline model only because those tests
fail to reject; first-half-only and second-half-only rates are reported as a
regime sensitivity band.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize, special, stats

from chart_style import (
    ACTION_GREEN,
    ALERT_RED,
    AMBER,
    OFF_WHITE,
    PANEL,
    WOLF_GREY,
    apply_scoreboard_style,
    fig_size,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"

SEASON = 2025
TEAM = "SEA"
QB_NAME = "Sam Darnold"
QB_ID = "00-0034869"
PLAYOFF_GAMES = 3
ROLL_WINDOW = 4

GROUND_TRUTH = {"interceptions": 14, "fumbles_lost": 6, "turnovers": 20, "games": 17}


def load_pbp(season: int = SEASON) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "pbp" / f"{season}.parquet")


def season_total_ground_truth() -> dict:
    """Phase 3's validated season-total row: one row per player, no week column."""
    df = pd.read_parquet(RAW_DIR / "player_stats_2024_2025_reg.parquet")
    row = df[(df["season"] == SEASON) & (df["player_display_name"] == QB_NAME)].iloc[0]
    fumbles_lost = int(row["sack_fumbles_lost"]) + int(row["rushing_fumbles_lost"])
    return {
        "interceptions": int(row["passing_interceptions"]),
        "sack_fumbles_lost": int(row["sack_fumbles_lost"]),
        "rushing_fumbles_lost": int(row["rushing_fumbles_lost"]),
        "fumbles_lost": fumbles_lost,
        "turnovers": int(row["passing_interceptions"]) + fumbles_lost,
        "games": int(row["games"]),
    }


def qb_plays(pbp: pd.DataFrame, season_type: str) -> pd.DataFrame:
    """Every snap Darnold was the primary offensive actor on (passer or rusher)."""
    off = pbp[(pbp["season_type"] == season_type) & (pbp["posteam"] == TEAM)]
    return off[(off["passer_player_id"] == QB_ID) | (off["rusher_player_id"] == QB_ID)]


def weekly_table(pbp: pd.DataFrame, season_type: str = "REG") -> pd.DataFrame:
    plays = qb_plays(pbp, season_type)
    off = pbp[(pbp["season_type"] == season_type) & (pbp["posteam"] == TEAM)]

    dropbacks = plays[plays["qb_dropback"] == 1].groupby("week").size()
    ints = plays[plays["interception"] == 1].groupby("week").size()
    fumbles = plays[
        (plays["fumbled_1_player_id"] == QB_ID) & (plays["fumble_lost"] == 1)
    ].groupby("week").size()
    drives = off.groupby("week")["fixed_drive"].nunique()

    weekly = pd.concat(
        [
            dropbacks.rename("dropbacks"),
            ints.rename("interceptions"),
            fumbles.rename("fumbles_lost"),
            drives.rename("drives"),
        ],
        axis=1,
    ).fillna(0).astype(int)

    weekly["turnovers"] = weekly["interceptions"] + weekly["fumbles_lost"]
    # int64 rather than nflverse's int32 so the weekly records stay json-serialisable
    weekly = weekly.sort_index().reset_index().astype("int64")
    # Week 8 is the bye; game_no keeps the x-axis in games played, not calendar
    # weeks, so the trend model is not fitting a phantom zero-turnover week.
    weekly["game_no"] = np.arange(1, len(weekly) + 1)
    return weekly


def validate_weekly(weekly: pd.DataFrame, truth: dict) -> None:
    got = {
        "interceptions": int(weekly["interceptions"].sum()),
        "fumbles_lost": int(weekly["fumbles_lost"].sum()),
        "turnovers": int(weekly["turnovers"].sum()),
        "games": len(weekly),
    }
    for key, expected in GROUND_TRUTH.items():
        if got[key] != expected:
            raise AssertionError(f"weekly {key}={got[key]}, expected {expected}")
        if truth[key] != expected:
            raise AssertionError(f"season-total {key}={truth[key]}, expected {expected}")
    print(f"  validated: {got['interceptions']} INT + {got['fumbles_lost']} lost fumbles "
          f"= {got['turnovers']} turnovers over {got['games']} games (weekly == season total)")


def dropback_reconciliation(pbp: pd.DataFrame) -> dict:
    """Document why the naive 506 and the modelled 516 differ."""
    off = pbp[(pbp["season_type"] == "REG") & (pbp["posteam"] == TEAM)]
    passer_rows = off[off["passer_player_id"] == QB_ID]
    passer_dropbacks = int((passer_rows["qb_dropback"] == 1).sum())
    spikes = int((passer_rows["play_type"] == "qb_spike").sum())
    scrambles = int(
        ((off["rusher_player_id"] == QB_ID) & (off["qb_dropback"] == 1)).sum()
    )
    return {
        "passer_rows": len(passer_rows),
        "passer_rows_that_are_dropbacks": passer_dropbacks,
        "qb_spikes_excluded": spikes,
        "scrambles_added": scrambles,
        "qb_dropbacks_modelled": passer_dropbacks + scrambles,
        "note": (
            "passer_player_name == 'S.Darnold' matches 506 REG rows, but 4 are "
            "qb_spikes (qb_dropback == 0). Scrambles are dropbacks with a null "
            "passer and Darnold as rusher. 502 passer dropbacks + 14 scrambles "
            "= 516 true qb_dropbacks, which is the denominator used here."
        ),
    }


def season_rates(weekly: pd.DataFrame) -> dict:
    turnovers = int(weekly["turnovers"].sum())
    dropbacks = int(weekly["dropbacks"].sum())
    drives = int(weekly["drives"].sum())
    games = len(weekly)
    return {
        "games": games,
        "turnovers": turnovers,
        "interceptions": int(weekly["interceptions"].sum()),
        "fumbles_lost": int(weekly["fumbles_lost"].sum()),
        "dropbacks": dropbacks,
        "drives": drives,
        "turnover_rate_per_dropback": turnovers / dropbacks,
        "turnover_rate_per_drive": turnovers / drives,
        "turnovers_per_game": turnovers / games,
        "dropbacks_per_game": dropbacks / games,
        "drives_per_game": drives / games,
    }


def rolling_rates(weekly: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    out = weekly.copy()
    # Exposure-weighted: sum(turnovers)/sum(dropbacks) over the window, NOT the
    # mean of weekly rates -- a 13-dropback game must not count as much as a
    # 45-dropback one.
    roll_to = out["turnovers"].rolling(window).sum()
    roll_db = out["dropbacks"].rolling(window).sum()
    out["rolling_rate_per_dropback"] = roll_to / roll_db
    out["cumulative_rate_per_dropback"] = (
        out["turnovers"].cumsum() / out["dropbacks"].cumsum()
    )
    out["rolling_turnovers_per_game"] = out["turnovers"].rolling(window).mean()
    return out


def _poisson_nll(params: np.ndarray, y: np.ndarray, t: np.ndarray, expo: np.ndarray) -> float:
    mu = np.exp(params[0] + params[1] * t) * expo
    return -np.sum(y * np.log(mu) - mu - special.gammaln(y + 1))


def trend_test(weekly: pd.DataFrame) -> dict:
    """Did he actually get safer? Log-linear Poisson trend with dropback offset."""
    y = weekly["turnovers"].to_numpy(float)
    expo = weekly["dropbacks"].to_numpy(float)
    t = weekly["game_no"].to_numpy(float)
    t = t - t.mean()

    rate = y.sum() / expo.sum()
    ll_null = -_poisson_nll(np.array([np.log(rate), 0.0]), y, t, expo)
    fit = optimize.minimize(
        _poisson_nll, x0=[np.log(rate), 0.0], args=(y, t, expo), method="BFGS"
    )
    ll_full = -fit.fun
    lr = 2 * (ll_full - ll_null)
    p_trend = float(stats.chi2.sf(lr, df=1))
    slope = float(fit.x[1])

    half = len(weekly) // 2
    first, second = weekly.iloc[:half], weekly.iloc[half:]
    to1, to2 = int(first["turnovers"].sum()), int(second["turnovers"].sum())
    db1, db2 = int(first["dropbacks"].sum()), int(second["dropbacks"].sum())
    # Conditional on the season total, first-half turnovers ~ Binomial(20, share
    # of dropbacks in the first half) under a constant rate.
    p_split = float(
        stats.binomtest(to1, to1 + to2, db1 / (db1 + db2)).pvalue
    )

    return {
        "log_linear_slope_per_game": slope,
        "rate_multiplier_per_game": float(np.exp(slope)),
        "rate_multiplier_over_season": float(np.exp(slope * (len(weekly) - 1))),
        "likelihood_ratio_stat": float(lr),
        "p_value_trend": p_trend,
        "first_half_games": len(first),
        "first_half_turnovers": to1,
        "first_half_dropbacks": db1,
        "first_half_rate_per_dropback": to1 / db1,
        "first_half_turnovers_per_game": to1 / len(first),
        "second_half_games": len(second),
        "second_half_turnovers": to2,
        "second_half_dropbacks": db2,
        "second_half_rate_per_dropback": to2 / db2,
        "second_half_turnovers_per_game": to2 / len(second),
        "p_value_half_split": p_split,
        "direction": "worse in second half" if (to2 / db2) > (to1 / db1) else "better in second half",
        "narrative_supported": bool((to2 / db2) < (to1 / db1) and p_split < 0.05),
    }


def _nb2_nll(params: np.ndarray, y: np.ndarray, expo: np.ndarray) -> float:
    log_rate, log_alpha = params
    alpha = np.exp(log_alpha)
    mu = np.exp(log_rate) * expo
    r = 1.0 / alpha
    ll = (
        special.gammaln(y + r)
        - special.gammaln(r)
        - special.gammaln(y + 1)
        + r * np.log(r / (r + mu))
        + y * np.log(mu / (r + mu))
    )
    return -np.sum(ll)


def dispersion_diagnostics(weekly: pd.DataFrame) -> dict:
    y = weekly["turnovers"].to_numpy(float)
    expo = weekly["dropbacks"].to_numpy(float)
    n = len(y)
    rate = y.sum() / expo.sum()
    mu = rate * expo

    raw_vmr = float(y.var(ddof=1) / y.mean())

    # Exposure-adjusted Pearson dispersion: the test that actually applies here,
    # since weekly dropback volume ranges 13-45 and unequal exposure alone
    # inflates the raw variance-to-mean ratio.
    pearson_chi2 = float(np.sum((y - mu) ** 2 / mu))
    df = n - 1
    dispersion = pearson_chi2 / df
    p_dispersion = float(stats.chi2.sf(pearson_chi2, df))

    ll_pois = -_poisson_nll(np.array([np.log(rate), 0.0]), y, np.zeros(n), expo)
    nb = optimize.minimize(
        _nb2_nll, x0=[np.log(rate), np.log(0.5)], args=(y, expo), method="Nelder-Mead"
    )
    ll_nb = -nb.fun
    alpha = float(np.exp(nb.x[1]))
    nb_rate = float(np.exp(nb.x[0]))
    lr = max(0.0, 2 * (ll_nb - ll_pois))
    # alpha = 0 is on the boundary of the parameter space, so the LRT null is a
    # 50:50 mixture of chi2(0) and chi2(1), not chi2(1).
    p_nb = float(0.5 * stats.chi2.sf(lr, df=1))

    use_poisson = (p_dispersion > 0.05) and (p_nb > 0.05)
    return {
        "raw_variance_to_mean_ratio": raw_vmr,
        "weekly_mean": float(y.mean()),
        "weekly_variance": float(y.var(ddof=1)),
        "pearson_chi2": pearson_chi2,
        "pearson_df": int(df),
        "exposure_adjusted_dispersion": float(dispersion),
        "p_value_overdispersion": p_dispersion,
        "nb2_alpha": alpha,
        "nb2_rate_per_dropback": nb_rate,
        "loglik_poisson": float(ll_pois),
        "loglik_nb2": float(ll_nb),
        "aic_poisson": float(2 * 1 - 2 * ll_pois),
        "aic_nb2": float(2 * 2 - 2 * ll_nb),
        "nb2_lr_stat": float(lr),
        "p_value_nb2_vs_poisson": p_nb,
        "chosen_distribution": "poisson" if use_poisson else "negative_binomial",
        "reason": (
            "Exposure-adjusted dispersion is not significantly above 1 and the "
            "boundary-corrected LRT does not prefer NB2, so the extra dispersion "
            "parameter is not earned. NB2 results retained as a conservative bound."
            if use_poisson
            else "Counts are overdispersed relative to Poisson; NB2 used."
        ),
    }


def playoff_exposure(rates: dict, n_games: int = PLAYOFF_GAMES) -> dict:
    return {
        "games": n_games,
        "dropbacks": rates["dropbacks_per_game"] * n_games,
        "drives": rates["drives_per_game"] * n_games,
        "basis": (
            f"{n_games} games at Darnold's own 2025 regular-season average of "
            f"{rates['dropbacks_per_game']:.2f} dropbacks / "
            f"{rates['drives_per_game']:.2f} drives per game"
        ),
    }


def playoff_actuals(pbp: pd.DataFrame) -> dict:
    plays = qb_plays(pbp, "POST")
    off = pbp[(pbp["season_type"] == "POST") & (pbp["posteam"] == TEAM)]
    ints = int((plays["interception"] == 1).sum())
    fumbles = int(
        ((plays["fumbled_1_player_id"] == QB_ID) & (plays["fumble_lost"] == 1)).sum()
    )
    return {
        "games": int(off["week"].nunique()),
        "dropbacks": int((plays["qb_dropback"] == 1).sum()),
        "drives": int(off.groupby("week")["fixed_drive"].nunique().sum()),
        "interceptions": ints,
        "fumbles_lost": fumbles,
        "turnovers": ints + fumbles,
    }


def zero_turnover_probability(rate: float, exposure: float, alpha: float | None = None) -> dict:
    lam = rate * exposure
    p_zero_pois = float(np.exp(-lam))
    out = {
        "expected_turnovers": float(lam),
        "p_zero_poisson": p_zero_pois,
        "odds_against_1_in": float(1 / p_zero_pois),
    }
    if alpha is not None and alpha > 0:
        r = 1.0 / alpha
        p_zero_nb = float((r / (r + lam)) ** r)
        out["p_zero_negative_binomial"] = p_zero_nb
        out["odds_against_nb_1_in"] = float(1 / p_zero_nb)
    return out


def plot_rolling(rolling: pd.DataFrame, rates: dict, trend: dict, path: Path) -> None:
    apply_scoreboard_style()
    fig, axes = plt.subplots(2, 1, figsize=fig_size(7.0), sharex=True)
    x = rolling["game_no"]
    labels = [f"W{int(w)}" for w in rolling["week"]]

    ax = axes[0]
    ax.bar(x, rolling["interceptions"], color=ALERT_RED, label="interceptions")
    ax.bar(x, rolling["fumbles_lost"], bottom=rolling["interceptions"],
           color=AMBER, label="lost fumbles")
    ax.axhline(rates["turnovers_per_game"], color=OFF_WHITE, ls="--", lw=1.2,
               label=f"season mean = {rates['turnovers_per_game']:.2f}/game")
    ax.set_ylabel("Turnovers")
    ax.set_yticks(range(int(rolling["turnovers"].max()) + 2))
    ax.set_title(f"Sam Darnold 2025 regular season: {rates['turnovers']} turnovers "
                 f"({rates['interceptions']} INT + {rates['fumbles_lost']} lost fumbles), NFL high")
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ax.plot(x, 100 * rolling["rolling_rate_per_dropback"], "o-", color=ALERT_RED,
            lw=2, label=f"trailing {ROLL_WINDOW}-game rate (exposure-weighted)")
    ax.plot(x, 100 * rolling["cumulative_rate_per_dropback"], "s--", color=WOLF_GREY,
            lw=1.6, ms=4, alpha=0.9, label="cumulative to date")
    ax.axhline(100 * rates["turnover_rate_per_dropback"], color=OFF_WHITE, ls=":", lw=1.2,
               label=f"full season = {100 * rates['turnover_rate_per_dropback']:.2f}%")

    half = len(rolling) // 2
    ax.hlines(100 * trend["first_half_rate_per_dropback"], 1, half, color=ACTION_GREEN,
              lw=3, alpha=0.55)
    ax.hlines(100 * trend["second_half_rate_per_dropback"], half + 1, len(rolling),
              color=AMBER, lw=3, alpha=0.55)
    ax.text(2.2, 100 * trend["first_half_rate_per_dropback"] + 0.3,
            f"games 1-{half}: {100 * trend['first_half_rate_per_dropback']:.2f}%",
            color=ACTION_GREEN, fontsize=9, ha="center", fontweight="bold")
    ax.text(len(rolling) - 1.5, 100 * trend["second_half_rate_per_dropback"] + 1.0,
            f"games {half + 1}-{len(rolling)}: "
            f"{100 * trend['second_half_rate_per_dropback']:.2f}%",
            color=AMBER, fontsize=9, ha="center", fontweight="bold")

    verdict = ("NARRATIVE NOT SUPPORTED" if not trend["narrative_supported"]
               else "narrative supported")
    ax.text(0.99, 0.96,
            f"'Got safer down the stretch': {verdict}\n"
            f"Poisson trend slope p = {trend['p_value_trend']:.2f}  |  "
            f"half-split p = {trend['p_value_half_split']:.2f}\n"
            f"Point estimate runs the other way ({trend['direction']}).",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=OFF_WHITE,
            bbox={"boxstyle": "round", "fc": PANEL, "ec": ALERT_RED, "alpha": 0.95})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Game (week label; week 8 bye omitted)")
    ax.set_ylabel("Turnover rate per dropback (%)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_probability(lam: float, alpha: float, result: dict, exposure: dict, path: Path) -> None:
    apply_scoreboard_style()
    k = np.arange(0, 11)
    pmf = stats.poisson.pmf(k, lam)
    r = 1.0 / alpha
    pmf_nb = stats.nbinom.pmf(k, r, r / (r + lam))

    fig, ax = plt.subplots(figsize=fig_size(5.6))
    colors = [ACTION_GREEN] + [WOLF_GREY] * (len(k) - 1)
    bars = ax.bar(k, pmf, color=colors, edgecolor=OFF_WHITE, width=0.72,
                  label=f"Poisson(lambda = {lam:.2f}), fitted model")
    ax.plot(k, pmf_nb, "^--", color=OFF_WHITE, ms=5, lw=1.1, alpha=0.65,
            label=f"NB2 (alpha = {alpha:.3f}), conservative bound")

    ax.set_ylim(0, max(pmf.max(), pmf_nb.max()) * 1.42)
    ax.axvline(lam, color=OFF_WHITE, ls=":", lw=1.3)
    ax.text(lam, ax.get_ylim()[1] * 0.985, f" expected = {lam:.2f}", fontsize=9,
            va="top", ha="left", style="italic")

    for bar, p in zip(bars, pmf):
        ax.text(bar.get_x() + bar.get_width() / 2, p + ax.get_ylim()[1] * 0.018,
                f"{100 * p:.1f}%", ha="center", fontsize=8, color=WOLF_GREY)

    ax.annotate(
        f"ACTUAL: 0 turnovers\nP = {100 * result['p_zero_poisson']:.2f}%  "
        f"(~1 in {result['odds_against_1_in']:.0f})\n"
        f"NB2 bound: {100 * result['p_zero_negative_binomial']:.2f}%",
        xy=(0.34, pmf[0] * 1.05), xytext=(1.15, ax.get_ylim()[1] * 0.44),
        arrowprops={"arrowstyle": "->", "color": ACTION_GREEN, "lw": 1.8,
                    "connectionstyle": "arc3,rad=0.25"},
        fontsize=11, color=OFF_WHITE, fontweight="bold", va="center",
        bbox={"boxstyle": "round", "fc": PANEL, "ec": ACTION_GREEN},
    )

    ax.set_xticks(k)
    ax.set_xlabel("Turnovers across a 3-game playoff run")
    ax.set_ylabel("Probability")
    ax.set_title(
        "Darnold's zero-turnover 2025-26 playoff run vs. his own regular-season rate\n"
        f"exposure = {exposure['dropbacks']:.1f} dropbacks "
        f"({exposure['drives']:.1f} drives), sized from his own per-game average",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    pbp = load_pbp()

    print("Phase 6: Darnold zero-turnover playoff run")
    print("(discrete rate model over bounded trials -- NOT extreme value theory)\n")

    truth = season_total_ground_truth()
    weekly = weekly_table(pbp)
    validate_weekly(weekly, truth)

    recon = dropback_reconciliation(pbp)
    print(f"  dropbacks: {recon['passer_rows_that_are_dropbacks']} passer + "
          f"{recon['scrambles_added']} scrambles = {recon['qb_dropbacks_modelled']} "
          f"({recon['passer_rows']} passer rows incl. {recon['qb_spikes_excluded']} spikes)")

    rates = season_rates(weekly)
    print(f"\nRegular-season turnover rate ({rates['turnovers']} turnovers):")
    print(f"  per dropback: {rates['turnover_rate_per_dropback']:.5f} "
          f"({100 * rates['turnover_rate_per_dropback']:.2f}%, 1 per "
          f"{1 / rates['turnover_rate_per_dropback']:.1f} dropbacks)")
    print(f"  per drive:    {rates['turnover_rate_per_drive']:.5f} "
          f"({100 * rates['turnover_rate_per_drive']:.2f}%, 1 per "
          f"{1 / rates['turnover_rate_per_drive']:.1f} drives)")

    rolling = rolling_rates(weekly)
    trend = trend_test(weekly)
    print("\n'Got safer as the season went on' check:")
    print(f"  first half  {trend['first_half_turnovers']} TO / "
          f"{trend['first_half_dropbacks']} db = "
          f"{100 * trend['first_half_rate_per_dropback']:.2f}%")
    print(f"  second half {trend['second_half_turnovers']} TO / "
          f"{trend['second_half_dropbacks']} db = "
          f"{100 * trend['second_half_rate_per_dropback']:.2f}%")
    print(f"  trend slope p={trend['p_value_trend']:.3f}, "
          f"half-split p={trend['p_value_half_split']:.3f} -> "
          f"{trend['direction']}; narrative supported: {trend['narrative_supported']}")

    disp = dispersion_diagnostics(weekly)
    print(f"\nDispersion: raw VMR={disp['raw_variance_to_mean_ratio']:.3f}, "
          f"exposure-adjusted={disp['exposure_adjusted_dispersion']:.3f} "
          f"(p={disp['p_value_overdispersion']:.3f}), "
          f"NB2 alpha={disp['nb2_alpha']:.4f} (p={disp['p_value_nb2_vs_poisson']:.3f})")
    print(f"  -> using {disp['chosen_distribution'].upper()}")

    exposure = playoff_exposure(rates)
    alpha = disp["nb2_alpha"]
    primary = zero_turnover_probability(
        rates["turnover_rate_per_dropback"], exposure["dropbacks"], alpha
    )
    per_drive = zero_turnover_probability(
        rates["turnover_rate_per_drive"], exposure["drives"], alpha
    )

    actual = playoff_actuals(pbp)
    if actual["turnovers"] != 0:
        raise AssertionError(f"expected 0 playoff turnovers, got {actual['turnovers']}")

    sensitivity = {
        "actual_playoff_dropbacks": zero_turnover_probability(
            rates["turnover_rate_per_dropback"], actual["dropbacks"], alpha
        ),
        "actual_playoff_drives": zero_turnover_probability(
            rates["turnover_rate_per_drive"], actual["drives"], alpha
        ),
        "first_half_rate_regime": zero_turnover_probability(
            trend["first_half_rate_per_dropback"], exposure["dropbacks"], alpha
        ),
        "second_half_rate_regime": zero_turnover_probability(
            trend["second_half_rate_per_dropback"], exposure["dropbacks"], alpha
        ),
        "strict_dropback_turnovers_only": zero_turnover_probability(
            (rates["turnovers"] - 1) / rates["dropbacks"], exposure["dropbacks"], alpha
        ),
    }

    print(f"\nPlayoff exposure (derived): {exposure['dropbacks']:.1f} dropbacks / "
          f"{exposure['drives']:.1f} drives over {PLAYOFF_GAMES} games")
    print(f"  actual: {actual['dropbacks']} dropbacks / {actual['drives']} drives")
    print(f"\nP(zero turnovers) = {primary['p_zero_poisson']:.4f} "
          f"({100 * primary['p_zero_poisson']:.2f}%, ~1 in "
          f"{primary['odds_against_1_in']:.0f}); expected "
          f"{primary['expected_turnovers']:.2f} turnovers, actual 0")
    print(f"  NB2 conservative bound: {100 * primary['p_zero_negative_binomial']:.2f}%")
    print(f"  at his ACTUAL playoff volume ({actual['dropbacks']} db): "
          f"{100 * sensitivity['actual_playoff_dropbacks']['p_zero_poisson']:.2f}%")

    results = {
        "framing": {
            "model_class": "discrete rate model (count of events over bounded trials)",
            "not_extreme_value_theory": (
                "This is explicitly NOT an extreme value theory problem. EVT models "
                "the tail of a CONTINUOUS magnitude distribution (e.g. the largest "
                "single-game passing yardage ever expected). Here we are counting a "
                "small number of discrete events over a bounded number of trials, so "
                "the correct tool is a count distribution (Poisson / negative "
                "binomial) fit to an observed rate. The outcome of interest is the "
                "count k = 0, which has no magnitude tail to model."
            ),
        },
        "ground_truth_check": {
            "season_total_file": truth,
            "weekly_derived": {
                "interceptions": int(weekly["interceptions"].sum()),
                "fumbles_lost": int(weekly["fumbles_lost"].sum()),
                "turnovers": int(weekly["turnovers"].sum()),
                "games": len(weekly),
            },
            "match": True,
        },
        "season_turnovers": rates["turnovers"],
        "dropbacks": rates["dropbacks"],
        "drives": rates["drives"],
        "dropback_reconciliation": recon,
        "turnover_rate_per_dropback": rates["turnover_rate_per_dropback"],
        "turnover_rate_per_drive": rates["turnover_rate_per_drive"],
        "season_rates": rates,
        "weekly": weekly.to_dict(orient="records"),
        "rolling_analysis": {
            "window_games": ROLL_WINDOW,
            "definition": "exposure-weighted: sum(turnovers)/sum(dropbacks) over window",
            **trend,
            "conclusion": (
                "The 'Darnold got safer with the ball as the season went on' "
                "narrative is NOT supported. The point estimate moves the wrong "
                "way (second-half rate is higher), and neither the log-linear "
                "trend test nor the half-split test is significant, so the honest "
                "read is a flat in-season rate with no improvement. The playoff "
                "zero is a discontinuity, not the end of a trend."
            ),
        },
        "distribution": {
            "chosen": disp["chosen_distribution"],
            **disp,
        },
        "playoff_exposure_derived": exposure,
        "playoff_actual": actual,
        "p_zero_turnovers": {
            "primary_per_dropback": primary,
            "per_drive": per_drive,
            "note": (
                "Per-dropback and per-drive give the same lambda by construction: "
                "sizing exposure as 3 x his own per-game average cancels the "
                "denominator, so lambda = 3 x (20/17) = 3.53 either way. The "
                "agreement is arithmetic, not independent confirmation."
            ),
        },
        "sensitivity": sensitivity,
    }

    OUT_DIR.mkdir(exist_ok=True)
    json_path = OUT_DIR / "turnover_rate_model_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {json_path}")

    rolling_path = OUT_DIR / "turnover_rolling_rate.png"
    plot_rolling(rolling, rates, trend, rolling_path)
    print(f"Chart   -> {rolling_path}")

    prob_path = OUT_DIR / "turnover_probability.png"
    plot_probability(primary["expected_turnovers"], alpha, primary, exposure, prob_path)
    print(f"Chart   -> {prob_path}")


if __name__ == "__main__":
    main()
