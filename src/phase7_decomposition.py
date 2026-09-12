"""Phase 7: quantifying the jump -- regression decomposition of SEA's 2024->2025 shift.

Depends on Phase 4's engineered feature file (data/processed/team_week_features.csv).
Goal: fit a cross-sectional model of point differential from the same rate stats
Phase 4 built (EPA/play, turnover margin, red-zone TD%, pressure rate), then use its
coefficients to decompose SEA's actual 2024->2025 change into per-feature
contributions -- how much of the jump does each stat "explain"?

MODEL CHOICE: linear regression, not XGBoost+SHAP. HANDOFF.md offers either;
this project doesn't have xgboost/shap installed, and more importantly,
gradient boosting is the wrong tool at this sample size: the regression sample
is 64 team-seasons (32 teams x 2 seasons), and both EPA/play metrics are near-mechanical inputs to point
differential (this is well established in public NFL analytics -- EPA/play
differential alone typically explains >90% of point differential). A tree
ensemble on 64 rows would overfit and its SHAP attributions would be noise,
while a linear model's coefficients ARE the decomposition (contribution_i =
coef_i * change_in_feature_i), which is exactly what "regression decomposition"
means -- no post-hoc explainability layer needed.

TARGET: point differential per game, not wins. Phase 2 already showed SEA
over-performed its Pythagorean win expectation both years (2024: 8.7 expected
wins vs 10 actual; 2025: 13.0 vs 14) -- i.e. wins already contain a
tiebreaker/luck component point differential does not explain by construction.
Decomposing THAT would conflate "what changed about how SEA played" with "how
close games bounced." A secondary wins~features regression is still fit and
reported, as a rougher cross-check, but the headline decomposition targets point
differential.

FEATURE AGGREGATION (team-week -> team-season): EPA/play and pressure-rate
percentages are averaged across weeks with EQUAL WEIGHT PER GAME, not
play-weighted -- Phase 4's file doesn't carry play counts for the EPA columns, so
exact play-weighting would require re-reading raw pbp, which defeats the point of
depending on Phase 4's file. This is a acknowledged simplification (each of a
team's 17 games counts equally regardless of play volume), documented rather than
hidden. Red-zone TD% and pressure rates ARE recomputed from their underlying
weekly counts (red_zone_trips/tds, dropbacks/pressured_dropbacks) rather than
averaged as weekly percentages, since averaging percentages across weeks of
uneven sample size (e.g. a 1-trip week at 100% vs a 5-trip week at 40%) would
overweight small samples. Turnover margin is a sum across weeks (equivalent to
computing it from summed forced/committed counts).

STANDARDIZATION: all 6 predictors are z-scored across the pooled 64 team-season
sample before fitting, so coefficients are directly comparable in magnitude and
the decomposition units are "standard deviations of feature x sample coefficient."

INFERENCE: OLS coefficients, standard errors, and t/p-values are computed by hand
(no statsmodels installed) via the standard (X'X)^-1 sigma^2 formula -- this is
exact classical OLS inference, not an approximation, and needs only numpy/scipy
(both already installed).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from chart_style import (
    ACTION_GREEN,
    ALERT_RED,
    OFF_WHITE,
    WOLF_GREY,
    apply_scoreboard_style,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

FOCUS_TEAM = "SEA"
FOCUS_SEASONS = (2024, 2025)

FEATURES = [
    "off_epa_per_play",
    "def_epa_per_play_allowed",
    "turnover_margin_per_game",
    "red_zone_td_pct",
    "pressure_rate_allowed",
    "pressure_rate_created",
]
FEATURE_LABELS = {
    "off_epa_per_play": "Offensive EPA/play",
    "def_epa_per_play_allowed": "Defensive EPA/play allowed",
    "turnover_margin_per_game": "Turnover margin/game",
    "red_zone_td_pct": "Red zone TD%",
    "pressure_rate_allowed": "Pressure rate allowed",
    "pressure_rate_created": "Pressure rate created",
}
GROUND_TRUTH_WINS = {2024: 10, 2025: 14}


def load_team_season_features() -> pd.DataFrame:
    """Team-week -> team-season aggregation described above.

    Reused as-is by the dashboard's "Engineered features" table so that table
    and this phase's per-feature decomposition always report the same numbers
    for red_zone_td_pct / pressure_rate_allowed / pressure_rate_created.
    """
    weekly = pd.read_csv(PROCESSED_DIR / "team_week_features.csv")

    agg = weekly.groupby(["season", "team"]).agg(
        off_epa_per_play=("off_epa_per_play", "mean"),
        def_epa_per_play_allowed=("def_epa_per_play_allowed", "mean"),
        turnover_margin=("turnover_margin", "sum"),
        red_zone_trips=("red_zone_trips", "sum"),
        red_zone_tds=("red_zone_tds", "sum"),
        dropbacks=("dropbacks", "sum"),
        pressured_dropbacks=("pressured_dropbacks", "sum"),
        opp_dropbacks=("opp_dropbacks", "sum"),
        pressures_created=("pressures_created", "sum"),
        games=("week", "nunique"),
    ).reset_index()

    agg["turnover_margin_per_game"] = agg["turnover_margin"] / agg["games"]
    agg["red_zone_td_pct"] = 100 * agg["red_zone_tds"] / agg["red_zone_trips"]
    agg["pressure_rate_allowed"] = 100 * agg["pressured_dropbacks"] / agg["dropbacks"]
    agg["pressure_rate_created"] = 100 * agg["pressures_created"] / agg["opp_dropbacks"]
    return agg


def load_team_season_scoring() -> pd.DataFrame:
    sched = pd.read_parquet(RAW_DIR / "schedules.parquet")
    reg = sched[(sched["season"].isin(FOCUS_SEASONS)) & (sched["game_type"] == "REG")]

    def half(df: pd.DataFrame, team_col: str, own_col: str, opp_col: str) -> pd.DataFrame:
        out = df[["season", team_col, own_col, opp_col]].rename(
            columns={team_col: "team", own_col: "points_for", opp_col: "points_against"}
        )
        out["win"] = np.select(
            [out["points_for"] > out["points_against"], out["points_for"] == out["points_against"]],
            [1.0, 0.5],
            default=0.0,
        )
        return out

    stacked = pd.concat(
        [
            half(reg, "home_team", "home_score", "away_score"),
            half(reg, "away_team", "away_score", "home_score"),
        ],
        ignore_index=True,
    )
    out = (
        stacked.groupby(["season", "team"])
        .agg(points_for=("points_for", "sum"), points_against=("points_against", "sum"),
             wins=("win", "sum"), games=("win", "size"))
        .reset_index()
    )
    out["point_diff_per_game"] = (out["points_for"] - out["points_against"]) / out["games"]
    return out


def fit_ols(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Classical OLS by hand: coefficients, SEs, t-stats, p-values, R^2."""
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])  # design matrix with intercept
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ y
    y_hat = Xd @ beta
    resid = y - y_hat
    dof = n - (k + 1)
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stats = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=dof)

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / dof

    names = ["intercept"] + feature_names
    return {
        "n": n,
        "dof_resid": int(dof),
        "r2": float(r2),
        "r2_adj": float(r2_adj),
        "coefficients": {name: float(b) for name, b in zip(names, beta)},
        "std_errors": {name: float(s) for name, s in zip(names, se)},
        "t_stats": {name: float(t) for name, t in zip(names, t_stats)},
        "p_values": {name: float(p) for name, p in zip(names, p_values)},
        "residual_std": float(np.sqrt(sigma2)),
    }


def row_for(df: pd.DataFrame, season: int, team: str = FOCUS_TEAM) -> pd.Series:
    match = df[(df["season"] == season) & (df["team"] == team)]
    if len(match) != 1:
        raise ValueError(f"expected exactly 1 row for {team} {season}, got {len(match)}")
    return match.iloc[0]


def decompose_sea_change(df: pd.DataFrame, ols: dict, z_cols: list[str]) -> dict:
    sea24, sea25 = row_for(df, 2024), row_for(df, 2025)
    coefs = ols["coefficients"]

    contributions = {}
    for raw, z in zip(FEATURES, z_cols):
        delta_z = float(sea25[z] - sea24[z])
        contributions[raw] = {
            "raw_2024": round(float(sea24[raw]), 4),
            "raw_2025": round(float(sea25[raw]), 4),
            "z_2024": round(float(sea24[z]), 3),
            "z_2025": round(float(sea25[z]), 3),
            "delta_z": round(delta_z, 3),
            "coefficient": round(coefs[raw], 4),
            "contribution_to_point_diff_change": round(coefs[raw] * delta_z, 4),
        }

    predicted_change = sum(c["contribution_to_point_diff_change"] for c in contributions.values())
    actual_change = float(sea25["point_diff_per_game"] - sea24["point_diff_per_game"])
    residual = actual_change - predicted_change

    return {
        "sea_point_diff_per_game_2024": round(float(sea24["point_diff_per_game"]), 3),
        "sea_point_diff_per_game_2025": round(float(sea25["point_diff_per_game"]), 3),
        "actual_change": round(actual_change, 3),
        "predicted_change_from_features": round(predicted_change, 3),
        "residual_unexplained": round(residual, 3),
        "residual_pct_of_actual_change": round(100 * residual / actual_change, 1) if actual_change else None,
        "per_feature": contributions,
    }


def feature_importance_chart(ols: dict, path: Path) -> None:
    apply_scoreboard_style()
    names = [f for f in FEATURES]
    coefs = [ols["coefficients"][f] for f in names]
    pvals = [ols["p_values"][f] for f in names]
    order = np.argsort(np.abs(coefs))
    names = [names[i] for i in order]
    coefs = [coefs[i] for i in order]
    pvals = [pvals[i] for i in order]
    labels = [FEATURE_LABELS[n] for n in names]
    colors = [ACTION_GREEN if c > 0 else ALERT_RED for c in coefs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, coefs, color=colors, edgecolor=OFF_WHITE)
    for bar, p in zip(bars, pvals):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        x = bar.get_width()
        ax.text(x + (0.02 if x >= 0 else -0.02) * max(abs(min(coefs)), abs(max(coefs))),
                bar.get_y() + bar.get_height() / 2, sig,
                va="center", ha="left" if x >= 0 else "right", fontsize=9)

    ax.axvline(0, color=OFF_WHITE, lw=0.8)
    ax.set_xlabel("Standardized OLS coefficient (points/game per 1 SD of feature)")
    ax.set_title(f"What predicts point differential across the NFL?\n"
                 f"({ols['n']} team-seasons, 2024-2025; R²={ols['r2']:.3f})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def waterfall_chart(decomp: dict, path: Path) -> None:
    apply_scoreboard_style()
    order = sorted(decomp["per_feature"].items(), key=lambda kv: -abs(kv[1]["contribution_to_point_diff_change"]))
    labels = ["SEA 2024\npoint diff/gm"] + [FEATURE_LABELS[k] for k, _ in order] + ["Residual\n(unexplained)", "SEA 2025\npoint diff/gm"]
    start = decomp["sea_point_diff_per_game_2024"]
    steps = [v["contribution_to_point_diff_change"] for _, v in order] + [decomp["residual_unexplained"]]

    cum = [start]
    for s in steps:
        cum.append(cum[-1] + s)
    end = decomp["sea_point_diff_per_game_2025"]

    fig, ax = plt.subplots(figsize=(11, 6))

    # First bar: base value 2024
    ax.bar(0, start, color=WOLF_GREY)
    ax.text(0, start + 0.15, f"{start:.2f}", ha="center", fontsize=9)

    for i, s in enumerate(steps, start=1):
        bottom = cum[i - 1]
        color = ACTION_GREEN if s >= 0 else ALERT_RED
        ax.bar(i, s, bottom=bottom, color=color)
        y = bottom + s + (0.15 if s >= 0 else -0.25)
        ax.text(i, y, f"{s:+.2f}", ha="center", fontsize=9)

    # Last bar: actual 2025
    ax.bar(len(labels) - 1, end, color=ACTION_GREEN)
    ax.text(len(labels) - 1, end + 0.15, f"{end:.2f}", ha="center", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Point differential per game")
    ax.axhline(0, color=OFF_WHITE, lw=0.6)
    ax.set_title("Decomposing SEA's 2024→2025 point-differential jump\n"
                 "(contribution = standardized OLS coefficient × change in that feature's z-score)")
    ax.grid(axis="y", alpha=0.25)

    # Extra headroom so value labels on the tallest bars never collide with
    # the title -- the running cumulative total (not just the step sizes) can
    # exceed every individual bar height, so pad relative to max(cum).
    ymin, ymax = min(0, *cum), max(cum)
    pad = 0.18 * (ymax - ymin)
    ax.set_ylim(ymin - pad * 0.3, ymax + pad)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    features = load_team_season_features()
    scoring = load_team_season_scoring()
    df = features.merge(scoring, on=["season", "team"], how="inner")
    assert len(df) == 64, f"expected 64 team-seasons (32 teams x 2 seasons), got {len(df)}"

    for season in FOCUS_SEASONS:
        wins = round(float(row_for(df, season)["wins"]))
        assert wins == GROUND_TRUTH_WINS[season], (
            f"SEA {season} wins={wins}, expected {GROUND_TRUTH_WINS[season]}"
        )

    z_cols = []
    for f in FEATURES:
        z_col = f"z_{f}"
        df[z_col] = (df[f] - df[f].mean()) / df[f].std(ddof=1)
        z_cols.append(z_col)

    X = df[z_cols].to_numpy()
    y_point_diff = df["point_diff_per_game"].to_numpy()
    ols_point_diff = fit_ols(X, y_point_diff, FEATURES)

    y_wins = df["wins"].to_numpy()
    ols_wins = fit_ols(X, y_wins, FEATURES)

    # sklearn cross-check: coefficients should match the hand-rolled OLS (up to
    # the intercept/scaling convention) as a sanity check on the by-hand linear
    # algebra above.
    sk = LinearRegression().fit(X, y_point_diff)
    max_coef_diff = float(np.max(np.abs(sk.coef_ - np.array([ols_point_diff["coefficients"][f] for f in FEATURES]))))
    assert max_coef_diff < 1e-8, f"hand-rolled OLS disagrees with sklearn: max diff {max_coef_diff}"

    decomp = decompose_sea_change(df, ols_point_diff, z_cols)

    OUT_DIR.mkdir(exist_ok=True)
    results = {
        "methodology": {
            "n_team_seasons": len(df),
            "seasons": list(FOCUS_SEASONS),
            "model": "OLS, standardized (z-scored) features, fit across all 32 teams x 2 seasons",
            "primary_target": "point_diff_per_game (points_for - points_against, per game)",
            "secondary_target": "wins (regular-season wins, ties=0.5), reported as a rougher cross-check",
            "why_not_wins_primary": (
                "Phase 2 showed SEA overachieved its Pythagorean win expectation both years, "
                "so wins already contain a tiebreaker/luck component that point differential "
                "does not capture by construction."
            ),
            "why_not_gradient_boosting": (
                "At n=64, a gradient-boosted model would overfit and its SHAP attributions "
                "would be noisier than linear coefficients, which already ARE the decomposition "
                "-- no separate explainability layer is needed on top of them."
            ),
            "sklearn_cross_check_max_coef_diff": max_coef_diff,
        },
        "ols_point_diff_per_game": ols_point_diff,
        "ols_wins_secondary": ols_wins,
        "sea_decomposition": decomp,
    }
    with open(OUT_DIR / "decomposition_results.json", "w") as f:
        json.dump(results, f, indent=2)

    feature_importance_chart(ols_point_diff, OUT_DIR / "decomposition_feature_importance.png")
    waterfall_chart(decomp, OUT_DIR / "decomposition_waterfall.png")

    print(f"Team-season table: {df.shape} ({len(df)} team-seasons)")
    print(f"\nOLS point_diff_per_game ~ features: R²={ols_point_diff['r2']:.3f} "
          f"(adj {ols_point_diff['r2_adj']:.3f}), n={ols_point_diff['n']}")
    for f in FEATURES:
        c = ols_point_diff["coefficients"][f]
        p = ols_point_diff["p_values"][f]
        print(f"  {FEATURE_LABELS[f]:<28} coef={c:+.3f}  p={p:.4f}")

    print(f"\nOLS wins ~ features (secondary): R²={ols_wins['r2']:.3f} (adj {ols_wins['r2_adj']:.3f})")

    print(f"\nSEA point diff/game: {decomp['sea_point_diff_per_game_2024']:.2f} (2024) "
          f"-> {decomp['sea_point_diff_per_game_2025']:.2f} (2025), "
          f"actual change {decomp['actual_change']:+.2f}")
    print("Decomposition:")
    for name, c in sorted(decomp["per_feature"].items(),
                           key=lambda kv: -abs(kv[1]["contribution_to_point_diff_change"])):
        print(f"  {FEATURE_LABELS[name]:<28} {c['contribution_to_point_diff_change']:+.3f}")
    print(f"  {'Residual (unexplained)':<28} {decomp['residual_unexplained']:+.3f} "
          f"({decomp['residual_pct_of_actual_change']}% of actual change)")

    print(f"\nWrote {OUT_DIR / 'decomposition_results.json'}")
    print(f"Wrote {OUT_DIR / 'decomposition_feature_importance.png'}")
    print(f"Wrote {OUT_DIR / 'decomposition_waterfall.png'}")


if __name__ == "__main__":
    main()
