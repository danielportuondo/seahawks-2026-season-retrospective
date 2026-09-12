"""Phase 5: how statistically extreme was the 2025 "Dark Side" defense?

Builds a team-season defensive dataset for every (season, team) from 2010-2025
(16 seasons x 32 teams = 512 rows) and asks where the 2025 Seahawks defense
sits in that historical distribution. As in Phase 4, the choices that could
quietly wreck the answer are spelled out here rather than left implicit.

Metrics (four, all regular season only -- no playoffs, so the Super Bowl LX run
does not flatter the 2025 row):

1. def_epa_per_play_allowed -- mean EPA the defense allowed on "core" scrimmage
   plays: down 1-4, play_type in {pass, run}, epa not null. GARBAGE TIME IS
   EXCLUDED (win probability between 5% and 95%), matching Phase 4 exactly so
   the two phases' EPA numbers are directly comparable. The tradeoff is worth
   stating: a dominant defense spends more snaps than average in >95% win
   probability, where prevent looks are deliberately soft, so filtering garbage
   time is the CONSERVATIVE choice here -- it removes snaps that would mostly
   have made 2025 look worse, not better. It also protects against the reverse
   artifact (a bad defense's mop-up snaps against a kneeling opponent).

2. sack_rate_created -- defensive sacks / opponent dropbacks (qb_dropback == 1,
   down not null, so 2-point tries and no-plays are out). Sacks are a subset of
   dropbacks, so this is a true rate in [0, 1].

3. takeaways_per_drive -- (interception == 1 or fumble_lost == 1) credited to
   defteam, divided by opponent drives faced. PER DRIVE, not per game, on
   purpose: within a single season, a team whose own offense plays fast hands
   its defense more possessions, so per-game takeaways partly measures the
   OFFENSE. Per-drive isolates the defense. A drive is a distinct fixed_drive
   with at least one scrimmage snap (down not null), so kickoff-only and
   end-of-half artifacts do not inflate the denominator. Turnovers are NOT
   garbage-time filtered (same reasoning as Phase 4: a takeaway is a real event
   regardless of score state). takeaways_per_game is also computed and stored
   for transparency but is not the metric fed into the model.

4. points_allowed_per_game -- from schedules.parquet, mean opponent score
   across that team's regular season games.

RATES, NEVER TOTALS. 2010-2020 were 16-game seasons, 2021-2025 are 17, and
2022 BUF/CIN played 16 after their game was cancelled. Every metric above is a
per-play, per-drive, or per-game average for exactly that reason.

WITHIN-SEASON NORMALIZATION. Each metric is z-scored against its OWN season's
32-team mean and sample SD (ddof=1), and only then pooled across the 16
seasons. Z-scoring against the pooled 2010-2025 distribution instead would
confuse era drift (league scoring, pass rate, and sack rate have all moved over
this window) with genuine team quality, and would systematically favor whichever
era happened to be low-scoring.

SIGN CONVENTION. After z-scoring, the EPA-allowed and points-allowed z-scores
are NEGATED so that for all four metrics "more positive = better defense".
Sack rate and takeaway rate are already higher-is-better and are left alone.
composite_z is the mean of the four signed z-scores.

OUTLIER SCORING.
- Mahalanobis distance (primary) of each team-season's 4-vector of signed
  z-scores from the pooled mean, using the pooled sample covariance. This
  accounts for the fact that the four metrics are correlated -- a defense that
  is good at everything is less surprising than the marginals suggest.
- Isolation Forest (secondary cross-check) on the same 512x4 matrix.

Both are UNSIGNED: a historically awful defense is just as "far from average"
as a historically great one. So the headline ranking is reported two ways --
Mahalanobis rank restricted to good-direction team-seasons (composite_z > 0),
and a plain composite_z rank -- rather than raw distance alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from chart_style import (
    ACTION_GREEN,
    ALERT_RED,
    OFF_WHITE,
    WOLF_GREY,
    apply_scoreboard_style,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"

SEASONS = list(range(2010, 2026))
FOCUS_TEAM = "SEA"
FOCUS_SEASONS = (2024, 2025)
RANDOM_STATE = 42

PBP_COLS = [
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "down",
    "play_type",
    "epa",
    "wp",
    "sack",
    "qb_dropback",
    "interception",
    "fumble_lost",
    "fixed_drive",
]

METRICS = [
    "def_epa_per_play_allowed",
    "sack_rate_created",
    "takeaways_per_drive",
    "points_allowed_per_game",
]
# Metrics where a LOW raw value is a GOOD defense -> z-score gets negated.
LOWER_IS_BETTER = {"def_epa_per_play_allowed", "points_allowed_per_game"}

# schedules.parquet keeps era-contemporary codes (STL/SD/OAK) while pbp.parquet
# has already been standardized to current franchise codes. Left unmapped, the
# two sources silently fail to join for 30 team-seasons.
RELOCATION_MAP = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

METRIC_LABELS = {
    "def_epa_per_play_allowed": "EPA/play allowed\n(inverted)",
    "sack_rate_created": "Sack rate\ncreated",
    "takeaways_per_drive": "Takeaways\nper drive",
    "points_allowed_per_game": "Points allowed/game\n(inverted)",
}


def z_col(metric: str) -> str:
    return f"z_{metric}"


def load_pbp(seasons: list[int] = SEASONS) -> pd.DataFrame:
    frames = []
    for season in seasons:
        df = pd.read_parquet(RAW_DIR / "pbp" / f"{season}.parquet", columns=PBP_COLS)
        frames.append(df[df["season_type"] == "REG"])
    return pd.concat(frames, ignore_index=True)


def epa_allowed(pbp: pd.DataFrame) -> pd.DataFrame:
    core = pbp[
        pbp["down"].notna()
        & pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
        & pbp["wp"].between(0.05, 0.95)
        & pbp["defteam"].notna()
    ]
    out = (
        core.groupby(["season", "defteam"])
        .agg(def_epa_per_play_allowed=("epa", "mean"), core_plays_defended=("epa", "size"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    return out


def sack_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    dropbacks = pbp[(pbp["qb_dropback"] == 1) & pbp["down"].notna() & pbp["defteam"].notna()]
    out = (
        dropbacks.assign(is_sack=(dropbacks["sack"] == 1).astype(int))
        .groupby(["season", "defteam"])
        .agg(opp_dropbacks=("is_sack", "size"), sacks_created=("is_sack", "sum"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    out["sack_rate_created"] = out["sacks_created"] / out["opp_dropbacks"]
    return out


def takeaway_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    defended = pbp[pbp["defteam"].notna()]

    # A "drive faced" must contain a real scrimmage snap, else kickoff-only and
    # end-of-half fragments pad the denominator.
    drives = (
        defended[defended["down"].notna()]
        .groupby(["season", "defteam"])
        .apply(lambda g: g.set_index(["week", "fixed_drive"]).index.nunique(), include_groups=False)
        .rename("opp_drives")
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    is_takeaway = (defended["interception"] == 1) | (defended["fumble_lost"] == 1)
    takeaways = (
        defended[is_takeaway]
        .groupby(["season", "defteam"])
        .size()
        .rename("takeaways")
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    out = drives.merge(takeaways, on=["season", "team"], how="left")
    out["takeaways"] = out["takeaways"].fillna(0)
    out["takeaways_per_drive"] = out["takeaways"] / out["opp_drives"]
    return out


def points_allowed(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules[(schedules["game_type"] == "REG") & schedules["season"].isin(SEASONS)]
    home = reg[["season", "home_team", "away_score"]].rename(
        columns={"home_team": "team", "away_score": "points_allowed"}
    )
    away = reg[["season", "away_team", "home_score"]].rename(
        columns={"away_team": "team", "home_score": "points_allowed"}
    )
    stacked = pd.concat([home, away], ignore_index=True)
    stacked["team"] = stacked["team"].replace(RELOCATION_MAP)
    out = (
        stacked.groupby(["season", "team"])
        .agg(points_allowed_total=("points_allowed", "sum"), games=("points_allowed", "size"))
        .reset_index()
    )
    out["points_allowed_per_game"] = out["points_allowed_total"] / out["games"]
    return out


def build_team_season_defense(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    df = (
        epa_allowed(pbp)
        .merge(sack_rate(pbp), on=["season", "team"], how="outer")
        .merge(takeaway_rate(pbp), on=["season", "team"], how="outer")
        .merge(points_allowed(schedules), on=["season", "team"], how="outer")
    )
    df["takeaways_per_game"] = df["takeaways"] / df["games"]
    return df.sort_values(["season", "team"]).reset_index(drop=True)


def add_within_season_z(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in METRICS:
        grouped = out.groupby("season")[metric]
        z = (out[metric] - grouped.transform("mean")) / grouped.transform("std")
        if metric in LOWER_IS_BETTER:
            z = -z
        out[z_col(metric)] = z
    out["composite_z"] = out[[z_col(m) for m in METRICS]].mean(axis=1)
    return out


def mahalanobis(z_matrix: np.ndarray) -> np.ndarray:
    centered = z_matrix - z_matrix.mean(axis=0)
    cov_inv = np.linalg.inv(np.cov(z_matrix, rowvar=False, ddof=1))
    return np.sqrt(np.einsum("ij,jk,ik->i", centered, cov_inv, centered))


def isolation_forest_scores(z_matrix: np.ndarray) -> np.ndarray:
    """Return anomaly scores where HIGHER = more anomalous (sklearn's are inverted)."""
    model = IsolationForest(n_estimators=500, random_state=RANDOM_STATE)
    model.fit(z_matrix)
    return -model.score_samples(z_matrix)


def add_outlier_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z_matrix = out[[z_col(m) for m in METRICS]].to_numpy()
    out["mahalanobis"] = mahalanobis(z_matrix)
    out["iso_forest_anomaly"] = isolation_forest_scores(z_matrix)

    n = len(out)
    out["mahalanobis_rank"] = out["mahalanobis"].rank(ascending=False).astype(int)
    out["mahalanobis_pctile"] = 100 * out["mahalanobis"].rank(pct=True)
    out["iso_forest_pctile"] = 100 * out["iso_forest_anomaly"].rank(pct=True)
    out["composite_z_rank"] = out["composite_z"].rank(ascending=False).astype(int)
    out["composite_z_pctile"] = 100 * out["composite_z"].rank(pct=True)

    # Restrict to good-direction outliers: a 2-win defense is also "far from
    # average", and lumping it in would make the headline claim meaningless.
    good = out["composite_z"] > 0
    out["good_direction"] = good
    out["mahalanobis_rank_good"] = np.nan
    out.loc[good, "mahalanobis_rank_good"] = out.loc[good, "mahalanobis"].rank(ascending=False)
    out.attrs["n"] = n
    out.attrs["n_good"] = int(good.sum())
    return out


def row_for(df: pd.DataFrame, season: int, team: str = FOCUS_TEAM) -> pd.Series:
    match = df[(df["season"] == season) & (df["team"] == team)]
    if len(match) != 1:
        raise ValueError(f"expected exactly 1 row for {team} {season}, got {len(match)}")
    return match.iloc[0]


def radar_chart(df: pd.DataFrame, path: Path) -> None:
    apply_scoreboard_style()
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7), subplot_kw={"projection": "polar"})
    for season, color in zip(FOCUS_SEASONS, [WOLF_GREY, ACTION_GREEN]):
        row = row_for(df, season)
        values = [row[z_col(m)] for m in METRICS]
        values += values[:1]
        ax.plot(closed, values, color=color, linewidth=2, label=f"SEA {season}")
        ax.fill(closed, values, color=color, alpha=0.18)

    ax.plot(closed, [0] * len(closed), color=WOLF_GREY, linewidth=1, linestyle="--")
    ax.set_xticks(angles)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9)
    ax.set_ylim(-2.5, 3.5)
    ax.set_yticks([-2, -1, 0, 1, 2, 3])
    ax.set_yticklabels(["-2", "-1", "0 (avg)", "+1", "+2", "+3"], fontsize=8)
    ax.set_rlabel_position(45)
    ax.set_title(
        "Seahawks defense, within-season z-scores\n(outward = better defense; dashed ring = league average)",
        fontsize=12,
        pad=24,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def historical_rank_chart(df: pd.DataFrame, path: Path) -> None:
    apply_scoreboard_style()
    sea24, sea25 = (row_for(df, s) for s in FOCUS_SEASONS)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(df["mahalanobis"], bins=40, color=WOLF_GREY, edgecolor=OFF_WHITE)
    axes[0].set_title(f"Mahalanobis distance from average\n(all {len(df)} team-seasons, 2010-2025)")
    axes[0].set_xlabel("Mahalanobis distance (unsigned)")
    axes[0].set_ylabel("Team-seasons")

    axes[1].hist(df["composite_z"], bins=40, color=WOLF_GREY, edgecolor=OFF_WHITE)
    axes[1].set_title("Composite defensive z-score\n(mean of 4 signed z-scores)")
    axes[1].set_xlabel("Composite z (higher = better defense)")
    axes[1].set_ylabel("Team-seasons")

    for ax, key in zip(axes[:2], ["mahalanobis", "composite_z"]):
        for row, color, season in ((sea24, WOLF_GREY, 2024), (sea25, ACTION_GREEN, 2025)):
            ax.axvline(row[key], color=color, linewidth=2)
            ax.annotate(
                f"SEA {season}\n{row[key]:.2f}",
                xy=(row[key], ax.get_ylim()[1] * 0.88),
                xytext=(6, 0),
                textcoords="offset points",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

    good = df["good_direction"]
    axes[2].scatter(
        df.loc[~good, "composite_z"], df.loc[~good, "mahalanobis"], s=12, color=ALERT_RED, alpha=0.35, label="Bad-direction outliers"
    )
    axes[2].scatter(
        df.loc[good, "composite_z"], df.loc[good, "mahalanobis"], s=12, color=WOLF_GREY, alpha=0.45, label="Good-direction"
    )
    for row, color, season in ((sea24, WOLF_GREY, 2024), (sea25, ACTION_GREEN, 2025)):
        axes[2].scatter(row["composite_z"], row["mahalanobis"], s=140, color=color, edgecolor=OFF_WHITE, zorder=5)
        axes[2].annotate(
            f"SEA {season}",
            xy=(row["composite_z"], row["mahalanobis"]),
            xytext=(8, -4),
            textcoords="offset points",
            color=color,
            fontsize=10,
            fontweight="bold",
        )
    axes[2].axvline(0, color=OFF_WHITE, linewidth=0.8, linestyle="--")
    axes[2].set_title("Distance is unsigned:\ngood and bad defenses both sit far out")
    axes[2].set_xlabel("Composite z (higher = better defense)")
    axes[2].set_ylabel("Mahalanobis distance")
    axes[2].legend(fontsize=8, loc="lower left")

    fig.suptitle("How extreme was the 2025 Seahawks defense? (2010-2025 team-seasons)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(df: pd.DataFrame) -> dict:
    def season_block(season: int) -> dict:
        row = row_for(df, season)
        return {
            "raw": {m: round(float(row[m]), 4) for m in METRICS},
            "z_scores": {m: round(float(row[z_col(m)]), 3) for m in METRICS},
            "composite_z": round(float(row["composite_z"]), 3),
            "composite_z_rank_of_n": int(row["composite_z_rank"]),
            "composite_z_pctile": round(float(row["composite_z_pctile"]), 2),
            "mahalanobis": round(float(row["mahalanobis"]), 3),
            "mahalanobis_pctile": round(float(row["mahalanobis_pctile"]), 2),
            "mahalanobis_rank_of_n": int(row["mahalanobis_rank"]),
            "mahalanobis_rank_among_good_direction": (
                int(row["mahalanobis_rank_good"]) if row["good_direction"] else None
            ),
            "good_direction": bool(row["good_direction"]),
            "iso_forest_anomaly_score": round(float(row["iso_forest_anomaly"]), 4),
            "iso_forest_pctile": round(float(row["iso_forest_pctile"]), 2),
            "takeaways_per_game": round(float(row["takeaways_per_game"]), 3),
            "games": int(row["games"]),
        }

    # Does the within-season normalization drive the answer? Compare each
    # headline metric's normalized rank to its era-naive raw rank. If the two
    # agree, era drift is not what is holding SEA 2025 out of the all-time top
    # 10 -- the defense genuinely was not that extreme.
    sea25 = row_for(df, 2025)
    robustness = {"era_naive_vs_within_season_rank": {}}
    for metric in ("def_epa_per_play_allowed", "points_allowed_per_game"):
        robustness["era_naive_vs_within_season_rank"][metric] = {
            "raw_rank_of_n": int(df[metric].rank(ascending=True)[sea25.name]),
            "within_season_z_rank_of_n": int(df[z_col(metric)].rank(ascending=False)[sea25.name]),
        }
    z_cols = [z_col(m) for m in METRICS]
    for label, cols in (
        ("all_4_metrics", z_cols),
        ("drop_sack_rate", [c for c in z_cols if "sack" not in c]),
        ("epa_and_points_only", [z_col("def_epa_per_play_allowed"), z_col("points_allowed_per_game")]),
    ):
        composite = df[cols].mean(axis=1)
        robustness.setdefault("composite_rank_by_metric_subset", {})[label] = {
            "composite_z": round(float(composite[sea25.name]), 3),
            "rank_of_n": int(composite.rank(ascending=False)[sea25.name]),
        }
    robustness["within_season_sd_def_epa"] = (
        df.groupby("season")["def_epa_per_play_allowed"].std().round(4).to_dict()
    )

    top = df.nlargest(10, "composite_z")[["season", "team", "composite_z", "mahalanobis"]]
    top_maha_good = (
        df[df["good_direction"]].nlargest(10, "mahalanobis")[["season", "team", "mahalanobis", "composite_z"]]
    )

    return {
        "methodology": {
            "seasons": [SEASONS[0], SEASONS[-1]],
            "n_team_seasons": int(df.attrs["n"]),
            "n_good_direction": int(df.attrs["n_good"]),
            "season_type": "REG only (regular season; playoffs excluded)",
            "garbage_time_filter": "EPA/play only, 0.05 <= wp <= 0.95 (matches Phase 4)",
            "takeaway_rate_denominator": "opponent drives with >=1 scrimmage snap",
            "sack_rate_denominator": "opponent dropbacks (qb_dropback == 1, down not null)",
            "normalization": "z-score within season across 32 teams (ddof=1), then pooled",
            "sign_convention": "EPA allowed and points allowed negated; higher z = better defense",
            "composite_z": "unweighted mean of the four signed z-scores",
        },
        "SEA_2024": season_block(2024),
        "SEA_2025": season_block(2025),
        "robustness": robustness,
        "top10_by_composite_z": top.round(3).to_dict(orient="records"),
        "top10_by_mahalanobis_good_direction": top_maha_good.round(3).to_dict(orient="records"),
        "metric_correlations_pooled": (
            df[[z_col(m) for m in METRICS]].corr().round(3).rename(columns=lambda c: c.replace("z_", "")).to_dict()
        ),
    }


def main() -> None:
    pbp = load_pbp()
    schedules = pd.read_parquet(RAW_DIR / "schedules.parquet")

    df = build_team_season_defense(pbp, schedules)
    assert df[METRICS].notna().all().all(), "missing metric values in team-season table"
    assert df.groupby("season").size().eq(32).all(), "expected 32 teams in every season"

    df = add_within_season_z(df)
    df = add_outlier_scores(df)

    # Sign sanity check: the best defenses by composite z must be the ones
    # allowing the FEWEST points, not the most. If this trips, a sign flipped.
    best = df.nlargest(32, "composite_z")["points_allowed_per_game"].mean()
    worst = df.nsmallest(32, "composite_z")["points_allowed_per_game"].mean()
    assert best < worst, f"sign convention broken: top composite_z allows {best:.1f} ppg vs {worst:.1f}"

    OUT_DIR.mkdir(exist_ok=True)
    results = summarize(df)
    with open(OUT_DIR / "defense_anomaly_results.json", "w") as f:
        json.dump(results, f, indent=2)

    radar_chart(df, OUT_DIR / "defense_anomaly_radar.png")
    historical_rank_chart(df, OUT_DIR / "defense_anomaly_historical_rank.png")

    n = df.attrs["n"]
    print(f"team-season defense table: {df.shape} ({n} team-seasons, {df.attrs['n_good']} good-direction)")
    print(f"sign check OK: best-32 composite_z allow {best:.1f} ppg, worst-32 allow {worst:.1f} ppg\n")

    for season in FOCUS_SEASONS:
        row = row_for(df, season)
        print(f"SEA {season}")
        for m in METRICS:
            flip = " (inverted)" if m in LOWER_IS_BETTER else ""
            print(f"  {m:<26} raw={row[m]:>8.4f}   z={row[z_col(m)]:+.2f}{flip}")
        print(f"  {'composite_z':<26}              z={row['composite_z']:+.2f}   "
              f"rank {int(row['composite_z_rank'])}/{n} (p{row['composite_z_pctile']:.1f})")
        print(f"  mahalanobis={row['mahalanobis']:.2f}  rank {int(row['mahalanobis_rank'])}/{n} "
              f"(p{row['mahalanobis_pctile']:.1f})  good-direction rank="
              f"{int(row['mahalanobis_rank_good']) if row['good_direction'] else 'n/a (bad direction)'}")
        print(f"  iso_forest_anomaly={row['iso_forest_anomaly']:.4f} (p{row['iso_forest_pctile']:.1f})\n")

    print("Top 10 team-season defenses by composite z, 2010-2025:")
    print(df.nlargest(10, "composite_z")[["season", "team", "composite_z", "mahalanobis"]].to_string(index=False))

    print(f"\nWrote {OUT_DIR / 'defense_anomaly_results.json'}")
    print(f"Wrote {OUT_DIR / 'defense_anomaly_radar.png'}")
    print(f"Wrote {OUT_DIR / 'defense_anomaly_historical_rank.png'}")


if __name__ == "__main__":
    main()
