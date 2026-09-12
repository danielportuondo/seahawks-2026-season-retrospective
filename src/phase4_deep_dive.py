"""Phase 4: statistical deep dive, 2024 vs 2025 regular seasons.

Builds team-week engineered features from play-by-play. This is the part of
the project most likely to go subtly wrong, so the filtering choices are
spelled out here rather than left implicit:

- EPA/play (offense and defense) is computed on "core" scrimmage plays only
  (down 1-4, play_type in {pass, run} — excludes kickoffs/PATs/2-pt tries/
  no-plays) AND excludes garbage time, defined the standard way used by
  public EPA tools (e.g. rbsdm.com): plays where win probability is between
  5% and 95%. Blowout 4th-quarter snaps otherwise swing a team's EPA/play
  without reflecting how the team actually played in competitive situations.
- Turnover margin, red zone TD%, and pressure rate are NOT garbage-time
  filtered — a turnover or a red zone trip is a real event regardless of
  score state, and excluding it would just throw away sample for no
  principled reason.
- Red zone TD% is computed at the drive level (a drive that ever reached
  inside the 20 counts as a red zone trip; it "hits" if the drive's fixed
  result is a touchdown), not at the play level.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

SEASONS = [2024, 2025]


def load_pbp() -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(RAW_DIR / "pbp" / f"{s}.parquet") for s in SEASONS],
        ignore_index=True,
    )


def epa_features(pbp: pd.DataFrame) -> pd.DataFrame:
    core = pbp[
        (pbp["season_type"] == "REG")
        & pbp["down"].notna()
        & pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
        & pbp["wp"].between(0.05, 0.95)
    ]

    off = core.groupby(["season", "week", "posteam"])["epa"].mean().rename("off_epa_per_play")
    dfn = core.groupby(["season", "week", "defteam"])["epa"].mean().rename("def_epa_per_play_allowed")

    off.index.set_names(["season", "week", "team"], inplace=True)
    dfn.index.set_names(["season", "week", "team"], inplace=True)
    return pd.concat([off, dfn], axis=1).reset_index()


def turnover_features(pbp: pd.DataFrame) -> pd.DataFrame:
    reg = pbp[pbp["season_type"] == "REG"]
    is_turnover = (reg["interception"] == 1) | (reg["fumble_lost"] == 1)

    committed = reg[is_turnover].groupby(["season", "week", "posteam"]).size().rename("turnovers_committed")
    forced = reg[is_turnover].groupby(["season", "week", "defteam"]).size().rename("turnovers_forced")

    committed.index.set_names(["season", "week", "team"], inplace=True)
    forced.index.set_names(["season", "week", "team"], inplace=True)
    out = pd.concat([committed, forced], axis=1).fillna(0)
    out["turnover_margin"] = out["turnovers_forced"] - out["turnovers_committed"]
    return out.reset_index()


def red_zone_features(pbp: pd.DataFrame) -> pd.DataFrame:
    reg = pbp[pbp["season_type"] == "REG"]
    drives = (
        reg.groupby(["season", "week", "posteam", "fixed_drive"])
        .agg(drive_inside20=("drive_inside20", "max"), fixed_drive_result=("fixed_drive_result", "first"))
        .reset_index()
    )
    rz = drives[drives["drive_inside20"] == 1]
    trips = rz.groupby(["season", "week", "posteam"]).size().rename("red_zone_trips")
    tds = (
        rz[rz["fixed_drive_result"] == "Touchdown"]
        .groupby(["season", "week", "posteam"])
        .size()
        .rename("red_zone_tds")
    )

    trips.index.set_names(["season", "week", "team"], inplace=True)
    tds.index.set_names(["season", "week", "team"], inplace=True)
    out = pd.concat([trips, tds], axis=1).fillna(0)
    out["red_zone_td_pct"] = (100 * out["red_zone_tds"] / out["red_zone_trips"]).round(1)
    return out.reset_index()


def pressure_features(pbp: pd.DataFrame) -> pd.DataFrame:
    dropbacks = pbp[(pbp["season_type"] == "REG") & (pbp["qb_dropback"] == 1) & pbp["down"].notna()]
    pressured = (dropbacks["sack"] == 1) | (dropbacks["qb_hit"] == 1)

    off_db = dropbacks.groupby(["season", "week", "posteam"]).size().rename("dropbacks")
    off_pr = dropbacks[pressured].groupby(["season", "week", "posteam"]).size().rename("pressured_dropbacks")
    def_db = dropbacks.groupby(["season", "week", "defteam"]).size().rename("opp_dropbacks")
    def_pr = dropbacks[pressured].groupby(["season", "week", "defteam"]).size().rename("pressures_created")

    off_db.index.set_names(["season", "week", "team"], inplace=True)
    off_pr.index.set_names(["season", "week", "team"], inplace=True)
    def_db.index.set_names(["season", "week", "team"], inplace=True)
    def_pr.index.set_names(["season", "week", "team"], inplace=True)

    out = pd.concat([off_db, off_pr, def_db, def_pr], axis=1).fillna(0)
    out["pressure_rate_allowed"] = (100 * out["pressured_dropbacks"] / out["dropbacks"]).round(1)
    out["pressure_rate_created"] = (100 * out["pressures_created"] / out["opp_dropbacks"]).round(1)
    return out.reset_index()


def main() -> None:
    pbp = load_pbp()

    features = epa_features(pbp)
    for build in (turnover_features, red_zone_features, pressure_features):
        features = features.merge(build(pbp), on=["season", "week", "team"], how="outer")

    # A missing turnover/red-zone row means zero events that team-week, not missing
    # data (off/def EPA columns, present for every row, confirm the team played) —
    # fill those counts with 0 rather than leaving them NaN and silently dropped
    # from downstream averages.
    zero_fill_cols = [
        "turnovers_committed",
        "turnovers_forced",
        "turnover_margin",
        "red_zone_trips",
        "red_zone_tds",
    ]
    features[zero_fill_cols] = features[zero_fill_cols].fillna(0)

    features = features.sort_values(["team", "season", "week"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "team_week_features.csv"
    features.to_csv(out_path, index=False)
    print(f"team_week_features: {features.shape} -> {out_path}")

    sea = features[features["team"] == "SEA"]
    print("\nSEA season averages:")
    print(
        sea.groupby("season")[
            ["off_epa_per_play", "def_epa_per_play_allowed", "turnover_margin", "red_zone_td_pct", "pressure_rate_allowed"]
        ].mean().round(3)
    )


if __name__ == "__main__":
    main()
