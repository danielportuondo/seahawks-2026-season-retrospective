"""Phase 1: pull and cache raw data to /data/raw so later phases never re-pull.

Historical window: 2010-2025 (confirmed with user for the Phase 5 defense
baseline). Play-by-play is cached one file per season so a partial run
resumes cleanly instead of re-downloading everything.
"""

from pathlib import Path

import nflreadpy as nfl
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PBP_DIR = RAW_DIR / "pbp"
SEASONS = list(range(2010, 2026))


def fetch_schedules() -> None:
    path = RAW_DIR / "schedules.parquet"
    if path.exists():
        print(f"skip schedules (cached at {path})")
        return
    df = nfl.load_schedules(seasons=SEASONS).to_pandas()
    df.to_parquet(path, index=False)
    print(f"schedules: {len(df)} rows -> {path}")


def fetch_pbp() -> None:
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        path = PBP_DIR / f"{season}.parquet"
        if path.exists():
            print(f"skip pbp {season} (cached)")
            continue
        df = nfl.load_pbp(seasons=[season]).to_pandas()
        df.to_parquet(path, index=False)
        print(f"pbp {season}: {len(df)} rows -> {path}")


def fetch_player_stats() -> None:
    # "reg+post" merges a player's regular+postseason into one row whenever
    # they appeared in both (e.g. Darnold 2025) instead of giving separate
    # REG and POST rows — fine for full-season totals, but it means a clean
    # like-for-like regular-season comparison needs its own "reg"-only pull.
    for summary_level, suffix in [("reg+post", ""), ("reg", "_reg")]:
        path = RAW_DIR / f"player_stats_2024_2025{suffix}.parquet"
        if path.exists():
            print(f"skip player_stats{suffix} (cached at {path})")
            continue
        df = nfl.load_player_stats(seasons=[2024, 2025], summary_level=summary_level).to_pandas()
        df.to_parquet(path, index=False)
        print(f"player_stats{suffix}: {len(df)} rows -> {path}")


def validate_ground_truth() -> None:
    df = pd.read_parquet(RAW_DIR / "schedules.parquet")
    df = df[df["game_type"] == "REG"]

    for season, expected_wins, expected_losses in [(2024, 10, 7), (2025, 14, 3)]:
        s = df[df["season"] == season]
        sea = s[(s["away_team"] == "SEA") | (s["home_team"] == "SEA")]
        played = sea.dropna(subset=["away_score", "home_score"])
        wins = (
            ((played["away_team"] == "SEA") & (played["away_score"] > played["home_score"]))
            | ((played["home_team"] == "SEA") & (played["home_score"] > played["away_score"]))
        ).sum()
        losses = (
            ((played["away_team"] == "SEA") & (played["away_score"] < played["home_score"]))
            | ((played["home_team"] == "SEA") & (played["home_score"] < played["away_score"]))
        ).sum()
        status = "OK" if (wins, losses) == (expected_wins, expected_losses) else "MISMATCH"
        print(
            f"{season} SEA record: {wins}-{losses} "
            f"(expected {expected_wins}-{expected_losses}) [{status}]"
        )
        if status == "MISMATCH":
            raise SystemExit(
                f"Ground-truth mismatch for {season}: got {wins}-{losses}, "
                f"expected {expected_wins}-{expected_losses}. Stopping per HANDOFF.md guardrails."
            )


if __name__ == "__main__":
    fetch_schedules()
    fetch_pbp()
    fetch_player_stats()
    validate_ground_truth()
