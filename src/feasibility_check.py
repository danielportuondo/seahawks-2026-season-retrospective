"""Phase 0 feasibility check: confirm nflreadpy can actually fetch data here.

Not a data validation step (that's Phase 1) — just proof the network path to
nflverse's GitHub-hosted releases works from this environment.
"""

import nflreadpy as nfl

if __name__ == "__main__":
    schedules = nfl.load_schedules(seasons=[2024])
    df = schedules.to_pandas() if hasattr(schedules, "to_pandas") else schedules

    print(f"Rows returned: {len(df)}")
    print(df[["week", "away_team", "home_team", "away_score", "home_score"]].head())

    sea = df[(df["away_team"] == "SEA") | (df["home_team"] == "SEA")]
    wins = ((sea["away_team"] == "SEA") & (sea["away_score"] > sea["home_score"])) | (
        (sea["home_team"] == "SEA") & (sea["home_score"] > sea["away_score"])
    )
    print(f"\nSEA 2024 games found: {len(sea)}, wins so far in data: {wins.sum()}")
