"""Phase 2: the 2024 puzzle — did the Seahawks' point differential predict
a 10-7 record, or did they under/over-perform their Pythagorean expectation?
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from chart_style import ACTION_GREEN, WOLF_GREY, apply_scoreboard_style, fig_size
from pythagorean import expected_wins, pythagorean_win_pct

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"

ACTUAL_RECORDS = {2024: (10, 7), 2025: (14, 3)}


def team_points(df: pd.DataFrame, season: int, team: str = "SEA") -> tuple[float, float, int]:
    s = df[(df["season"] == season) & (df["game_type"] == "REG")]
    s = s[(s["away_team"] == team) | (s["home_team"] == team)].dropna(subset=["away_score", "home_score"])

    points_for = (s["away_score"].where(s["away_team"] == team, 0) + s["home_score"].where(s["home_team"] == team, 0)).sum()
    points_against = (s["home_score"].where(s["away_team"] == team, 0) + s["away_score"].where(s["home_team"] == team, 0)).sum()
    return float(points_for), float(points_against), len(s)


def main() -> None:
    schedules = pd.read_parquet(RAW_DIR / "schedules.parquet")

    results = {}
    for season, (actual_wins, actual_losses) in ACTUAL_RECORDS.items():
        pf, pa, games = team_points(schedules, season)
        win_pct = pythagorean_win_pct(pf, pa)
        exp_wins = expected_wins(pf, pa, games)
        results[season] = {
            "points_for": pf,
            "points_against": pa,
            "games": games,
            "actual_wins": actual_wins,
            "actual_losses": actual_losses,
            "pythagorean_win_pct": round(win_pct, 4),
            "pythagorean_expected_wins": round(exp_wins, 2),
            "wins_over_expectation": round(actual_wins - exp_wins, 2),
        }
        print(
            f"{season}: PF={pf:.0f} PA={pa:.0f} -> expected {exp_wins:.1f} wins, "
            f"actual {actual_wins} ({'over' if actual_wins > exp_wins else 'under'}achieved by "
            f"{abs(actual_wins - exp_wins):.1f})"
        )

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "pythagorean_results.json", "w") as f:
        json.dump(results, f, indent=2)

    seasons = list(results.keys())
    actual = [results[s]["actual_wins"] for s in seasons]
    expected = [results[s]["pythagorean_expected_wins"] for s in seasons]

    apply_scoreboard_style()
    x = range(len(seasons))
    width = 0.22
    fig, ax = plt.subplots(figsize=fig_size(4.6))
    ax.bar([i - width / 2 for i in x], actual, width, label="Actual wins", color=ACTION_GREEN)
    ax.bar([i + width / 2 for i in x], expected, width, label="Pythagorean expected wins", color=WOLF_GREY)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in seasons])
    ax.set_ylabel("Wins")
    ax.set_title("Seahawks: Actual vs. Pythagorean-Expected Wins")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pythagorean_actual_vs_expected.png", dpi=150)
    print(f"\nChart saved to {OUT_DIR / 'pythagorean_actual_vs_expected.png'}")


if __name__ == "__main__":
    main()
