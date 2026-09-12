"""Phase 3: personnel/scheme deltas, 2024 (Geno Smith / Ryan Grubb) vs.
2025 (Sam Darnold / Klint Kubiak). Scene-setting, not the analytical core —
a comparison table + chart is enough here.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from chart_style import ACTION_GREEN, WOLF_GREY, apply_scoreboard_style, fig_size

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"

STARTERS = {2024: "Geno Smith", 2025: "Sam Darnold"}
OC = {2024: "Ryan Grubb", 2025: "Klint Kubiak"}


def qb_splits() -> dict:
    df = pd.read_parquet(RAW_DIR / "player_stats_2024_2025_reg.parquet")
    results = {}
    for season, name in STARTERS.items():
        row = df[(df["recent_team"] == "SEA") & (df["player_display_name"] == name)].iloc[0]
        completions, attempts = row["completions"], row["attempts"]
        interceptions = row["passing_interceptions"]
        fumbles_lost = row["rushing_fumbles_lost"] + row["sack_fumbles_lost"]
        results[season] = {
            "qb": name,
            "oc": OC[season],
            "completions": int(completions),
            "attempts": int(attempts),
            "completion_pct": round(100 * completions / attempts, 1),
            "passing_yards": int(row["passing_yards"]),
            "interceptions": int(interceptions),
            "int_rate_pct": round(100 * interceptions / attempts, 2),
            "fumbles_lost": int(fumbles_lost),
            "sacks_suffered": int(row["sacks_suffered"]),
        }
    return results


def pressure_rate_allowed() -> dict:
    results = {}
    for season in STARTERS:
        pbp = pd.read_parquet(RAW_DIR / "pbp" / f"{season}.parquet")
        off = pbp[(pbp["posteam"] == "SEA") & (pbp["season_type"] == "REG") & (pbp["qb_dropback"] == 1)]
        dropbacks = len(off)
        pressured = ((off["sack"] == 1) | (off["qb_hit"] == 1)).sum()
        results[season] = {
            "dropbacks": int(dropbacks),
            "pressured_dropbacks": int(pressured),
            "pressure_rate_pct": round(100 * pressured / dropbacks, 1),
        }
    return results


def main() -> None:
    qb = qb_splits()
    pressure = pressure_rate_allowed()
    combined = {season: {**qb[season], **pressure[season]} for season in STARTERS}

    print("2024 vs 2025 offensive personnel/scheme comparison:")
    for season, row in combined.items():
        print(f"  {season} ({row['qb']} / OC {row['oc']}): "
              f"comp%={row['completion_pct']}, yds={row['passing_yards']}, "
              f"INT rate={row['int_rate_pct']}%, pressure rate allowed={row['pressure_rate_pct']}%")

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "personnel_scheme_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    apply_scoreboard_style()
    fig, axes = plt.subplots(1, 3, figsize=fig_size(4.0))
    seasons = list(combined.keys())
    labels = [f"{s}\n{combined[s]['qb']}" for s in seasons]
    season_colors = [WOLF_GREY, ACTION_GREEN]

    axes[0].bar(labels, [combined[s]["completion_pct"] for s in seasons], color=season_colors)
    axes[0].set_title("Completion %")

    axes[1].bar(labels, [combined[s]["int_rate_pct"] for s in seasons], color=season_colors)
    axes[1].set_title("INT rate (%)")

    axes[2].bar(labels, [combined[s]["pressure_rate_pct"] for s in seasons], color=season_colors)
    axes[2].set_title("Pressure rate allowed (%)")

    fig.suptitle("Seahawks Offense: 2024 vs 2025")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "personnel_scheme_comparison.png", dpi=150)
    print(f"\nChart saved to {OUT_DIR / 'personnel_scheme_comparison.png'}")


if __name__ == "__main__":
    main()
