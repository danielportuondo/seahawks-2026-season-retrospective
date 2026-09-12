"""Shared matplotlib styling for the dashboard's "Scoreboard" visual direction.

Imported by the phase 2/3/5/6/7 chart-generation scripts so all 8 static PNGs
read as one visual system with the Streamlit dashboard's theme, rather than
each carrying matplotlib's default palette. Only touches figure/axes chrome
and explicit series colors -- never the statistical computation above it.
"""

import matplotlib.pyplot as plt

NAVY = "#0B162A"
PANEL = "#0F1E38"
ACTION_GREEN = "#69BE28"
WOLF_GREY = "#A5ACAF"
OFF_WHITE = "#F5F6F7"
ALERT_RED = "#D6432D"
AMBER = "#F2C14E"
ACCENT_BLUE = "#4C8FBF"
GRID = "#1C2C48"

# Convention used throughout: ACTION_GREEN = 2025 / the good outcome,
# WOLF_GREY = 2024 / baseline, ALERT_RED = negative, AMBER = secondary warning.
CATEGORICAL = [ACTION_GREEN, WOLF_GREY, ACCENT_BLUE, AMBER]

# Every PNG is displayed at one fixed on-page width, so a figure's apparent type
# size is its point size divided by its width in inches -- not its point size.
# Authoring every figure at the same width is what makes a `fontsize=9` label
# mean the same thing in all 17 charts; heights vary freely.
FIG_WIDTH = 11.0


def fig_size(height: float) -> tuple[float, float]:
    return (FIG_WIDTH, height)


def apply_scoreboard_style() -> None:
    plt.rcParams.update({
        # Tuned so a FIG_WIDTH figure renders type at roughly the dashboard's
        # own scale: titles just under a section heading, ticks under body copy.
        "font.size": 10,
        "axes.titlesize": 13.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "figure.facecolor": NAVY,
        "savefig.facecolor": NAVY,
        "axes.facecolor": PANEL,
        "axes.edgecolor": WOLF_GREY,
        "axes.labelcolor": OFF_WHITE,
        "axes.titlecolor": OFF_WHITE,
        "axes.titleweight": "bold",
        "text.color": OFF_WHITE,
        "xtick.color": WOLF_GREY,
        "ytick.color": WOLF_GREY,
        "grid.color": GRID,
        "legend.facecolor": PANEL,
        "legend.edgecolor": WOLF_GREY,
        "legend.labelcolor": OFF_WHITE,
    })
