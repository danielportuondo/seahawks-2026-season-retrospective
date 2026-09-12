"""Seahawks 2024->2025 retrospective dashboard.

Reads only from /outputs and /data/processed (Phases 2-7 results, cached in
prior runs) -- no live recomputation of any statistics.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FEATURES_CSV = ROOT / "data" / "processed" / "team_week_features.csv"

st.set_page_config(
    page_title="Seahawks 2024→2025 Retrospective",
    page_icon="\U0001f985",
    layout="wide",
)


@st.cache_data
def load_json(name: str) -> dict:
    with open(OUTPUTS / name) as f:
        return json.load(f)


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_CSV)


pyth = load_json("pythagorean_results.json")
personnel = load_json("personnel_scheme_results.json")
defense = load_json("defense_anomaly_results.json")
turnover = load_json("turnover_rate_model_results.json")
decomp = load_json("decomposition_results.json")
features = load_features()

METRIC_OPTIONS = {
    "Offensive EPA/play (higher = better)": "off_epa_per_play",
    "Defensive EPA/play allowed (lower = better)": "def_epa_per_play_allowed",
    "Turnover margin": "turnover_margin",
    "Red zone TD%": "red_zone_td_pct",
    "Pressure rate allowed (lower = better)": "pressure_rate_allowed",
    "Pressure rate created (higher = better)": "pressure_rate_created",
}

st.title("\U0001f985 From missing the playoffs to Super Bowl champions")
st.caption(
    "A statistical retrospective on the 2024→2025 Seattle Seahawks — "
    "Super Bowl LX champions"
)

tab_story, tab_method = st.tabs(["The Story", "Methodology"])

# ---------------------------------------------------------------------------
# Story tab
# ---------------------------------------------------------------------------
with tab_story:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("2024 record", "10–7", "Missed playoffs (tiebreaker)", delta_color="off")
    c2.metric("2025 record", "14–3", "NFC West champs", delta_color="off")
    c3.metric("Point differential/game", "+11.2", "vs. +0.4 in 2024")
    c4.metric("Playoff turnovers", "0", "First-ever clean postseason", delta_color="off")

    st.divider()

    st.header("2024: a good team that still went home early")
    st.markdown(
        """
The 2024 Seahawks won 10 games and missed the playoffs anyway — the only
10-win team to miss the playoffs since the NFL went to a 17-game schedule,
losing the NFC West tiebreaker to the Rams on strength of victory. A standard
model of "how many games should a team with this scoring margin win" says this
team was worth about **8.7 wins**, not 10 — Seattle actually
*overachieved* relative to its underlying performance and still missed the
playoffs, purely on a tiebreaker.

Geno Smith was at quarterback under OC Ryan Grubb, and the offensive line
struggled to protect him — pressured on nearly 1 in 6 dropbacks (16.8%).
"""
    )
    st.image(str(OUTPUTS / "pythagorean_actual_vs_expected.png"), width="stretch")

    st.header("2025: everything clicked")
    st.markdown(
        """
The 2025 Seahawks won 14 games, the NFC West, and Super Bowl LX. Sam Darnold
took over at quarterback (signed after Smith was traded), Klint Kubiak became
OC, and the same DC (Aden Durde) got dramatically better results from
essentially the same defensive unit.

Not everything improved cleanly: Darnold's completion percentage was actually
*lower* than Smith's (67.7% vs. 70.4%), and his interception rate was
slightly *higher* (2.94% vs. 2.6%). The jump was concentrated in a smaller
number of things that mattered enormously — pass protection (pressure
rate 16.8% → 12.1%, sacks allowed 50 → 27) and overall per-play
efficiency.
"""
    )
    st.image(str(OUTPUTS / "personnel_scheme_comparison.png"), width="stretch")

    st.header("The defense that carried them — but let's not oversell it")
    st.markdown(
        """
The "Dark Side" defense was the best defense in the NFL in 2025 by both
EPA/play and points allowed per game. It's tempting to call it the greatest
defense ever — that didn't hold up under a closer look. Ranked against
every team-defense back to 2010 (over 500 team-seasons), it lands in the
**top 10%**, but not the top 10 outright. That distinction belongs to
Seattle's *own* 2013 "Legion of Boom" defense, which remains the more
statistically extreme unit in franchise history. Elite, genuinely — just
not unprecedented.
"""
    )
    dcol1, dcol2 = st.columns(2)
    dcol1.image(str(OUTPUTS / "defense_anomaly_radar.png"), width="stretch")
    dcol2.image(str(OUTPUTS / "defense_anomaly_historical_rank.png"), width="stretch")

    st.header("The turnover-free playoff run")
    st.markdown(
        """
Sam Darnold led the NFL in turnovers during the 2025 regular season — 20
of them across 17 games. Then, across three playoff games, he committed
**zero** — the first Super Bowl champion ever to complete an entire
postseason without one.
"""
    )
    p_zero = turnover["p_zero_turnovers"]["primary_per_dropback"]["p_zero_poisson"]
    odds = turnover["p_zero_turnovers"]["primary_per_dropback"]["odds_against_1_in"]
    tcol1, tcol2 = st.columns([1, 2])
    with tcol1:
        st.metric("P(zero turnovers, 3 playoff games)", f"{p_zero:.1%}", f"~1 in {odds:.0f}")
        st.markdown(
            """
One popular explanation is that Darnold "settled down" as the season wore
on. The data doesn't support that: his turnover rate was actually *higher*
in the second half of the season than the first. The clean playoff run was a
sudden break from his season-long pattern, not the tail end of a gradual
improvement.
"""
        )
    with tcol2:
        st.image(str(OUTPUTS / "turnover_probability.png"), width="stretch")
    st.image(str(OUTPUTS / "turnover_rolling_rate.png"), width="stretch")

    st.header("So what actually explains the jump?")
    st.markdown(
        """
Across all 32 NFL teams over these two seasons, a team's per-game scoring
margin is very well explained by two things: offensive efficiency and
defensive efficiency (per play). Applying that relationship to Seattle
specifically, the improved offense and improved defense together account for
essentially the *entire* jump in Seattle's scoring margin from 2024 to 2025.
Turnover luck, red-zone execution, and pass rush played only minor,
statistically inconclusive roles by comparison.
"""
    )
    ecol1, ecol2 = st.columns(2)
    ecol1.image(str(OUTPUTS / "decomposition_waterfall.png"), width="stretch")
    ecol2.image(str(OUTPUTS / "decomposition_feature_importance.png"), width="stretch")

    st.header("Season trend explorer")
    st.caption("SEA's weekly team features, 2024 vs. 2025 — from `team_week_features.csv`.")
    metric_label = st.selectbox("Metric", list(METRIC_OPTIONS.keys()))
    metric_col = METRIC_OPTIONS[metric_label]
    sea = features[features["team"] == "SEA"].copy()
    sea["season"] = sea["season"].astype(str)
    fig = px.line(
        sea,
        x="week",
        y=metric_col,
        color="season",
        markers=True,
        labels={"week": "Week", metric_col: metric_label, "season": "Season"},
    )
    fig.update_layout(hovermode="x unified", legend_title_text="Season")
    st.plotly_chart(fig, width="stretch")

    st.header("The honest caveats")
    st.markdown(
        """
- The 2025 defense is elite, not literally the best ever — a claim the
  data didn't support once tested, so the framing was changed rather than
  keeping the more dramatic (and wrong) version.
- Not every underlying stat improved: red-zone TD% and pressures created
  both dipped slightly — the turnaround was concentrated in specific
  things, not a uniform team-wide leap.
- "1 in 34" comes from a model built on one player's one season, not a law
  of nature — a different reasonable modeling choice could plausibly land
  in the 2–4% range. The point is the order of magnitude: genuinely
  improbable, not "he just got a little lucky."
- Both 2024 and 2025 show Seattle winning slightly more games than its
  scoring margin alone would predict — a recurring team characteristic,
  not something 2025 erased.
"""
    )

# ---------------------------------------------------------------------------
# Methodology tab
# ---------------------------------------------------------------------------
with tab_method:
    st.caption(
        "All figures below are read directly from this project's output files "
        "(`outputs/*.json`), not re-derived for this dashboard."
    )

    st.header("Data and scope")
    st.markdown(
        """
Play-by-play, schedule, and player-stat data come from `nflverse` via
`nflreadpy` (CC-BY-4.0), covering 2010–2025 for the historical baseline
used in the defense anomaly detection, and 2024–2025 specifically for
everything else. Regular-season games only, unless noted.
"""
    )

    st.header("Pythagorean win expectation")
    st.markdown("`PF^2.37 / (PF^2.37 + PA^2.37)` — the standard NFL exponent.")
    pyth_df = pd.DataFrame(pyth).T
    pyth_df.index.name = "season"
    st.dataframe(pyth_df, width="stretch")

    st.header("Personnel and scheme deltas")
    personnel_df = pd.DataFrame(personnel).T
    personnel_df.index.name = "season"
    st.dataframe(personnel_df, width="stretch")

    st.header("Engineered features (Phase 4)")
    st.markdown(
        """
Team-week features from play-by-play: offensive/defensive EPA per play
(core scrimmage plays only, 0.05 ≤ win probability ≤ 0.95, garbage
time excluded), turnover margin, red-zone TD rate, and pressure rate.
"""
    )
    sea_season_avg = (
        features[features["team"] == "SEA"]
        .groupby("season")[
            [
                "off_epa_per_play",
                "def_epa_per_play_allowed",
                "turnover_margin",
                "red_zone_td_pct",
                "pressure_rate_allowed",
                "pressure_rate_created",
            ]
        ]
        .mean()
        .round(3)
    )
    st.dataframe(sea_season_avg, width="stretch")

    st.header("Defense anomaly detection (Phase 5)")
    meth = defense["methodology"]
    st.markdown(
        f"""
Built a team-season defensive dataset for all **{meth["n_team_seasons"]}**
team-seasons from {meth["seasons"][0]}–{meth["seasons"][1]} (EPA/play
allowed, sack rate, takeaways/drive, points allowed/game), z-scored **within
each season** before pooling across years, signs oriented so higher = better
defense. Mahalanobis distance and an Isolation Forest were run as
cross-checks on top of the composite z-score.
"""
    )
    m1, m2 = st.columns(2)
    with m1:
        st.subheader("SEA 2024")
        st.json(defense["SEA_2024"], expanded=False)
    with m2:
        st.subheader("SEA 2025")
        st.json(defense["SEA_2025"], expanded=False)
    st.markdown("**Top 10 team-seasons by composite z-score, 2010–2025:**")
    st.dataframe(pd.DataFrame(defense["top10_by_composite_z"]), width="stretch", hide_index=True)

    st.header("Turnover rate model (Phase 6)")
    st.markdown(turnover["framing"]["not_extreme_value_theory"])
    st.markdown(
        f"""
Regular-season rate: **{turnover["season_turnovers"]}** turnovers over
{turnover["dropbacks"]} dropbacks ({turnover["turnover_rate_per_dropback"]:.2%}) /
{turnover["drives"]} drives ({turnover["turnover_rate_per_drive"]:.2%}).
Distribution choice: **{turnover["distribution"]["chosen_distribution"]}**
({turnover["distribution"]["reason"]})
"""
    )
    r1, r2, r3 = st.columns(3)
    r1.metric(
        "P(zero turnovers) — Poisson",
        f"{turnover['p_zero_turnovers']['primary_per_dropback']['p_zero_poisson']:.2%}",
    )
    r2.metric(
        "P(zero turnovers) — neg. binomial (conservative bound)",
        f"{turnover['p_zero_turnovers']['primary_per_dropback']['p_zero_negative_binomial']:.2%}",
    )
    r3.metric(
        "'Got safer' narrative supported?",
        "No" if not turnover["rolling_analysis"]["narrative_supported"] else "Yes",
        turnover["rolling_analysis"]["direction"],
        delta_color="off",
    )
    st.markdown("**Sensitivity to modeling assumptions:**")
    st.dataframe(pd.DataFrame(turnover["sensitivity"]).T, width="stretch")

    st.header("Regression decomposition (Phase 7)")
    ols = decomp["ols_point_diff_per_game"]
    st.markdown(
        f"""
OLS of point differential/game on six standardized Phase 4 features, across
32 teams × 2 seasons (n={ols["n"]}). **R² = {ols["r2"]:.3f}**
(adjusted {ols["r2_adj"]:.3f}). Linear regression was used deliberately
instead of a gradient-boosted model with SHAP: at n={ols["n"]}, a tree
ensemble would overfit, and a linear model's coefficients already *are* the
decomposition — no separate explainability layer is needed on top of them.
"""
    )
    coef_df = pd.DataFrame(
        {
            "coefficient": ols["coefficients"],
            "std_error": ols["std_errors"],
            "p_value": ols["p_values"],
        }
    ).drop("intercept")
    st.dataframe(coef_df, width="stretch")

    sea_decomp = decomp["sea_decomposition"]
    st.markdown(
        f"""
SEA's point differential/game rose from **{sea_decomp["sea_point_diff_per_game_2024"]:.2f}**
to **{sea_decomp["sea_point_diff_per_game_2025"]:.2f}**
(actual change: {sea_decomp["actual_change"]:.2f}; model residual:
{sea_decomp["residual_unexplained"]:.2f}, or
{sea_decomp["residual_pct_of_actual_change"]:.1f}% of the actual change).
"""
    )
    st.markdown("**Per-feature contribution to the point-differential change:**")
    per_feature_df = pd.DataFrame(sea_decomp["per_feature"]).T
    st.dataframe(per_feature_df, width="stretch")

    st.header("Limitations")
    st.markdown(
        """
- Phase 4–7's engineered features are aggregated with equal weight per
  game, not weighted by play volume within a season.
- The regression decomposition is correlational, built on 64 observations,
  and does not control for strength of schedule, injuries, or other omitted
  context.
- The defense anomaly detection and turnover rate model are both
  single-season, single-team analyses layered on a historical baseline —
  neither is a causal claim, only a statement about where this team-season
  sits relative to a well-defined comparison set.
"""
    )
