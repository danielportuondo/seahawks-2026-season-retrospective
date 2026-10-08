"""Seahawks 2024->2025 retrospective dashboard.

Reads only from /outputs and /data/processed (Phases 2-7 results, cached in
prior runs) -- no live recomputation of any statistics.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"
FEATURES_CSV = PROCESSED / "team_week_features.csv"

sys.path.insert(0, str(ROOT / "src"))
from phase7_decomposition import load_team_season_features

st.set_page_config(
    page_title="Seahawks 2024→2025 Retrospective",
    page_icon="\U0001f985",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Streamlit's default top padding pushes the hero a full screen-inch down. */
[data-testid="stMain"] .block-container { padding-top: 1.2rem; }

.hero-band {
    background: radial-gradient(ellipse 800px 260px at 15% 0%, rgba(105,190,40,.20), transparent 65%), #0F1E38;
    padding: 1.8rem 1.8rem 1.5rem;
    border-bottom: 4px solid #69BE28;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 20px;
}
.hero-band svg { width: 56px; height: 56px; flex-shrink: 0; }
.hero-band h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    color: #F5F6F7;
    margin: 0 0 6px;
    line-height: 1.05;
    letter-spacing: 0.005em;
}
.hero-band p { color: #C3C9CC; margin: 0; font-size: 1rem; }

[data-testid="stMetric"] {
    background: #0F1E38;
    border-bottom: 4px solid #69BE28;
    padding: 14px 14px 10px;
}
[data-testid="stMetricValue"] { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #C3C9CC; }

/* delta_color="off" still ships a trend arrow and a pill; these captions are
   context, not movement, so both read as a change that never happened. */
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] {
    background: none !important;
    padding-left: 0 !important;
    color: #A5ACAF;
    font-size: 0.82rem;
}

/* Green marks the hero and the KPI row; a green divider on top of that leaves
   the accent marking nothing. */
hr {
    height: 1px !important;
    background-color: #22324E !important;
    border: none !important;
    opacity: 1 !important;
    margin: 1.6rem 0 0.4rem !important;
}

h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; }
h2 { font-size: 1.8rem !important; line-height: 1.15; margin-top: 2rem !important; }
h3 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #E6E9EA;
    letter-spacing: 0.02em;
    margin-top: 1.9rem !important;
}

/* Prose caps at a readable measure; charts, tables and images stay full width. */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { max-width: 44rem; line-height: 1.62; }
[data-testid="stMarkdownContainer"] li { margin-bottom: 0.35rem; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #C3C9CC; }

/* The static charts are authored at a fixed pixel size; stretching them to a
   1280px column renders their internal titles larger than the page's own. */
[data-testid="stImage"] img { max-width: 880px !important; }

@media (max-width: 640px) {
    .hero-band {
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
        padding: 1.2rem 1.1rem 1rem;
    }
    .hero-band svg { width: 40px; height: 40px; }
    .hero-band h1 { font-size: 1.55rem; }
    h2 { font-size: 1.45rem !important; margin-top: 2rem !important; }
    h3 { font-size: 1.1rem !important; }
    /* Four full-width tiles otherwise cost two screens before any content. */
    [data-testid="stMetric"] { padding: 10px 12px 8px; }
    [data-testid="stMetricValue"] { font-size: 1.9rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_json(name: str) -> dict:
    with open(OUTPUTS / name) as f:
        return json.load(f)


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_CSV)


@st.cache_data
def load_processed(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / name)


pyth = load_json("pythagorean_results.json")
personnel = load_json("personnel_scheme_results.json")
defense = load_json("defense_anomaly_results.json")
turnover = load_json("turnover_rate_model_results.json")
decomp = load_json("decomposition_results.json")
historical = load_json("historical_percentiles.json")
control = load_json("game_control.json")
defense_deep = load_json("defense_deep_dive.json")
counterfactual = load_json("counterfactual.json")
players = load_json("players_deep_dive.json")
scheme = load_json("scheme_deep_dive.json")
features = load_features()

PLOT_BG = "#0F1E38"
GRID = "#1C2C48"
GREEN = "#69BE28"
GREY = "#A5ACAF"
AMBER = "#F2C14E"
RED = "#D6432D"
INK = "#F5F6F7"

# Matches the image cap: charts, tables and prose all share one left column.
NARROW_TABLE = 880


def ordinal(n: int) -> str:
    """1 -> 1st, 21 -> 21st, 11 -> 11th. The teens are the whole reason this exists."""
    n = int(n)
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def style_fig(fig, height: int | None = None):
    """The scoreboard Plotly treatment, applied identically to every figure."""
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        font_color=INK,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig

METRIC_OPTIONS = {
    "Offensive EPA/play (higher = better)": "off_epa_per_play",
    "Defensive EPA/play allowed (lower = better)": "def_epa_per_play_allowed",
    "Turnover margin": "turnover_margin",
    "Red zone TD%": "red_zone_td_pct",
    "Pressure rate allowed (lower = better)": "pressure_rate_allowed",
    "Pressure rate created (higher = better)": "pressure_rate_created",
}

st.markdown(
    """
<div class="hero-band">
  <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
    <polygon points="32,4 52,28 40,26 56,58 32,38 8,58 24,26 12,28" fill="#69BE28" stroke="#0B162A" stroke-width="1.5"/>
  </svg>
  <div>
    <h1>From missing the playoffs to Super Bowl champions</h1>
    <p>A statistical retrospective on the 2024→2025 Seattle Seahawks — Super Bowl LX champions</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_story, tab_alltime, tab_players, tab_coach, tab_method = st.tabs(
    ["The Story", "All-Time Great?", "Players", "Coach & Scheme", "Methodology"]
)

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

    st.info(
        """
**The short answer.** Seattle went 10–7 to 14–3 and won Super Bowl LX because
it got better at the two things that decide football games: yards per play
gained and yards per play allowed. Fixing the pass protection was the single
biggest lever. The takeaways everyone credited — turnovers, splash plays —
were never the story. This was a genuinely elite team that also caught a
normal amount of good luck in a 17-game sample.
"""
    )

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
The 2025 Seahawks won 14 games, the NFC West, and Super Bowl LX — and they did
it after gutting the skill positions that defined the previous era. In one week
of March 2025 the front office released Tyler Lockett, traded DK Metcalf to
Pittsburgh, and traded Geno Smith to Las Vegas. Sam Darnold came in at
quarterback, Cooper Kupp signed on, Klint Kubiak took over as OC, and Jaxon
Smith-Njigba — the third option on his own depth chart a year earlier — became
the whole passing game. He responded by leading the NFL in receiving yards and
breaking Metcalf's franchise record along the way.

The defense barely changed personnel at all. Mike Macdonald kept calling it,
and essentially the same unit played dramatically better.

The real fix was up front. Seattle rebuilt the interior of the offensive line
around first-round rookie left guard **Grey Zabel** and new starting center
**Jalen Sundell**, with **Charles Cross** and a healthy **Abraham Lucas** at
the tackles and **Anthony Bradford** at right guard. A line that had been among
the league's worst in 2024 turned into a middle-of-the-pack pass-blocking unit,
and that was enough: pressure rate fell 16.8% → 12.1% and sacks allowed went
50 → 27.

Not everything improved cleanly. Darnold's completion percentage was actually
*lower* than Smith's (67.7% vs. 70.4%), and his interception rate was slightly
*higher* (2.94% vs. 2.6%). The jump was concentrated in a small number of
things that mattered enormously — protection, and overall per-play efficiency.
"""
    )
    st.image(str(OUTPUTS / "personnel_scheme_comparison.png"), width="stretch")

    st.header("The defense that carried them — but let's not oversell it")
    st.markdown(
        """
The "Dark Side" defense was the best defense in the NFL in 2025 by both
EPA/play and points allowed per game. *(EPA/play is just "how much did the
average snap help or hurt your chances of scoring" — a defense with a negative
number is taking points off the board every time it lines up.)*

It's tempting to call it the greatest defense ever. That didn't survive a
closer look. Ranked against every team-defense back to 2010, it lands in the
**top 10%**, but not the top 10 outright.

**So, better than the Legion of Boom or not?** Both, depending on the question,
and it's worth saying plainly instead of hedging:

- **Over a full season, no.** Seattle's own 2013 defense is still the more
  statistically extreme unit in franchise history.
- **Over its best stretch, yes.** The 2025 defense's best eight-game run beat
  anything the 2013 group put together.

That is a real distinction, not a dodge: 2013 was relentless for a whole year,
2025 hit a higher peak. Elite either way — just not unprecedented.

*(A note on the yardstick: this defensive comparison runs back to 2010, while
the all-time rankings elsewhere on the site go back to 1999. The anomaly model
leans on sack and pressure data that nflverse doesn't record reliably before
2010, so it gets the shorter window on purpose rather than ranking 2025 against
seasons where half the inputs don't exist.)*
"""
    )
    st.image(str(OUTPUTS / "defense_anomaly_radar.png"), width="stretch")
    # Three panels in one figure: at half-column width its axis labels are
    # unreadable, so it gets the full column to itself.
    st.image(str(OUTPUTS / "defense_anomaly_historical_rank.png"), width="stretch")

    st.header("The night the season turned")
    st.markdown(
        """
If you want one game, it's **Week 16, Thursday night, Lumen Field, against the
Rams** — both teams 11–3, the NFC's top seed on the table. Seattle trailed
**30–14** in the fourth quarter and won **38–37** in overtime. After Los
Angeles scored to open the extra period, Seattle answered with a nine-play,
65-yard drive, took the touchdown, and went for two rather than play for
another possession. The conversion looked incomplete live; on review it was
ruled a backward pass that had deflected off a Rams defender, and therefore a
live ball. Good. The Rams are still mad about it.

That one result flipped the conference: Los Angeles slid from the No. 1 seed to
No. 5, Seattle took the top spot, and every game of the run that followed was
played at Lumen.
"""
    )

    st.header("How the title was actually won")
    st.markdown(
        """
Three games, three very different kinds of win, and the site owes you more than
a dropdown about them.

**Divisional — Seattle 41, San Francisco 6 (Jan 17, Lumen Field).** The least
competitive playoff game of the year. Rashid Shaheed took the opening kickoff
95 yards, it was 17–0 after a quarter, and Seattle never trailed for a single
second. Kenneth Walker III ran for 116 yards and three touchdowns. San
Francisco turned it over three times; Seattle not once. Darnold barely had to
play — 12 of 17, 124 yards.

**NFC Championship — Seattle 31, Los Angeles 27 (Jan 25, Lumen Field).** The
hard one. Darnold threw three touchdowns and the defense made the stop that
mattered: McVay and Stafford went for it on fourth-and-4 from the Seattle
6-yard line late rather than take the points, and the Dark Side kept them out.
Third meeting of the year with the Rams, third one decided by one score.

**Super Bowl LX — Seattle 29, New England 13 (Feb 8, Levi's Stadium).** Eleven
years after Super Bowl XLIX, against the same franchise, Seattle got the
rematch and never made it close. Jason Myers kicked three first-half field
goals (41, 39, 33) and that was the entire first half — New England did not
score until the fourth quarter. Seattle sacked Drake Maye six times with eight
tackles for loss. The clincher: Devon Witherspoon came free on a blitz and hit
Maye as he threw, Uchenna Nwosu caught the ball out of the air and ran it back
45 yards, 29–7 with 4:27 to go. (Scorers initially ruled it a strip-sack and
later changed it to an interception.)

**Kenneth Walker III was the MVP** — 27 carries, 135 rushing yards, 161 yards
from scrimmage, and the first running back to win the award since Terrell Davis
in Super Bowl XXXII. Three playoff games, zero Seattle turnovers.
"""
    )

    st.header("The turnover-free playoff run")
    st.markdown(
        """
Sam Darnold led the NFL in turnovers during the 2025 regular season — 20
of them across 17 games. Then, across three playoff games, he committed
**zero** — the first Super Bowl champion ever to complete an entire
postseason without one.

How unlikely was that? It depends on how you count his workload. Projecting
his season turnover rate across three more games puts it at **2.9%, about 1 in
34**. Counting the 99 dropbacks he actually took in those games puts it at
**2.2%, about 1 in 46**. Same event, two reasonable denominators — call it
somewhere between 1 in 35 and 1 in 45. The point is the order of magnitude, not
the decimal.
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
margin tracks two things almost perfectly: offensive efficiency and defensive
efficiency (per play). Applying that relationship to Seattle, the improved
offense and improved defense together account for **most** of the jump in
scoring margin — they predict a 12.8-point swing against the 11.2 points that
actually happened, leaving about 18% unaccounted for. Turnover luck, red-zone
execution, and pass rush played only minor, statistically inconclusive roles by
comparison.

**An honest caveat about that 92%.** This model is better understood as
accounting than as discovery. EPA is itself built out of expected *points*, so
regressing point differential on EPA/play is close to circular — a very high
fit is the floor here, not a finding. What the model is genuinely good for is
*splitting* the improvement into offense versus defense, not proving that
efficiency causes winning. Read the chart below as "here is where the margin
came from," not "here is why they won."

One more wrinkle: the model says Seattle's underlying performance jumped by
slightly *more* than the record shows — the team may have quietly left a little
improvement on the table.
"""
    )
    st.image(str(OUTPUTS / "decomposition_waterfall.png"), width="stretch")
    st.image(str(OUTPUTS / "decomposition_feature_importance.png"), width="stretch")

    st.header("Week by week, 2024 vs. 2025")
    metric_label = st.selectbox("Metric", list(METRIC_OPTIONS.keys()))
    metric_col = METRIC_OPTIONS[metric_label]
    sea = features[features["team"] == "SEA"].copy()
    sea["season"] = sea["season"].astype(str)

    # Raw weekly values cross constantly and read as noise; the season level is
    # the actual claim, so the mean leads and a 4-week mean carries the shape.
    fig = go.Figure()
    season_means = {}
    for season, color in (("2024", "#A5ACAF"), ("2025", "#69BE28")):
        s = sea[sea["season"] == season].sort_values("week")
        if s.empty:
            continue
        season_means[season] = s[metric_col].mean()
        fig.add_trace(
            go.Scatter(
                x=s["week"],
                y=s[metric_col],
                mode="markers",
                marker=dict(color=color, size=7, opacity=0.35),
                name=f"{season} weekly",
                hovertemplate=f"{season} wk %{{x}}: %{{y:.3f}}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=s["week"],
                y=s[metric_col].rolling(4, min_periods=1).mean(),
                mode="lines",
                line=dict(color=color, width=3),
                name=f"SEA {season}",
                hovertemplate=f"{season} 4-wk avg: %{{y:.3f}}<extra></extra>",
            )
        )

    for season, color in (("2024", "#A5ACAF"), ("2025", "#69BE28")):
        if season in season_means:
            fig.add_hline(
                y=season_means[season],
                line=dict(color=color, width=1, dash="dot"),
                annotation_text=f"{season} season avg {season_means[season]:.3f}",
                annotation_position="right",
                annotation_font_color=color,
            )

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Week",
        yaxis_title=metric_label,
        plot_bgcolor="#0F1E38",
        paper_bgcolor="#0F1E38",
        font_color="#F5F6F7",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(r=170),
    )
    fig.update_xaxes(gridcolor="#1C2C48")
    fig.update_yaxes(gridcolor="#1C2C48")
    st.plotly_chart(fig, width="stretch")

    if len(season_means) == 2:
        gap = season_means["2025"] - season_means["2024"]
        st.caption(
            f"Faint dots are single weeks, the solid line is a 4-week rolling average, "
            f"and the dotted lines are season averages. On **{metric_label}** the gap "
            f"between the two season averages is **{gap:+.3f}**. Week-to-week lines cross "
            "constantly in both seasons — single-game swings are mostly noise, and the "
            "level is what moved. From `team_week_features.csv`."
        )

    st.header("Were they actually all-time great?")
    st.markdown(
        f"""
Everything above says *what changed*. None of it says *how good this was by
historical standards* — for that you need something to measure against.
Compared with all 861 team-seasons since 1999:

- **Yes, on how games went.** Explosive plays allowed ranked
  {historical["metrics"]["def_explosive_rate_allowed"]["rank_of_n"][0]} of 861;
  clock-weighted lead ranked
  {historical["metrics"]["time_weighted_margin"]["rank_of_n"][0]}. Among Super Bowl
  champions since 2000, no one held a bigger lead for longer.
- **No, on taking the ball away.** Full-season turnover margin was
  {historical["metrics"]["turnover_margin"]["sea_2025_value"]:+.0f} — about average — and
  three-and-outs forced were ordinary too.
- **14–3 was not a fluke — but a 17-game season is a small sample.**
  Replaying the season {counterfactual["methodology"]["n_simulations"]:,} times makes 14
  wins the single most likely outcome, but only at
  {counterfactual["monte_carlo"]["win_distribution"]["14"]:.0f}%; the 90% range runs
  {counterfactual["monte_carlo"]["p05_wins"]:.0f} to
  {counterfactual["monte_carlo"]["p95_wins"]:.0f} wins.

The full breakdown, with the published claims tested one by one, is on the
**All-Time Great?** tab.
"""
    )

    st.header("The honest caveats")
    st.markdown(
        """
- The 2025 defense is elite, not literally the best ever. The "best defense
  in history" version of this story is more fun and does not survive contact
  with the data.
- Not every underlying stat improved: red-zone TD% and pressures created
  both dipped slightly — the turnaround was concentrated in specific
  things, not a uniform team-wide leap.
- The playoff turnover odds come from a model built on one player's one
  season, not a law of nature — other reasonable modeling choices land
  anywhere in the 2–4% range. The point is the order of magnitude: genuinely
  improbable, not "he just got a little lucky."
- Both 2024 and 2025 show Seattle winning slightly more games than its
  scoring margin alone would predict — a recurring team characteristic,
  not something 2025 erased.
"""
    )

# ---------------------------------------------------------------------------
# All-Time Great? tab (Phases 13-16)
# ---------------------------------------------------------------------------
with tab_alltime:
    st.caption(
        "Every number on this tab is measured against all 861 team-seasons since 1999 — "
        "so a rank here means 'out of every team-season in the modern record', not "
        "'out of 2025'."
    )

    hm = historical["metrics"]
    si = control["si_claim"]
    streak = defense_deep["eight_week_streak_claim"]
    mc = counterfactual["monte_carlo"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Clock-weighted lead (points)",
        f"+{hm['time_weighted_margin']['sea_2025_value']:.1f}",
        f"Rank {hm['time_weighted_margin']['rank_of_n'][0]} of "
        f"{hm['time_weighted_margin']['rank_of_n'][1]} since 1999",
        delta_color="off",
    )
    k2.metric(
        "Explosive plays allowed",
        f"{100 * hm['def_explosive_rate_allowed']['sea_2025_value']:.1f}%",
        f"Rank {hm['def_explosive_rate_allowed']['rank_of_n'][0]} of "
        f"{hm['def_explosive_rate_allowed']['rank_of_n'][1]} since 1999",
        delta_color="off",
    )
    k3.metric(
        "Best 8-game defensive stretch (EPA/play)",
        f"{streak['garbage_time_filtered']['sea_2025_best_window']:.3f}",
        f"Rank {streak['garbage_time_filtered']['sea_2025_rank_of_n'][0]} of "
        f"{streak['garbage_time_filtered']['sea_2025_rank_of_n'][1]} "
        f"(rank {streak['unfiltered']['sea_2025_rank_of_n'][0]} unfiltered)",
        delta_color="off",
    )
    k4.metric(
        "Replays reaching 14+ wins",
        f"{mc['pct_of_seasons_at_least_14_wins']:.0f}%",
        f"Exactly 14 wins: {mc['win_distribution']['14']:.0f}%",
        delta_color="off",
    )

    st.caption(
        "The 8-game stretch is shown on the garbage-time-filtered basis, which is the "
        "harsher test for a dominant defense (it spends more snaps in garbage time "
        "precisely because it is winning). Without that filter the same stretch ranks "
        f"{streak['unfiltered']['sea_2025_rank_of_n'][0]} of "
        f"{streak['unfiltered']['sea_2025_rank_of_n'][1]} instead of "
        f"{streak['garbage_time_filtered']['sea_2025_rank_of_n'][0]}. Both bases agree it "
        "beat the 2013 Legion of Boom's best stretch; the gap between them is a reminder "
        "that a single filter choice moves this number a long way."
    )

    st.divider()

    st.header("Were they actually all-time great?")
    st.markdown(
        f"""
Two terms, defined once so the rest reads cleanly. An **explosive play** is a pass of 15+
yards or a run of 10+. **Clock-weighted lead** is the lead Seattle held averaged over every
second of the season, so a 10-point lead held for three quarters counts for far more than
the same 10 points scored at the final whistle — it measures control, not just the final
scoreboard.

On the measures that describe **how games went**, yes, emphatically. Seattle gave up
explosive plays at a rate bettered by only
**{hm["def_explosive_rate_allowed"]["rank_of_n"][0] - 1} team-seasons since 1999**, and held
a clock-weighted lead bettered by only
**{hm["time_weighted_margin"]["rank_of_n"][0] - 1}**.

On the measures that describe **taking the ball away**, no. The full-season turnover margin
was **{hm["turnover_margin"]["sea_2025_value"]:+.0f}** — the
{hm["turnover_margin"]["raw_percentile"]:.0f}th percentile, which is to say roughly average.
The "they fixed the turnovers" story is a story about the end of the season, not the whole
of it. Forcing three-and-outs was similarly ordinary
({hm["three_and_out_rate_forced"]["raw_percentile"]:.0f}th percentile).

This team won by never letting anything big happen, not by generating chaos.
"""
    )

    st.subheader("Pick a metric, see where 2025 lands")
    st.caption(
        "Pick a metric to see the full 1999–2025 distribution. Grey is every other "
        "team-season; green is SEA 2025."
    )

    metric_labels = {v["label"]: k for k, v in hm.items()}
    metric_choices = sorted(metric_labels)
    headline_label = hm["def_explosive_rate_allowed"]["label"]
    picked_label = st.selectbox(
        "Metric",
        metric_choices,
        index=metric_choices.index(headline_label),
        key="alltime_metric",
    )
    picked = metric_labels[picked_label]
    meta = hm[picked]

    advanced = load_processed("team_season_advanced.csv")
    reg_all = advanced[advanced["season_type"] == "REG"]

    hist = px.histogram(reg_all, x=picked, nbins=45, opacity=0.85)
    hist.update_traces(marker_color=GREY, hovertemplate="%{x}<br>%{y} team-seasons<extra></extra>")
    hist.add_vline(
        x=meta["sea_2025_value"],
        line_color=GREEN,
        line_width=3,
        annotation_text="SEA 2025",
        annotation_font_color=GREEN,
    )
    hist.update_layout(
        title=f"{picked_label} — SEA 2025 ranks {meta['rank_of_n'][0]} of {meta['rank_of_n'][1]}",
        xaxis_title=picked_label,
        yaxis_title="Team-seasons",
        bargap=0.02,
    )
    st.plotly_chart(style_fig(hist, 400), width="stretch")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Raw percentile", f"{meta['raw_percentile']:.0f}")
    mc2.metric("Era-adjusted percentile", f"{meta['era_adjusted_percentile']:.0f}")
    mc3.metric(
        "All-time best",
        f"{meta['all_time_best']['value']:.3g}",
        f"{meta['all_time_best']['season']} {meta['all_time_best']['team']}",
        delta_color="off",
    )
    st.caption(
        "Raw compares values directly; era-adjusted compares each season against its own "
        "league. Where they disagree, the era-adjusted number is the honest one."
    )

    st.subheader("Game by game")
    st.caption(
        "Win probability through each of Seattle's 20 games — the nflverse model, "
        "not a betting line."
    )

    curve = load_processed("focus_wp_curve.csv")
    per_game = pd.DataFrame(control["per_game"])
    game_meta = curve.drop_duplicates("game_id")[
        ["game_id", "season_type", "week", "opponent", "at_home"]
    ]
    game_meta = game_meta.merge(
        per_game[["game_id", "final_margin", "time_weighted_margin", "pct_game_time_leading"]],
        on="game_id",
        how="left",
    ).sort_values(["season_type", "week"], ascending=[False, True])

    # Postseason weeks continue the regular-season numbering (19 = wild card),
    # which reads as a meaningless "Wk 21" unless it's mapped back to the round.
    PLAYOFF_ROUNDS = {19: "Wild Card", 20: "Divisional", 21: "NFC Championship", 22: "Super Bowl LX"}

    def game_label(r) -> str:
        result = "W" if r.final_margin > 0 else "L"
        stage = (
            f"Week {int(r.week)}"
            if r.season_type == "REG"
            else PLAYOFF_ROUNDS.get(int(r.week), f"Playoffs Wk {int(r.week)}")
        )
        # The Super Bowl is played at a neutral site, but the schedule still
        # designates one team "home" -- so vs/at would read as a lie there.
        venue = "" if int(r.week) == 22 else ("vs " if r.at_home else "at ")
        return f"{stage} — {venue}{r.opponent} ({result} {abs(int(r.final_margin))})"

    labels = {game_label(r): r.game_id for r in game_meta.itertuples()}
    picked_game = st.selectbox("Game", list(labels), key="alltime_game")
    gid = labels[picked_game]
    g = curve[curve["game_id"] == gid].sort_values("seconds_elapsed")
    grow = game_meta[game_meta["game_id"] == gid].iloc[0]

    wp_fig = px.area(g, x="seconds_elapsed", y="wp_sea")
    wp_fig.update_traces(
        line_color=GREEN,
        fillcolor="rgba(105,190,40,0.22)",
        hovertemplate="Win probability %{y:.0%}<extra></extra>",
    )
    wp_fig.add_hline(y=0.5, line_color=GREY, line_width=1, line_dash="dot")
    for q in (900, 1800, 2700):
        wp_fig.add_vline(x=q, line_color=GRID, line_width=1)
    wp_fig.update_layout(
        title=f"Seattle win probability — {picked_game}",
        xaxis={
            "title": "",
            "tickmode": "array",
            "tickvals": [0, 900, 1800, 2700, 3600],
            "ticktext": ["Kickoff", "End Q1", "Half", "End Q3", "Final"],
        },
        yaxis={"title": "Win probability", "tickformat": ".0%", "range": [0, 1]},
        hovermode="x unified",
    )
    st.plotly_chart(style_fig(wp_fig, 330), width="stretch")

    g1, g2, g3 = st.columns(3)
    g1.metric("Final margin", f"{int(grow['final_margin']):+d}")
    g2.metric("Clock-weighted margin", f"{grow['time_weighted_margin']:+.1f}")
    g3.metric("Share of clock leading", f"{grow['pct_game_time_leading']:.0f}%")

    st.subheader("Champion comparison")
    st.caption(
        "The Sports Illustrated case rested on average end-of-game margin. Weighting that "
        "margin by how long it was held re-orders the list — and moves Seattle to the top."
    )

    combined = (
        advanced.groupby(["season", "team"])
        .apply(
            lambda d: pd.Series(
                {
                    "avg_final_margin": (d["points_for"].sum() - d["points_against"].sum())
                    / d["games"].sum(),
                    "time_weighted_margin": (d["time_weighted_margin"] * d["games"]).sum()
                    / d["games"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    champs = pd.DataFrame(historical["champions"])
    champ_rows = combined.merge(champs, on=["season", "team"], how="inner")
    champ_rows = champ_rows[champ_rows["season"] >= 2000]
    champ_rows["label"] = champ_rows["season"].astype(str) + " " + champ_rows["team"]

    default_picks = [
        lab
        for lab in champ_rows.sort_values("avg_final_margin", ascending=False)["label"].head(8)
    ]
    picked_champs = st.multiselect(
        "Champions to compare (SEA 2025 is always shown)",
        sorted(champ_rows["label"]),
        default=default_picks,
        key="alltime_champs",
    )
    shown = champ_rows[champ_rows["label"].isin(set(picked_champs) | {"2025 SEA"})]

    basis = st.radio(
        "Compare on",
        ["Clock-weighted margin", "Average final margin"],
        horizontal=True,
        key="alltime_basis",
    )
    col = "time_weighted_margin" if basis.startswith("Clock") else "avg_final_margin"
    shown = shown.sort_values(col)

    champ_fig = px.bar(shown, x=col, y="label", orientation="h")
    champ_fig.update_traces(
        marker_color=[
            GREEN if lab == "2025 SEA" else AMBER if lab == "2013 SEA" else GREY
            for lab in shown["label"]
        ],
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    )
    champ_fig.update_layout(
        title=f"{basis}, regular season + playoffs",
        xaxis_title=f"{basis} (points)",
        yaxis_title="",
    )
    st.plotly_chart(style_fig(champ_fig, max(320, 34 * len(shown))), width="stretch")

    st.markdown(
        f"""
On the published statistic, Seattle ranks **{si["sea_2025_rank"][0]} of
{si["n_champions"]}** champions since 2000 — not the 2nd that was reported, because
**{" and ".join(f"{c['season']} {c['team']}" for c in si["champions_ahead"])}** also finished ahead.
On the clock-weighted version it ranks
**{si["stronger_measure"]["sea_2025_rank_among_champions"][0]} of
{si["stronger_measure"]["sea_2025_rank_among_champions"][1]}** — first.
"""
    )

    st.subheader("How wide was the range?")
    st.caption(
        "All three losses were one-score games, nine points combined. Replaying the "
        "same schedule with the same team quality shows how much a 17-game record can swing."
    )

    st.image(str(OUTPUTS / "counterfactual_record_distribution.png"), width="stretch")
    st.image(str(OUTPUTS / "counterfactual_close_games.png"), width="stretch")

    st.markdown(
        f"""
14 wins is the single most likely outcome — but only at
**{mc["win_distribution"]["14"]:.0f}%** of replays. The mean is
**{mc["mean_wins"]:.1f}**, the 90% range runs **{mc["p05_wins"]:.0f} to
{mc["p95_wins"]:.0f} wins**, and **{mc["pct_of_seasons_at_most_11_wins"]:.0f}%** of
seasons finish at 11 or fewer.

So the record was not a fluke: 14–3 is exactly what this team's quality predicts, and the
simulation is *built from* their +191 differential rather than doubting it. The point is
the width. The same team, playing the same seventeen opponents, lands on 12–5 about as
often as on 15–2. A season is a small sample, and three losses by nine total points is
what the favorable side of that noise looks like — not a different, worse team.
"""
    )

# ---------------------------------------------------------------------------
# Players tab (Phase 17)
# ---------------------------------------------------------------------------
with tab_players:
    jsn = players["smith_njigba"]
    jsn_rank = jsn["rankings"]
    barrier = jsn["volume_barrier_claim"]
    darnold = players["darnold"]
    backfield = players["backfield"]
    leap = players["year_two_leap"]
    n_receiver_seasons = players["methodology"]["receiver_reference_set"]

    st.caption(
        f"Ranked against {n_receiver_seasons:,} receiver-seasons and every quarterback "
        "season since 1999."
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric(
        "Receiving yards per team pass attempt",
        f"{jsn_rank['yards_per_team_pass_attempt']['value']:.2f}",
        f"1st of {n_receiver_seasons:,} since 1999",
        delta_color="off",
    )
    p2.metric(
        "Receiving yards",
        f"{jsn['season']['receiving_yards']:,}",
        f"Rank {jsn_rank['receiving_yards']['rank_of_n'][0]} of {n_receiver_seasons:,}",
        delta_color="off",
    )
    p3.metric(
        "Share of team receiving yards",
        f"{100 * jsn_rank['team_receiving_yards_share']['value']:.1f}%",
        f"Rank {jsn_rank['team_receiving_yards_share']['rank_of_n'][0]} of {n_receiver_seasons:,}",
        delta_color="off",
    )
    p4.metric(
        "Darnold's playoff turnovers",
        "0",
        f"In {darnold['clean_postseason']['playoff_dropbacks']} dropbacks",
        delta_color="off",
    )

    st.divider()

    st.header("Jaxon Smith-Njigba did more with less than anyone on record")
    st.markdown(
        f"""
Remember the setup. Seattle released Tyler Lockett and traded DK Metcalf in the same week
of March 2025, which left a 22-year-old who had never been a No. 1 receiver as the only
real threat on the roster. Defenses knew it. He went out and led the NFL in receiving
yards anyway, won Offensive Player of the Year, and was a unanimous first-team All-Pro.
The franchise record he broke, by 490 yards, belonged to Metcalf.

He is only the second Seahawk ever to lead the league in receiving, after Steve Largent,
and the second to win the award, after Shaun Alexander in 2005.

The number that separates him from everyone else is not the yardage. It is the yardage set
against how rarely his offense threw. Measured as receiving yards per team pass attempt,
his {jsn_rank["yards_per_team_pass_attempt"]["value"]:.2f} is **first among all
{n_receiver_seasons:,} receiver-seasons since 1999** — ahead of Steve Smith's 2005, and
ahead of every thousand-yard season by anyone else in the window.

**One caveat worth putting right here rather than burying it.** That measure divides by
team pass attempts, and Seattle's pass attempts were low partly *because* Smith-Njigba was
so productive — the offense led, so it ran the ball. Cause and effect run both directions,
which flatters the stat. On measures that don't have that problem he is 6th on target
share and 53rd on yards per target. First on this one, excellent but not unprecedented on
the others. All three are in the table below.
"""
    )

    st.image(str(OUTPUTS / "players_jsn_concentration.png"), width="stretch")

    st.subheader("Where he ranks, measure by measure")
    jsn_table = pd.DataFrame(
        [
            {
                "Measure": v["label"],
                "Value": round(v["value"], 3),
                "Rank": f"{v['rank_of_n'][0]} of {v['rank_of_n'][1]:,}",
                "Window": f"{v['reference_window'][0]}–{v['reference_window'][1]}",
            }
            for v in jsn_rank.values()
        ]
    )
    st.dataframe(jsn_table, width=NARROW_TABLE, hide_index=True)
    st.caption(
        "Yards over expected is built on air yards and expected yards after catch, which "
        "nflverse publishes from 2006 — hence the shorter window on that row only."
    )

    st.subheader("Nobody has ever done this much with this little")
    st.markdown(
        f"""
The published version of this story is that every receiver with a bigger season had far
more volume to work with. It holds up
exactly: **{barrier["computed"]["n_receiver_seasons_above"]} receiver-seasons since 1999
gained more than {jsn["season"]["receiving_yards"]:,} yards, and the fewest team pass
attempts any of them had was
{barrier["computed"]["min_team_pass_attempts_above"]}.** Seattle threw
{barrier["computed"]["sea_2025_team_pass_attempts"]} times.
"""
    )
    st.image(str(OUTPUTS / "players_jsn_volume_barrier.png"), width="stretch")

    st.header("Sam Darnold's worst season, and then none at all")
    st.markdown(
        f"""
Darnold led the NFL in turnovers in 2025 with {darnold["career"][-2]["turnovers"]}. Then he
played three playoff games and committed none.

The usual telling is that he settled down as the year went on. The career record says
something better than that. **2025 was the highest turnover rate of his career** —
{100 * darnold["career_arc"]["worst_rate_per_dropback"]:.2f} per 100 dropbacks, nominally
above his rookie year with the Jets and above any of the seasons that got him labeled a
bust.

Treat "highest ever" loosely: the gap over his 2018 rookie season is 0.17 per 100 on about
500 dropbacks, which is a coin-flip's worth of difference (95% interval on the gap runs
from −2.2 to +2.6). The defensible claim is not that 2025 was uniquely bad, but that it was
**no better than his worst years** — which is still the opposite of a player who gradually
cleaned things up. He was as careless as he had ever been, and then he stopped entirely for
a month.
"""
    )
    st.image(str(OUTPUTS / "players_darnold_career.png"), width="stretch")

    d1, d2, d3 = st.columns(3)
    d1.metric(
        "Playoff dropbacks without a turnover",
        f"{darnold['clean_postseason']['playoff_dropbacks']}",
    )
    d2.metric(
        "Expected turnovers at his own 2025 rate",
        f"{darnold['clean_postseason']['expected_turnovers']:.1f}",
    )
    d3.metric(
        "P(zero turnovers)",
        f"{darnold['clean_postseason']['p_zero_poisson']:.1%}",
        f"~1 in {darnold['clean_postseason']['odds_against_1_in']:.0f}",
        delta_color="off",
    )
    st.caption(
        darnold["clean_postseason"]["caveat"]
        + " This figure counts the 99 dropbacks he actually took in the playoffs; the "
        "Story tab's 1-in-34 instead projects his season rate across three more games. "
        "Both are reasonable, which is why the honest answer is a range rather than a "
        "single number."
    )

    st.subheader("The career, season by season")
    raw_career = pd.DataFrame(darnold["career"])
    career_df = pd.DataFrame(
        {
            "Season": raw_career["season"].astype(str) + " " + raw_career["season_type"],
            "Team": raw_career["team"],
            "Games": raw_career["games"],
            "Dropbacks": raw_career["dropbacks"],
            "INT": raw_career["interceptions"],
            "Fumbles lost": raw_career["fumbles_lost"],
            "Turnovers": raw_career["turnovers"],
            "Per 100 dropbacks": (100 * raw_career["turnover_rate_per_dropback"]).round(2),
        }
    )
    st.dataframe(career_df, width=NARROW_TABLE, hide_index=True)

    st.markdown(
        """
He played the Super Bowl through an oblique strain. And on the night in 2019 that gave him
a nickname he carried for six years, he went 11 of 32 for 86 yards with four interceptions.
The opponent was New England. The team he beat in Super Bowl LX was also New England.
"""
    )

    st.header("Two backs, one backfield, and a touchdown gap")
    st.markdown(
        f"""
Zach Charbonnet scored **{backfield["backs"][1]["rushing_tds"]} rushing touchdowns** —
more than twice Kenneth Walker III's {backfield["backs"][0]["rushing_tds"]} — on
{backfield["backs"][0]["carries"] - backfield["backs"][1]["carries"]} fewer carries and
almost a yard less per attempt. That total ranks
{backfield["touchdown_rank_of_n"][0]} of {backfield["touchdown_rank_of_n"][1]:,}
rusher-seasons since 1999.

The explanation is visible the moment goal-line carries are counted on their own.
Charbonnet took **{backfield["backs"][1]["goal_line_carry_share"]:.0%} of Seattle's carries
inside the five**, to Walker's {backfield["backs"][0]["goal_line_carry_share"]:.0%}, and
turned {backfield["backs"][1]["tds_inside_5"]} of those
{backfield["backs"][1]["carries_inside_5"]} into touchdowns. Walker got the yards and the
Super Bowl MVP; Charbonnet got the ball on the two-yard line.
"""
    )
    st.image(str(OUTPUTS / "players_backfield_split.png"), width="stretch")

    st.header("Byron Murphy's second year")
    st.markdown(
        f"""
Murphy recorded **{leap["prior_sacks"]:g} sacks as a rookie in {leap["prior_season"]}**. In
{leap["season"]} he recorded **{leap["sacks"]:g}**, tied for the team lead on the NFL's
best scoring defense.

A raw jump flatters anyone who simply played more, so it needs something to be measured
against: every pair of consecutive seasons by the same player for the same team since 1999,
a **+{leap["jump"]:g} sack increase ranks {leap["rank_of_n"][0]} of
{leap["rank_of_n"][1]:,}** — inside the top {100 - leap["percentile"]:.1f}% of
year-over-year jumps by any defender.
"""
    )
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{leap['prior_season']} sacks", f"{leap['prior_sacks']:g}")
    m2.metric(f"{leap['season']} sacks", f"{leap['sacks']:g}", f"+{leap['jump']:g}")
    m3.metric(
        "Rank among year-over-year jumps",
        f"{leap['rank_of_n'][0]}",
        f"of {leap['rank_of_n'][1]:,} since 1999",
        delta_color="off",
    )

# ---------------------------------------------------------------------------
# Coach & Scheme tab (Phase 18)
# ---------------------------------------------------------------------------
with tab_coach:
    dist = scheme["pass_rush_distribution"]
    roster = scheme["sea_2025_pass_rush"]
    wall = scheme["rushing_wall"]
    yoy = scheme["year_over_year"]
    disruption = scheme["ball_disruption"]
    run_game = scheme["run_game"]
    n_team_seasons = scheme["methodology"]["team_seasons"]

    st.caption(
        f"The 2025 defense against all {n_team_seasons} team-seasons since 1999."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Points allowed per game",
        f"{yoy['points_against_per_game']['2025']['value']:.1f}",
        f"1st in 2025; {yoy['points_against_per_game']['2025']['rank_all_time'][0]} "
        f"of {n_team_seasons} since 1999",
        delta_color="off",
    )
    c2.metric(
        "Opponent yards per carry",
        f"{yoy['def_yards_per_carry_allowed']['2025']['value']:.2f}",
        f"1st in 2025, up from "
        f"{ordinal(yoy['def_yards_per_carry_allowed']['2024']['rank_in_season'])} in 2024",
        delta_color="off",
    )
    c3.metric(
        "Players with a sack",
        f"{dist['distinct_sackers']['value']:.0f}",
        f"Rank {dist['distinct_sackers']['rank_of_n'][0]} of {n_team_seasons} since 1999",
        delta_color="off",
    )
    c4.metric(
        "Share held by the top sacker",
        f"{100 * dist['top_sacker_share']['value']:.1f}%",
        f"League average {100 * dist['top_sacker_share']['league_mean_2025']:.1f}%",
        delta_color="off",
    )

    st.divider()

    st.header("The best defense in football had no star pass rusher")
    st.markdown(
        f"""
Seattle allowed the fewest points in the NFL in 2025, the franchise's first scoring title
since the Legion of Boom era. It did it without a single player reaching eight sacks.

{len(roster)} different Seahawks recorded a sack. The leading total was
{dist["best_individual"]["value"]:.0f}, shared three ways, and that total ranks
**{dist["best_individual"]["rank_of_n"][0]:,} of
{dist["best_individual"]["rank_of_n"][1]:,}** individual seasons since 1999 — which is the
point rather than a criticism. The team's leading sacker accounted for just
**{100 * dist["top_sacker_share"]["value"]:.1f}%** of its sacks, against a league average of
{100 * dist["top_sacker_share"]["league_mean_2025"]:.1f}%. That is the
{ordinal(dist["top_sacker_share"]["rank_of_n"][0])} most evenly distributed pass rush of
{n_team_seasons} team-seasons since 1999.
"""
    )

    st.image(str(OUTPUTS / "scheme_sack_concentration.png"), width="stretch")
    st.image(str(OUTPUTS / "scheme_sack_roster.png"), width="stretch")

    st.markdown(
        f"""
The run defense was the other half of it. Seattle allowed
{yoy["def_yards_per_carry_allowed"]["2025"]["value"]:.2f} yards per carry, the best in the
league, and went **{wall["games_without_allowing_a_100_yard_rusher"]} consecutive games
without allowing an individual 100-yard rusher** — counted through the Super Bowl, which is
why it runs longer than the in-season figure of 26 that was quoted at the time.
"""
    )

    st.header("Year one was the same defense, unfinished")
    st.markdown(
        """
Mike Macdonald had the same job and most of the same players in 2024, when Seattle went
10–7 and missed the playoffs. The unit did not change hands. It changed rank.
"""
    )
    yoy_df = pd.DataFrame(
        [
            {
                "Measure": v["label"],
                "2024": round(v["2024"]["value"], 3),
                "Rank in 2024": v["2024"]["rank_in_season"],
                "2025": round(v["2025"]["value"], 3),
                "Rank in 2025": v["2025"]["rank_in_season"],
                "All-time rank (2025)": f"{v['2025']['rank_all_time'][0]} of "
                f"{v['2025']['rank_all_time'][1]}",
            }
            for v in yoy.values()
        ]
    )
    st.dataframe(yoy_df, width="stretch", hide_index=True)
    st.caption(
        "Rank in season is among the 32 teams that year; all-time rank is among all "
        f"{n_team_seasons} team-seasons since 1999."
    )

    st.header("The disruption came from everywhere, not from one star")
    st.markdown(
        f"""
The pass rush was not the only thing spread across the roster. Seattle broke up
**{disruption["team_passes_defensed"]} passes** and added
**{disruption["team_interceptions"]} interceptions** on top of that, with
**{disruption["distinct_defenders_with_a_breakup"]} different players** recording at least
one breakup — the same shape as the sack chart, in coverage rather than on the rush.

The leader in interceptions was not a cornerback or a safety at all. Ernest Jones IV, an
off-ball linebacker, took five of them and returned one 85 yards for a touchdown.
"""
    )
    disruption_df = pd.DataFrame(disruption["leaders"]).rename(
        columns={
            "player": "Player",
            "passes_defensed": "Passes defensed",
            "interceptions": "Interceptions",
            "sacks": "Sacks",
        }
    )
    st.dataframe(disruption_df, width=NARROW_TABLE, hide_index=True)
    st.caption(
        "A pass defensed is a defender physically breaking up a throw. It measures ball "
        "disruption rather than how often a player was thrown at."
    )

    st.header("They ran left most, and ran right best")
    st.markdown(
        f"""
Seattle ran the ball more than almost anyone, and it ran it best to the right:
**{run_game["by_direction"]["right"]["yards_per_carry"]:.2f} yards per carry**, fifth in the
NFL, against {run_game["by_direction"]["left"]["yards_per_carry"]:.2f} to the left on more
than a third more carries. The obvious objection is "so why keep running left?" — and the
honest answer is that this table can't settle it. Run direction isn't chosen at random:
short-yardage and goal-line carries behind the rookie left guard are exactly the runs that
produce low yards per carry *by design*, and defenses tilt toward the side they expect.
A clean answer would need play-level down, distance and box-count controls, which this
split doesn't have. Read it as a description of what happened, not as a coaching error.

On the other side of it, the run defense was the best in football by some distance.
Opponents managed {run_game["metrics"]["def_rush_epa_allowed"]["value"]:.3f} expected points
per rush against Seattle — **1st in the league**, and
{run_game["metrics"]["def_rush_epa_allowed"]["rank_of_n"][0]} of
{run_game["metrics"]["def_rush_epa_allowed"]["rank_of_n"][1]} since 1999.
"""
    )
    direction_df = pd.DataFrame(
        [
            {
                "Running": loc.capitalize(),
                "Share of Seattle's carries": f"{v['share_of_carries']:.1%}",
                "Yards per carry": v["yards_per_carry"],
                "Rank in 2025": v["rank_in_2025"],
            }
            for loc, v in run_game["by_direction"].items()
        ]
    )
    st.dataframe(direction_df, width=NARROW_TABLE, hide_index=True)

    st.header("The coach who called it himself")
    st.markdown(
        """
Macdonald is the first head coach in NFL history to win a Super Bowl while calling his own
defensive plays. Only two head coaches in the league even did it in 2025. He was the
youngest head coach in the NFL when Seattle hired him at 36, and at 38 he is the
third-youngest ever to win the title.

He arrived with a track record. His 2023 Baltimore defense was the first in NFL history to
lead the league in scoring defense, sacks and takeaways in the same season — a combination
no team had managed in the previous hundred-odd years of the sport.

He finished third in Coach of the Year voting behind Mike Vrabel. Five weeks later he beat
Vrabel's Patriots 29–13 in Super Bowl LX, holding them to 78 yards and five first downs
through three quarters.
"""
    )
    st.caption(
        "Records, awards and voting are cited from published reporting; everything else on "
        "this page is computed from the play-by-play. Sources are in REFERENCES.md."
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
`nflreadpy` (CC-BY-4.0). Coverage widens with the question being asked:
2024–2025 for the season-over-season comparison, 2010–2025 for the defense
anomaly baseline, and **1999–2025 — 861 team-seasons — for the
all-time comparisons**, which is as far back as nflverse
publishes EPA and win probability. Regular-season games only, unless noted.

Two things in the published coverage of this team **cannot** be reproduced
here and are cited as external context rather than recomputed: **DVOA**,
which is proprietary, and **blitz rate**, because the play-by-play carries
no participation or pass-rusher data. For the same reason, "pressure" in
this project is a `sack OR qb_hit` proxy and reads lower than a charted
pressure rate.
"""
    )

    st.header("Pythagorean win expectation")
    st.markdown("`PF^2.37 / (PF^2.37 + PA^2.37)` — the standard NFL exponent.")
    PYTH_ROWS = {
        "points_for": "Points for",
        "points_against": "Points against",
        "games": "Games",
        "actual_wins": "Actual wins",
        "actual_losses": "Actual losses",
        "pythagorean_win_pct": "Pythagorean win %",
        "pythagorean_expected_wins": "Expected wins",
        "wins_over_expectation": "Wins over expectation",
    }
    pyth_df = pd.DataFrame(pyth).loc[list(PYTH_ROWS)].rename(index=PYTH_ROWS)
    pyth_df.index.name = ""
    # Two data columns stretched to 1280px put each number a screen away from
    # its own label.
    st.dataframe(pyth_df, width=NARROW_TABLE)

    st.header("Personnel and scheme deltas")
    # 13 columns across 2 seasons overflows the container and truncates; one
    # column per season keeps every field visible.
    PERSONNEL_ROWS = {
        "qb": "Quarterback",
        "oc": "Offensive coordinator",
        "completions": "Completions",
        "attempts": "Attempts",
        "completion_pct": "Completion %",
        "passing_yards": "Passing yards",
        "interceptions": "Interceptions",
        "int_rate_pct": "Interception rate %",
        "fumbles_lost": "Fumbles lost",
        "sacks_suffered": "Sacks allowed",
        "dropbacks": "Dropbacks",
        "pressured_dropbacks": "Pressured dropbacks",
        "pressure_rate_pct": "Pressure rate allowed %",
    }
    personnel_df = (
        pd.DataFrame(personnel).loc[list(PERSONNEL_ROWS)].rename(index=PERSONNEL_ROWS)
    )
    personnel_df.index.name = ""
    st.dataframe(personnel_df, width=NARROW_TABLE)

    st.header("Engineered features")
    st.markdown(
        """
Team-week features from play-by-play: offensive/defensive EPA per play
(core scrimmage plays only, 0.05 ≤ win probability ≤ 0.95, garbage
time excluded), turnover margin, red-zone TD rate, and pressure rate.
"""
    )
    FEATURE_LABELS = {
        "off_epa_per_play": "Offensive EPA/play",
        "def_epa_per_play_allowed": "Defensive EPA/play allowed",
        "turnover_margin_per_game": "Turnover margin per game",
        "red_zone_td_pct": "Red zone TD %",
        "pressure_rate_allowed": "Pressure rate allowed %",
        "pressure_rate_created": "Pressure rate created %",
    }
    team_season = load_team_season_features()
    sea_season_avg = (
        team_season[team_season["team"] == "SEA"]
        .set_index("season")[list(FEATURE_LABELS)]
        .rename(columns=FEATURE_LABELS)
        .round(2)
        .T
    )
    sea_season_avg.index.name = ""
    st.dataframe(sea_season_avg, width=NARROW_TABLE)

    st.header("Defense anomaly detection")
    meth = defense["methodology"]
    st.markdown(
        f"""
Built a team-season defensive dataset for all **{meth["n_team_seasons"]}**
team-seasons from {meth["seasons"][0]}–{meth["seasons"][1]} (EPA/play
allowed, sack rate, takeaways/drive, points allowed/game), z-scored **within
each season** before pooling across years, signs oriented so higher = better
defense. Mahalanobis distance and an Isolation Forest were run as
cross-checks on top of the composite z-score.

The cross-checks do **not** reproduce the headline percentile, and that is worth
stating plainly rather than glossing. The composite z-score puts 2025 at the
**91.8th percentile** (rank 43 of 512). Mahalanobis distance puts it at the
**53.7th** (129th of 261 good-direction team-seasons), and the Isolation Forest
at the **65.8th**.

What the three agree on is the *qualitative* claim — elite, not off the charts —
and they disagree on how elite. The gap is informative rather than alarming:
the composite z-score rewards being good on all four metrics at once, while
Mahalanobis explicitly down-weights correlated metrics (EPA allowed and points
allowed move together) and rewards being *unusual* rather than *good*. A team
that is excellent in an ordinary shape will rank high on the first and middling
on the second, which is exactly what happens here. Treat the 91.8th percentile
as the headline and the other two as evidence against over-claiming.
"""
    )
    DEFENSE_ROWS = {
        "def_epa_per_play_allowed": "EPA/play allowed",
        "sack_rate_created": "Sack rate created",
        "takeaways_per_drive": "Takeaways per drive",
        "points_allowed_per_game": "Points allowed per game",
    }
    st.subheader("Inputs and scores, SEA 2024 vs. 2025")
    anomaly_df = pd.DataFrame(
        {
            f"{season} raw": [defense[f"SEA_{season}"]["raw"][k] for k in DEFENSE_ROWS]
            for season in (2024, 2025)
        }
        | {
            f"{season} z-score": [
                defense[f"SEA_{season}"]["z_scores"][k] for k in DEFENSE_ROWS
            ]
            for season in (2024, 2025)
        },
        index=list(DEFENSE_ROWS.values()),
    ).round(3)[["2024 raw", "2024 z-score", "2025 raw", "2025 z-score"]]
    anomaly_df.index.name = ""
    st.dataframe(anomaly_df, width=NARROW_TABLE)

    SCORE_ROWS = {
        "composite_z": "Composite z-score",
        "composite_z_rank_of_n": "Composite rank (of 512)",
        "composite_z_pctile": "Composite percentile",
        "mahalanobis": "Mahalanobis distance",
        "mahalanobis_rank_among_good_direction": "Mahalanobis rank (good-direction)",
        "iso_forest_anomaly_score": "Isolation Forest score",
        "iso_forest_pctile": "Isolation Forest percentile",
    }
    score_df = pd.DataFrame(
        {
            str(season): [defense[f"SEA_{season}"][k] for k in SCORE_ROWS]
            for season in (2024, 2025)
        },
        index=list(SCORE_ROWS.values()),
    )
    score_df.index.name = ""
    st.dataframe(score_df, width=NARROW_TABLE)

    st.markdown("**Top 10 team-seasons by composite z-score, 2010–2025:**")
    top10_df = pd.DataFrame(defense["top10_by_composite_z"]).rename(
        columns={
            "season": "Season",
            "team": "Team",
            "composite_z": "Composite z-score",
            "mahalanobis": "Mahalanobis distance",
        }
    )
    st.dataframe(top10_df, width=NARROW_TABLE, hide_index=True)

    st.header("Turnover rate model")
    st.markdown(turnover["framing"]["not_extreme_value_theory"])
    st.markdown(
        f"""
Regular-season rate: **{turnover["season_turnovers"]}** turnovers over
{turnover["dropbacks"]} dropbacks ({turnover["turnover_rate_per_dropback"]:.2%}) /
{turnover["drives"]} drives ({turnover["turnover_rate_per_drive"]:.2%}).
Distribution choice: **{turnover["distribution"]["chosen_distribution"]}**
({turnover["distribution"]["reason"]})

Playoff exposure is sized from Darnold's own regular-season pace: an
estimated **{turnover["playoff_exposure_derived"]["dropbacks"]:.0f}** dropbacks
over 3 games. He actually saw **{turnover["playoff_actual"]["dropbacks"]}** —
higher than his own average, which makes the zero-turnover run *more*
surprising, not less.
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
    SENSITIVITY_COLS = {
        "expected_turnovers": "Expected turnovers",
        "p_zero_poisson": "P(zero) Poisson",
        "odds_against_1_in": "Odds against, 1 in",
        "p_zero_negative_binomial": "P(zero) neg. binomial",
        "odds_against_nb_1_in": "Odds against (NB), 1 in",
    }
    SENSITIVITY_ROWS = {
        "actual_playoff_dropbacks": "Actual playoff dropbacks",
        "actual_playoff_drives": "Actual playoff drives",
        "first_half_rate_regime": "First-half rate regime",
        "second_half_rate_regime": "Second-half rate regime",
    }
    sensitivity_df = (
        pd.DataFrame(turnover["sensitivity"])
        .T.rename(index=SENSITIVITY_ROWS, columns=SENSITIVITY_COLS)
        .round(3)
    )
    sensitivity_df.index.name = ""
    st.dataframe(sensitivity_df, width="stretch")

    st.header("Regression decomposition")
    ols = decomp["ols_point_diff_per_game"]
    st.markdown(
        f"""
OLS of point differential/game on the six standardized team features, across
32 teams × 2 seasons (n={ols["n"]}). **R² = {ols["r2"]:.3f}**
(adjusted {ols["r2_adj"]:.3f}). Linear regression was used deliberately
instead of a gradient-boosted model with SHAP: at n={ols["n"]}, a tree
ensemble would overfit, and a linear model's coefficients already *are* the
decomposition — no separate explainability layer is needed on top of them.
"""
    )
    coef_df = (
        pd.DataFrame(
            {
                "Coefficient": ols["coefficients"],
                "Std. error": ols["std_errors"],
                "p-value": ols["p_values"],
            }
        )
        .drop("intercept")
        .rename(index=FEATURE_LABELS)
    )
    coef_df.index.name = ""
    st.dataframe(coef_df, width=NARROW_TABLE)

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
    PER_FEATURE_COLS = {
        "raw_2024": "2024 raw",
        "raw_2025": "2025 raw",
        "z_2024": "2024 z",
        "z_2025": "2025 z",
        "delta_z": "Δ z",
        "coefficient": "Coefficient",
        "contribution_to_point_diff_change": "Points contributed",
    }
    per_feature_df = (
        pd.DataFrame(sea_decomp["per_feature"])
        .T.rename(index=FEATURE_LABELS, columns=PER_FEATURE_COLS)
        .round(3)
    )
    per_feature_df.index.name = ""
    st.dataframe(per_feature_df, width="stretch")

    st.header("Historical baseline")
    st.markdown(
        f"""
`src/season_metrics.py` builds one row per team-season for **1999–2025** and commits it
as `data/processed/team_season_advanced.csv`, so nothing downstream needs the raw
play-by-play. EPA and success rate reuse the same core-play filter used throughout (downs 1–4,
pass/run, win probability 5–95%) so every number on the site stays comparable.

Every metric is reported at **two percentiles**: raw, and era-adjusted via a
within-season z-score. The NFL's scoring environment moved substantially across this
window, so a raw percentile quietly flatters modern offenses. Where the two disagree,
the headline number is deliberately the *less* flattering of the pair.

Ranks count ties as half and run in the metric's own good direction (1 = best).
Reference set: **{historical["methodology"]["n_team_seasons_reg"]} regular seasons,
{historical["methodology"]["n_champions"]} champions.**
"""
    )
    claims_df = pd.DataFrame(
        [
            {
                "Claim": c["claim"],
                "Source": c["source"],
                "Computed": c["computed"],
                "Verdict": c["verdict"],
            }
            for c in historical["published_claims"]
        ]
    )
    st.dataframe(claims_df, width="stretch", hide_index=True)

    st.header("Game control")
    st.markdown(
        f"""
Clock-weighted margin is `sum(margin_after_play × seconds_until_next_play) / total_seconds`.
It separates a team that led wire-to-wire from one that won late by the same score —
something average final margin cannot do. Overtime carries zero weight, because nflverse
reports zero seconds remaining throughout OT, making this a *regulation*-clock measure.

Testing the published claim on its own terms: SEA 2025 averaged
**{si["sea_2025_avg_final_margin"]:+.2f}** points per game including playoffs, which ranks
**{si["sea_2025_rank"][0]} of {si["n_champions"]}** champions since 2000 — the reported
figure was 2nd. On the clock-weighted version it ranks
**{si["stronger_measure"]["sea_2025_rank_among_champions"][0]}**.
"""
    )
    st.image(str(OUTPUTS / "game_control_season_arc.png"), width="stretch")

    st.header("Defense deep dive")
    st.markdown(
        f"""
The published claim — a best-in-25-years eight-week stretch at −0.34 EPA/play — is tested
by computing the same rolling window for **every** team-season since 1999, not just
Seattle's. Two bases are reported, because published streak figures are normally computed
*without* a garbage-time filter while the rest of this project applies one:

| Basis | SEA 2025 best 8-game window | Rank | Beats 2013 SEA? |
|---|---|---|---|
| Unfiltered | {streak["unfiltered"]["sea_2025_best_window"]:+.3f} | {streak["unfiltered"]["sea_2025_rank_of_n"][0]} of {streak["unfiltered"]["sea_2025_rank_of_n"][1]} | {streak["unfiltered"]["beats_2013_seahawks"]} |
| Garbage-time filtered | {streak["garbage_time_filtered"]["sea_2025_best_window"]:+.3f} | {streak["garbage_time_filtered"]["sea_2025_rank_of_n"][0]} of {streak["garbage_time_filtered"]["sea_2025_rank_of_n"][1]} | {streak["garbage_time_filtered"]["beats_2013_seahawks"]} |

Neither basis reproduces −0.34 exactly, so the *superlative* isn't confirmed. The
*comparison* the claim makes is: on both bases, this defense's best stretch beat the 2013
Legion of Boom's. Note the filtered number is the harsher test for a dominant defense,
which spends more snaps in garbage time precisely because it is dominant.
"""
    )
    st.image(str(OUTPUTS / "defense_rolling_epa.png"), width="stretch")

    st.header("Counterfactual")
    cf_model = counterfactual["methodology"]["margin_model_fit"]
    st.markdown(
        f"""
A margin model — `expected margin = own point differential/game − opponent's +
home-field advantage` — is fit across all {cf_model["n_games"]} games of 2025, recovering a
home-field edge of **{cf_model["home_field_advantage"]:+.2f}** points and a residual SD of
**{cf_model["residual_sd"]:.2f}**. Seattle's actual 17-game schedule is then replayed
{counterfactual["methodology"]["n_simulations"]:,} times.

This is retrospective, not predictive: the ratings already know how the season went, and
games are treated as independent. Its job is to bound the variance in a 17-game sample.
The residual SD comes in slightly under the ~13 usually quoted for NFL margins, precisely
because the ratings are fit in-sample — so the win range it produces is, if anything,
a little too narrow.
"""
    )
    tl = counterfactual["turnover_luck"]
    st.markdown(
        f"""
**Turnover luck.** Forcing a fumble is a skill; recovering one is close to a coin flip.
Holding recovery share at the league rate
({tl["league_fumble_recovery_rate_by_defense"]:.0%}) and leaving the forcing alone moves
Seattle's turnover margin from **{tl["actual_turnover_margin"]:+d}** to
**{tl["turnover_margin_at_league_recovery_rate"]:+.1f}** — a bounce component of
**{tl["bounce_component"]:+.1f}**.
"""
    )

    st.header("Player deep dives")
    st.markdown(
        f"""
`season_metrics.py` emits one row per player-season for 1999–2025. The receiver
reference set is every regular season with 50+ targets —
**{players["methodology"]["receiver_reference_set"]:,}** of them.

**The denominator decides a target-share number.** nflverse's `pass_attempt` flag also
fires on sacks and two-point conversion passes; left alone it gives SEA 2025 510 attempts
against the league's official 481. Removing both lands on exactly 481, which puts
Smith-Njigba's target share on 33.9% and his share of team receiving yards on 44.1% —
matching the official figures. Three different pass-attempt counts are already
circulating in public coverage of this season, so the one used here is the one that
reconciles to the league's.

**A silent trap, guarded.** nflverse ships two receiver keys. `receiver_player_id` is null
on *incomplete* passes for 2003–2008 — 0.7% populated there against 80%+ for `receiver_id` —
so aggregating targets on it counts only catches for six seasons. This project uses
`receiver_id`, asserts the coverage at build time, and has a test that fails if the
window's catch rate ever drifts.

**Opportunity is a proxy.** Target share is measured over team pass attempts. The stricter
denominator would be routes run, which no public play-by-play carries, so this measure
slightly favors a receiver who never leaves the field. It is the same measure the
published figures use.

**Darnold's turnovers are counted over every snap he touched**, not only dropbacks. The turnover model
established that one of his 20 giveaways in 2025 was an aborted snap coded as a run;
restricting the numerator to dropbacks returns 19 and disagrees with every published total.
The rate keeps dropbacks as its denominator, which makes it slightly conservative.

**Rankings rather than probabilities, for the receiving season.** The turnover model's Poisson works
because turnovers are rare, discrete, near-independent events. Receiving yards are none of
those, and pass volume is jointly determined with the production it would be conditioning
on — Seattle ran because it led, and led partly because Smith-Njigba produced. The surprise
is also definitional: yards per team pass attempt is target share times yards per target,
and he is 1st on the product, 6th on the share, 53rd on yards per target. Positions in a
stated reference set are reported instead.
"""
    )

    st.header("Coach and scheme")
    st.markdown(
        f"""
Sacks are counted with **half-credits**, because `sack_player_id` alone misses roughly a
tenth of sacks — the shared ones — and a rotational pass rush is exactly what that
undercounts. With halves included Seattle's 2025 total reconciles to the published 47.0
exactly.

Concentration is described two ways. Top-sacker share sees only the leader; the Herfindahl
index sees the whole distribution, so a team with two co-leaders and a team with one leader
plus a long tail can share a top-sacker share and separate here. Both rank against all
{scheme["methodology"]["team_seasons"]} team-seasons.

**Passes defensed measure ball disruption**, not coverage volume. They fire on a stable
~30% of incompletions in every season since 1999, which makes them comparable across eras,
but they record a defender getting a hand to the throw rather than how often he was
targeted.

**Run direction** uses `run_location`, populated on ~96% of carries in every season since
1999. It describes where a run game worked. It cannot separate the blocking from the back,
because the play-by-play names no blockers.

**Pressure is ranked from 2006, not 1999**, and it is the only metric in this project on a
short window. nflverse's QB-hit attribution is not stable: 2003–2005 contain zero QB hits
and 1999–2002 about half the modern rate, so the `sack OR qb_hit` proxy degrades into a
bare sack rate in those years. Ranking 2025 against all 861 team-seasons compared it
against seasons where half the metric did not exist.

**Cited, not computed.** Records, awards and voting margins on the Players and
Coach & Scheme tabs come from published reporting and are listed under
`external_context_not_computed` in the relevant
output JSON — the same treatment DVOA and blitz rate get. That covers the OPOY
vote, All-Pro selections, Macdonald's play-calling and coaching records, Baltimore's 2023
league lead, and games started, which no play-by-play records.
"""
    )

    st.header("Limitations")
    st.markdown(
        """
- The team EPA/play features are aggregated with equal weight per game,
  not weighted by play volume within a season; red-zone TD% and pressure
  rates are instead computed from summed season counts (not an average of
  weekly percentages), so uneven weekly sample sizes don't distort the rate.
- The regression decomposition is correlational, built on 64 observations,
  and does not control for strength of schedule, injuries, or other omitted
  context.
- The defense anomaly detection and turnover rate model are both
  single-season, single-team analyses layered on a historical baseline —
  neither is a causal claim, only a statement about where this team-season
  sits relative to a well-defined comparison set.
- The 1999–2025 window is bounded by nflverse's EPA and win-probability
  coverage, not by when the NFL got interesting. "All-time" throughout this
  project means "since 1999," and the 1985 Bears and 1972 Dolphins are
  simply not in the comparison set.
- The Monte Carlo replay assumes games are independent and uses full-season
  ratings, so it measures the variance in a 17-game sample rather than
  forecasting anything.
"""
    )
