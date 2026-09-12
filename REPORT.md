# From missing the playoffs to Super Bowl champions: the 2024→2025 Seattle Seahawks

*A statistical retrospective on the fastest turnaround in recent Seahawks history.*

---

## The story

### 2024: a good team that still went home early

The 2024 Seahawks won 10 games and missed the playoffs anyway — the only
10-win team to miss the playoffs since the NFL went to a 17-game schedule.
They lost the NFC West tiebreaker to the Rams on strength of victory. It wasn't a fluke of
bad luck, either: when you look at how many points Seattle scored versus how
many they allowed all season (375 for, 368 against), a standard model of "how
many games should a team with this scoring margin win" says this team was
worth about **8.7 wins**, not 10. Seattle actually *overachieved* relative to
its underlying performance and still missed the playoffs, purely on a
tiebreaker. The problem wasn't luck — the team simply wasn't playing quite
well enough, game to game, and a razor-thin margin (winning close games) only
gets you so far.

Geno Smith was at quarterback, Ryan Grubb was the offensive coordinator, and
the offensive line struggled to protect him — Smith was pressured on nearly
1 in 6 of his dropbacks (16.8%), one of the higher rates among starting
quarterbacks that year.

### 2025: everything clicked

The 2025 Seahawks won 14 games, the NFC West, and Super Bowl LX. Sam Darnold
took over at quarterback (signed after Smith was traded), Klint Kubiak became
offensive coordinator, and the same defensive coordinator, Aden Durde, who ran
the "Dark Side" defense both years, got dramatically better results out of
essentially the same unit philosophy.

Some of what changed is what you'd expect: pressure on the quarterback dropped
from 16.8% to 12.1%, and sacks allowed nearly halved (50 → 27). But not
everything improved cleanly. Darnold's raw completion percentage was actually
*lower* than Smith's the year before (67.7% vs. 70.4%), and his individual
interception rate was slightly *higher* (2.94% vs. 2.6%). The jump wasn't a
story of every single number getting better — it was concentrated in a
smaller number of things that happened to matter enormously: pass protection,
and the two units' overall efficiency on a per-play basis.

### The defense that carried them — but let's not oversell it

The "Dark Side" defense was, by the numbers, the best defense in the NFL in
2025 — first in the league in both defensive efficiency per play and points
allowed per game. It's tempting to reach for "greatest defense of all time"
here, and initially that's the story this project expected to find. It didn't
hold up under a closer look, and the honest version is more interesting: when
you rank the 2025 Seahawks defense against every team-defense in the NFL going
back to 2010 — over 500 team-seasons — it lands solidly in the **top 10%**,
but not in the top 10 outright. That distinction actually belongs to Seattle's
*own* 2013 "Legion of Boom" defense, which remains the more statistically
extreme unit in franchise history. The 2025 defense won primarily by being
excellent at preventing efficient plays and takeaways; it was merely
average, league-wide, at generating a pass rush (sacks), which is part of why
it falls just short of all-time company. Elite, genuinely — just not
unprecedented.

### The turnover-free playoff run

Sam Darnold led the entire NFL in turnovers during the 2025 regular season —
20 of them (14 interceptions, 6 lost fumbles) across 17 games, the same
player who would go on to win a championship. Then, across three playoff
games (a 41-6 rout of the 49ers, a 31-27 win over the Rams, and a 29-13 Super
Bowl win over the Patriots), he committed **zero** turnovers — the first
Super Bowl champion ever to complete an entire postseason without one.

Given how often he turned the ball over all regular season, how surprising is
that? We built a model of his turnover rate and used it to estimate the odds
of a clean three-game stretch. The answer: **about a 3% chance, or roughly 1
in 34** — genuinely unlikely, not just a good week.

One popular explanation for the turnaround is that Darnold "settled down" and
got safer with the ball as the season wore on. The data doesn't support that
story: his turnover rate was actually *higher* in the second half of the
season than the first. The clean playoff run wasn't the tail end of a
gradual improvement — it was a sudden break from his entire season-long
pattern.

### So what actually explains the jump?

Pulling this together: across all 32 NFL teams over these two seasons, a
team's per-game scoring margin is very well explained by two things — how
efficiently its offense moves the ball, and how efficiently its defense
prevents the other team from moving the ball. Applying that same relationship
to Seattle specifically, those two factors — the improved offense and the
improved defense — account for essentially the *entire* jump in Seattle's
scoring margin from 2024 to 2025. Turnover luck, red-zone execution, and pass
rush played only minor, statistically inconclusive roles by comparison. If
anything, the model suggests Seattle's underlying performance jump was
*slightly bigger* than the win/scoring outcome it produced — the team may
have quietly left a little more improvement on the table than the record
shows.

### The honest caveats

This project tries to show its work rather than round off inconvenient
detail, so a few things are worth stating plainly:

- The 2025 defense is elite, not literally the best ever — a claim the data
  didn't support once actually tested, and we changed the framing rather than
  keep the more dramatic (and wrong) version.
- Not every underlying stat improved between the two years (red-zone
  touchdown rate and pressures created both dipped slightly) — the turnaround
  was concentrated in specific things, not a uniform team-wide leap.
- "1 in 34" is an estimate from a model built on one player's one regular
  season, not a law of nature — a different, entirely reasonable set of
  modeling choices could plausibly land in the 2-4% range rather than exactly
  2.93%. The point is the order of magnitude: this was genuinely improbable,
  not "he just got a little lucky."
- Both regular seasons (2024 and 2025) still show Seattle winning slightly
  more games than its scoring margin alone would predict. That pattern is
  present in *both* years, so it's a recurring team characteristic worth
  noting, not something the 2025 turnaround erased.

---

## Technical details & methodology

*For readers who want to evaluate the statistical rigor behind the story
above. All figures below are pulled directly from this project's output
files (`outputs/*.json`), not re-derived for this writeup.*

### Data and scope

Play-by-play, schedule, and player-stat data come from `nflverse` via the
`nflreadpy` package (CC-BY-4.0), covering 2010–2025 for the historical
baseline used in Phase 5, and 2024–2025 specifically for everything else.
Regular-season games only, unless a section says otherwise.

### Pythagorean win expectation (Phase 2)

NFL Pythagorean win percentage: `PF^2.37 / (PF^2.37 + PA^2.37)`, the
standard exponent used in public NFL analytics (not the base-2 exponent from
baseball). Results:

| Season | PF | PA | Actual W-L | Pythagorean win% | Expected wins | Wins over expectation |
|---|---|---|---|---|---|---|
| 2024 | 375 | 368 | 10-7 | .511 | 8.69 | +1.31 |
| 2025 | 483 | 292 | 14-3 | .767 | 13.04 | +0.96 |

Both seasons show Seattle outperforming its point-differential-implied win
total by roughly one game — a mild, consistent pattern across both years, not
unique to either one.

### Personnel and scheme deltas (Phase 3)

QB-level regular-season stats, pulled via `nflreadpy`'s player-stats source
and cross-checked to an exact match against the known ground truth of
Darnold's 2025 turnover total (14 INT + 6 lost fumbles = 20).

| | 2024 (Geno Smith / Grubb) | 2025 (Sam Darnold / Kubiak) |
|---|---|---|
| Completion % | 70.4% | 67.7% |
| INT rate | 2.6% | 2.94% |
| Sacks suffered | 50 | 27 |
| Pressure rate allowed | 16.8% | 12.1% |

### Engineered features (Phase 4)

Team-week features built from play-by-play: offensive/defensive EPA per play
(core scrimmage plays only — down 1-4, pass/run, garbage time excluded via
0.05 ≤ win probability ≤ 0.95), turnover margin, drive-level red-zone
touchdown rate, and pressure rate (sack or QB hit on a dropback). SEA's
season averages:

| Metric | 2024 | 2025 |
|---|---|---|
| Offensive EPA/play | -0.044 | +0.146 |
| Defensive EPA/play allowed | -0.007 | -0.245 |
| Turnover margin/game | -0.118 | -0.059 |
| Red zone TD% | 55.3% | 52.7% |
| Pressure rate allowed | 17.2% | 12.2% |
| Pressure rate created | 17.6% | 16.1% |

Note red-zone TD% and pressure rate created both *declined* slightly — a
detail the narrative section calls out rather than smoothing over, and one
the Phase 7 decomposition below independently confirms mattered little either
way.

### Defense anomaly detection (Phase 5)

Built a team-season defensive dataset for all 512 team-seasons from
2010–2025 (four metrics: EPA/play allowed, sack rate, takeaways per drive,
points allowed/game), z-scored **within each season** before pooling across
years (so era-level scoring trends don't get mistaken for team quality), with
signs oriented so higher = better defense.

- SEA 2025 composite z-score (mean of the four signed z-scores): **+1.03**,
  ranking **43rd of 512** team-seasons (91.8th percentile) — a genuinely elite
  defense.
- **Mahalanobis distance** (accounting for correlation between the four
  metrics) was used as a second, stricter test of how statistically unusual
  the profile is, not just how good: SEA 2025 ranks 129th of 261
  good-direction team-seasons by this measure — still strong, but this is the
  test that keeps the defense out of the all-time top tier, because its sack
  rate (12th in the league, +0.18 SD) is unremarkable relative to how
  dominant it was on the other three metrics.
- **Isolation Forest** (an independent, non-parametric anomaly detector) was
  run as a cross-check and placed SEA 2025 at the 65.8th percentile for
  anomalousness — consistent with "clearly above average, not statistically
  extreme."
- The 10 most statistically extreme defenses in the dataset by composite
  z-score: 2017 Jacksonville, **2013 Seattle**, 2018 Chicago, 2019
  Pittsburgh, 2013 Carolina, 2019 New England, 2020 Pittsburgh, 2012 Chicago,
  2010 Pittsburgh, 2022 Philadelphia. The 2025 Seahawks defense (rank 43) does
  not crack this list; the franchise's own 2013 defense does.

### Turnover rate model (Phase 6)

Modeled Darnold's regular-season turnovers as a **Poisson rate process** —
the correct tool for counting a small number of discrete events (turnovers)
over a bounded number of trials (dropbacks), as opposed to extreme value
theory, which models the tail of a *continuous magnitude* distribution (e.g.
single-game passing-yardage records) and doesn't apply here. A Poisson-vs-
negative-binomial dispersion test did not favor the extra complexity of
negative binomial (p = 0.10), so Poisson was used, with the negative-binomial
result retained as a conservative upper bound.

- Regular-season rate: 20 turnovers over 516 qb dropbacks (3.88%) / 193
  drives (10.36%).
- Playoff exposure was sized from Darnold's own regular-season per-game
  average (not an arbitrary guess): 3 games × his own rate ≈ 91 dropbacks /
  34 drives. His actual playoff workload (99 dropbacks / 34 drives) came in
  close to that estimate — matching almost exactly on drives, moderately
  higher on dropbacks — which if anything makes the clean streak slightly
  *more* impressive (more snaps to turn the ball over on, not fewer).
- **P(zero turnovers) = 2.93%** (~1 in 34) under the fitted Poisson model;
  3.83% under the more conservative negative-binomial bound.
- Tested, rather than assumed, the "got safer down the stretch" narrative via
  a formal trend test: first-half rate 3.07% vs. second-half rate 4.51%
  (trend and half-split significance tests both p > 0.5) — the point estimate
  runs in the *opposite* direction of the popular narrative, and neither test
  is significant. The playoff zero-turnover streak reads as a discontinuity,
  not the visible endpoint of an in-season trend.

### Regression decomposition (Phase 7)

Fit an OLS regression of point differential per game on the six standardized
Phase 4 features, across all 32 teams × 2 seasons (n = 64), then used the
fitted coefficients to decompose Seattle's actual year-over-year change.
Point differential (not wins) was the target, since Phase 2 already showed
Seattle's win total contains a tiebreaker/luck component beyond what point
differential captures.

- Model fit: **R² = 0.919** (adjusted 0.910). Offensive EPA/play
  (coefficient +4.08, p < .001) and defensive EPA/play allowed (coefficient
  -2.70, p < .001) are the only statistically significant predictors at this
  sample size; turnover margin, red-zone TD%, and pressure rates point the
  expected direction but aren't significant.
- A secondary model using wins instead of point differential as the target
  (R² = 0.799) was fit as a cross-check and tells a consistent story.
- SEA's point differential per game rose from **+0.41 to +11.23** (matching
  Phase 2's points for/against exactly). Decomposed: defensive EPA
  improvement contributes **+6.99** points/game, offensive EPA improvement
  contributes **+6.42**, with the four remaining features contributing small
  and partly offsetting amounts (net roughly -0.6). The model's residual is
  **-1.99** (about 18% of the actual change) — meaning the six engineered
  features alone would have predicted an even *larger* jump than what
  actually occurred, i.e. there's no "unexplained extra improvement" left
  over to account for; if anything the reverse.
- Linear regression was used deliberately instead of a gradient-boosted model
  with SHAP, even though the latter is this project's default tool for
  explainability work: at n = 64, a tree ensemble would overfit, and a linear
  model's coefficients already *are* the decomposition — no separate
  explainability layer is needed on top of them.

### Limitations

- Phase 4-7's engineered features are aggregated with **equal weight per
  game**, not weighted by play volume within a season — a documented
  simplification, not an oversight, made to avoid re-reading raw play-by-play
  a second time after Phase 4 already built the feature file.
- The regression decomposition (Phase 7) is correlational, built on 64
  observations, and does not control for strength of schedule, injuries, or
  other omitted context.
- Phase 5 and 6 are both single-season, single-team analyses layered on top
  of a historical baseline; neither is a claim about causality, only about
  where this specific team-season sits relative to a well-defined comparison
  set.
- All facts about coaches, quarterbacks, playoff results, and season records
  used throughout are the project's confirmed ground truth (see
  `HANDOFF.md`); no narrative claims here are drawn from unverified external
  sources.
