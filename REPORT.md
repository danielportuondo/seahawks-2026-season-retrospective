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

### Were they actually all-time great?

Everything above compares 2025 to 2024. That's the right frame for asking what
changed, and the wrong one for the question the rest of the football world spent
the offseason on: was this one of the great teams? Answering that needs a
denominator, so the analysis was extended to every team-season nflverse covers —
861 regular seasons from 1999 through 2025, and the 27 Super Bowl champions in
that span.

**On how their games went, the case is overwhelming.** Seattle allowed explosive
plays — 15-plus-yard passes, 10-plus-yard runs — at a rate bettered by only
**11 team-seasons in 27 years**. They held a bigger lead for longer than all but
**12**. Among Super Bowl champions since 2000, nobody has controlled games more
completely. This was a team that made the game small and then sat on it.

**On taking the ball away, the case collapses.** Their full-season turnover margin
was **−1**: 25 takeaways against 26 giveaways, a hair below league average and
the 46th percentile historically. They forced three-and-outs at the 49th
percentile. The widely repeated "they fixed their turnovers" line is true about
the *end* of the season — Phase 6 showed that trend — but it is not true about the
season. Seattle won 14 games with a negative turnover margin.

Three specific published claims were checked against the data rather than
repeated:

| Claim | Verdict |
|---|---|
| +191 regular season / +246 with playoffs, best by a champion since the 1999 Rams | **Supported** — both figures reproduce exactly |
| 17.2 points per game allowed, NFL's No. 1 scoring defense | **Supported** — 17.2, ranked 1st |
| 2nd-best average end-of-game margin among champions since 2000, behind only the 2013 Seahawks | **Differs** — 3rd. The 2016 Patriots also finished ahead |
| Best 8-week defensive stretch in 25 years, at −0.34 EPA/play | **Not confirmed as a superlative** — see below |

The last two are worth dwelling on, because in both cases the underlying instinct
was right and the specific number was not.

The "average end-of-game margin" statistic is just point differential per game
wearing a better name. It cannot tell a team that led 24–0 at halftime from one
that trailed and scored 24 unanswered. But the article's own argument — that
Seattle "gets on top early, piles on, and keeps it up" — is a claim about the
*shape* of a game, and that is measurable: weight the score margin by how long it
was held. Do that, and Seattle ranks **1st among all 26 champions since 2000**,
ahead of the 2013 team. The published stat put them 3rd; the stat the published
argument was actually reaching for puts them 1st.

Similarly, the −0.34 EPA/play eight-week defensive stretch does not reproduce
here on either of the two reasonable definitions. Computing the same rolling
window for every team-season since 1999 puts Seattle's best stretch 19th of 861
unfiltered, or 2nd of 861 once garbage time is excluded — excellent, but not a
clean "best in 25 years." What *does* hold on both definitions is the comparison
the claim was built around: this defense's best stretch was better than the 2013
Legion of Boom's.

**And the record sat inside a wide range.** Replaying the season 20,000 times —
same team, same opponents, same +191 quality, only the bounces re-rolled — makes
14–3 the single most likely outcome, but only 25% of the time. The 90% range runs
from 11 wins to 16. The same team lands on 12–5 about as often as on 15–2. Three
losses by nine total points is not evidence of a team that couldn't lose; it's
what the favourable side of a 17-game sample looks like. That is not an argument
that Seattle was lucky — the simulation assumes they were exactly as good as they
were. It's an argument that a season is a small sample, and greatness and record
are not the same measurement.

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
- "All-time" in this report means "since 1999," which is as far back as
  nflverse publishes EPA and win probability. The 1985 Bears and the 1972
  Dolphins are not in the comparison set, and no claim here should be read as
  ranking against them.
- Two widely cited figures about this team could not be checked at all, and
  are reported as external context rather than reproduced: **DVOA**, which is
  proprietary, and **blitz rate**, because the play-by-play carries no
  participation or pass-rusher data. For the same reason "pressure" throughout
  this project is a `sack OR qb_hit` proxy, which is narrower than a charted
  pressure and reads lower than published rates.
- Where a computed result disagreed with a published one, the computed value
  is reported and the disagreement is named. That happened twice, and in both
  cases the published direction was right while the specific number was not.

---

## Technical details & methodology

*For readers who want to evaluate the statistical rigor behind the story
above. All figures below are pulled directly from this project's output
files (`outputs/*.json`), not re-derived for this writeup.*

### Data and scope

Play-by-play, schedule, and player-stat data come from `nflverse` via the
`nflreadpy` package (CC-BY-4.0). Coverage widens by phase: 2024–2025 for the
year-over-year comparison (Phases 2–7), 2010–2025 for the defense anomaly
baseline (Phase 5), and **1999–2025 — 861 team-seasons — for the all-time
comparisons** (Phases 13–16), which is as far back as nflverse publishes EPA and
win probability. Regular-season games only, unless a section says otherwise.

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
| Red zone TD% | 57.1% | 54.2% |
| Pressure rate allowed | 16.9% | 12.2% |
| Pressure rate created | 17.5% | 16.1% |

Red-zone TD% and the two pressure rates are computed from summed season
counts (touchdowns/trips, pressures/dropbacks) rather than an average of
weekly percentages, so a 1-trip week doesn't get the same weight as a
5-trip week — the same method the Phase 7 decomposition below uses, so the
two sections' numbers agree exactly. Note red-zone TD% and pressure rate
created both *declined* slightly — a detail the narrative section calls out
rather than smoothing over, and one the Phase 7 decomposition below
confirms mattered little either way.

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

### Historical baseline (Phase 13)

`src/season_metrics.py` builds one row per team-season for 1999–2025 and commits
it as `data/processed/team_season_advanced.csv`, so every downstream consumer —
the four new phase scripts, the dashboard, the tests — reads a small CSV rather
than the multi-gigabyte play-by-play cache. EPA and success rate reuse Phase 4's
exact core-play filter (downs 1–4, pass/run, win probability 5–95%), so the
numbers stay directly comparable to the earlier phases instead of being a second,
subtly different definition of the same statistic.

Two era details the wider window forces, both already solved in Phase 5: the
relocation map (STL→LA, SD→LAC, OAK→LV) keeps a franchise as one entity, and the
league had 31 teams in 1999–2001 before Houston arrived, so the regular-season row
count is 3×31 + 24×32 = **861**, not 32×27. A blank-string team code in the 2000
play-by-play, which `notna()` lets through, produced a phantom 32nd team in that
season until it was caught by that row-count assertion.

Every metric is reported at **two percentiles**: raw, and era-adjusted via a
within-season z-score. The scoring environment moved substantially across this
window, so a raw percentile quietly flatters modern offenses. Where the two
disagree the headline figure is deliberately the *less* flattering of the pair.
Ranks count ties as half and run in each metric's own good direction (1 = best).

- Top-15 all-time for SEA 2025: explosive plays allowed (**12th of 861**) and
  clock-weighted lead (**13th**). Point differential per game ranks 25th.
- Middle of the pack: turnover margin (**450th**) and three-and-outs forced
  (**438th**).
- The era adjustment moves things in both directions — points per drive allowed
  looks better era-adjusted (98th vs 88th percentile), points per drive scored
  looks worse (72nd vs 87th).

### Game control (Phase 14)

Clock-weighted margin is `Σ(margin after play × seconds until next play) /
total seconds` — the score margin integrated against the game clock rather than
against plays. It separates a team that led wire-to-wire from one that won late
by the same score, which average final margin cannot do. Overtime carries zero
weight, because nflverse reports zero seconds remaining throughout OT; the metric
is therefore regulation-clock-weighted, and is described that way rather than as
a whole-game measure.

Combining regular season and playoffs weights each phase by its game count —
averaging the two averages would give a three-game postseason the same say as a
seventeen-game regular season.

- The published claim, tested on its own terms: SEA 2025 averaged **+12.30**
  points per game including playoffs, ranking **3rd of 26** champions since
  2000, behind 2013 SEA (+12.37) and 2016 NE. The reported figure was 2nd.
- On clock-weighted margin, SEA 2025 is **1st of 26** champions at **+7.40**,
  ahead of 2013 SEA's +6.21.
- Across all 861 regular seasons: clock-weighted margin 13th, share of clock
  with win probability above 75% 48th, share of clock leading 73rd.
- Seattle never trailed at any point in **9 of its 20 games**.

### Defense deep dive (Phase 15)

The "best eight-week stretch in 25 years" claim is tested by computing the same
rolling eight-game window for *every* team-season since 1999, not just Seattle's
— computing it for one team and repeating the label would assume the conclusion.
Windows never cross seasons, and the postseason is excluded so deep playoff runs
don't supply extra windows to draw from.

Two bases are reported, because published streak numbers are normally computed
without a garbage-time filter while the rest of this project applies one:

| Basis | SEA 2025 best window | Rank of 861 | Beats 2013 SEA? |
|---|---|---|---|
| Unfiltered | −0.245 (weeks 11–18) | 19th | Yes (−0.195) |
| Garbage-time filtered | −0.452 (weeks 7–15) | 2nd | Yes (−0.254) |

Neither reproduces the published −0.34, so the superlative isn't confirmed; the
relative comparison the claim was built on is, on both bases. Note the filtered
version is the *harsher* test for a dominant defense, which spends more of its
snaps in garbage time precisely because it is dominant.

Season-long, the defense ranks 1st in the 2025 NFL in EPA/play allowed, points
allowed per game, points per drive allowed, and yards per carry allowed, and 2nd
in explosive plays allowed and opponent scoring-drive rate — but only 6th in
takeaways and 11th in three-and-outs forced.

### Counterfactual (Phase 16)

Three independent angles on how much of 14–3 was the team and how much was the
bounce.

**Pythagorean** (reusing `src/pythagorean.py`, NFL exponent 2.37): 483 points for
and 292 against imply **13.04 expected wins** against 14 actual, a +0.96 gap
consistent with the pattern Phase 2 found in both seasons.

**Monte Carlo**: a margin model — `expected margin = own point differential per
game − opponent's + home-field advantage` — fit across all 272 games of 2025
recovers a home-field edge of **+2.10** points and a residual SD of **11.47**.
Seattle's actual 17-game schedule is then replayed 20,000 times. This is
retrospective rather than predictive (the ratings already know how the season
went) and treats games as independent; its job is to bound the variance in a
17-game sample. The residual SD comes in under the ~13 usually quoted for NFL
margins precisely because the ratings are fit in-sample, so the resulting win
range is, if anything, slightly too narrow.

- 14 wins is the modal outcome at **25.2%**; the mean is **13.60**.
- 90% range: **11 to 16 wins**. 8.8% of replays finish at 11 or fewer.
- Seattle went **6–3** in one-score games, with all three losses in that bucket.

**Turnover luck**: forcing a fumble is a skill, recovering one is close to a coin
flip. Holding the recovery share at the league rate and leaving the forcing alone
moves Seattle's turnover margin from **−1** to **−5.5** — a bounce component of
**+4.5**. So the already-mediocre turnover margin was, if anything, flattered by
how loose balls fell.

### Limitations

- Phase 4-7's EPA/play features are aggregated with **equal weight per
  game**, not weighted by play volume within a season — a documented
  simplification, not an oversight, made to avoid re-reading raw play-by-play
  a second time after Phase 4 already built the feature file. Red-zone TD%
  and the two pressure rates use a different, count-weighted aggregation
  instead (see Phase 4 above) so uneven weekly sample sizes don't distort
  the rate.
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
