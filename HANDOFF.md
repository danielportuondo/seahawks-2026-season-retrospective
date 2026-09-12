# Seahawks Super Bowl LX Project — Claude Code Handoff

## Context

This is a personal data science / statistics portfolio project analyzing how the
Seattle Seahawks went from missing the playoffs in 2024 to winning Super Bowl LX
in 2025. The goal is a working, genuinely interesting analysis — not a
production system. **Move fast. Favor a complete, working pipeline over
polish. Light testing is fine; a full test suite/CI is not the goal.**

**This build spans multiple sessions.** Don't try to push through all 11
phases in one sitting — work through however many phases fit well in the
current session, then stop at a clean point (data cached, a phase's outputs
written, etc.) rather than leaving something half-finished. The user runs
their own `/session-handoff` skill between sessions to carry context
forward, so you don't need to build your own handoff mechanism — just leave
the repo in a coherent state (committed/saved outputs, clear on which phases
are done) so the next session picks up cleanly. Good natural stopping points:
after Phase 1 (data cached), after Phases 5–6 (both subagent analyses done),
after Phase 9 (dashboard working).

### Ground-truth facts (verify data against these — don't re-derive or guess)

- **2024 Seahawks:** 10–7, missed playoffs (lost NFC West tiebreaker to the
  Rams on strength-of-victory; only 10-win team to miss the playoffs since the
  17-game era began). Geno Smith at QB, Ryan Grubb as OC.
- **2025 Seahawks:** 14–3, NFC West champions. Sam Darnold at QB (signed after
  Geno Smith was traded), Klint Kubiak as OC. Same DC both years (Aden Durde,
  "Dark Side" defense).
- **Playoffs:** beat 49ers 41–6 (Divisional), beat Rams 31–27 (NFC
  Championship), beat Patriots 29–13 in Super Bowl LX (Feb 8, 2026, Levi's
  Stadium). **Zero turnovers across all three playoff games** — the first
  Super Bowl champion ever to complete an entire postseason without one.
  Kenneth Walker III was Super Bowl MVP (135 yards on 27 carries).
- **Darnold's regular season:** led the NFL with 20 turnovers (14 INT, 6 lost
  fumbles) over 17 games — the league's most turnover-prone qualifying passer
  — then had zero turnovers in the postseason.

### Deliverables

1. A repo (`/data`, `/src`, `/outputs`, `/dashboard`, `README.md`,
   `requirements.txt`, `REFERENCES.md`) with a working, reproducible
   pipeline.
2. A markdown narrative report — **written for a general, non-technical
   audience** (recruiters, hiring managers) — with a clearly separated
   "Technical Details & Methodology" section for anyone who wants the
   statistics.
3. A Streamlit dashboard (chosen over Plotly Dash for build speed) with the
   same split: a main tab in plain language, and a distinct "Methodology"
   tab holding the statistical detail.
4. A `REFERENCES.md` crediting data sources and any external facts/quotes
   used, maintained throughout rather than reconstructed at the end.

### Testing philosophy

Spot-check outputs against the ground-truth facts above and sanity-check that
statistical results are in plausible ranges (probabilities in [0,1], z-scores
not absurd, etc.). Write a handful of unit tests only for pure calculation
functions (Pythagorean win expectation, the turnover-probability model). Skip
broad coverage, mocking, and CI setup — this isn't that kind of project.

### Guardrails & practical constraints

- **Data access:** `nfl_data_py` is now deprecated in favor of `nflreadpy` —
  prefer `nflreadpy` if it's a clean swap, otherwise `nfl_data_py` still
  works fine for this. Either way, the underlying data comes from
  nflverse's GitHub releases, so in Phase 0, confirm the environment can
  actually reach `github.com`, `raw.githubusercontent.com`, and
  `release-assets.githubusercontent.com` before building anything on top of
  it. If those are blocked in whatever environment Claude Code is running
  in, ask the user how to get the data rather than working around it
  silently.
- **Team logos/broadcast photos:** okay to use, including screenshots of
  broadcast footage or team photos, for this personal, non-commercial
  portfolio piece. Source from team media pages, Wikimedia Commons, or
  broadcast screenshots as needed. Player names and stats are separately
  fine to use freely either way — this is factual analysis, not creative
  content involving public figures.
- **Quoting news articles:** when writing the narrative, don't lift more
  than a short, attributed phrase from any single article — paraphrase in
  your own words and log the source in `REFERENCES.md`.
- Keep `REFERENCES.md` current as you go (see Phases 1 and 11) rather than
  trying to reconstruct sourcing at the end.

### Before starting — ask the user

1. Repo location, and whether to init git / push to a GitHub remote now or
   just build locally first.
2. Python env manager preference for the actual project (venv/conda/uv) —
   default to `venv` if no answer. (This is separate from `cenv`, the
   session-tracking tool — see the verification step in Phase 0.)
3. Dashboard target: local-only demo (`streamlit run`) or deploy to Streamlit
   Community Cloud (needs a GitHub repo).
4. How far back to pull historical team-defense data for the anomaly
   baseline (Phase 5) — recommend 2010–2025 as a default that balances
   sample size against how much the game has changed; confirm before
   committing to it.

**Also stop and ask** (don't guess and push forward) if: a data pull fails,
an actual number contradicts the ground-truth table, or a statistical result
looks implausible.

---

## Model & effort plan at a glance

| Phase | Task | Model | Effort | Subagent? |
|---|---|---|---|---|
| 0 | Setup & feasibility check | Sonnet 5 | low | no |
| 1 | Data acquisition | Sonnet 5 | medium | no |
| 2 | The puzzle (2024 Pythagorean gap) | Sonnet 5 | low | no |
| 3 | Personnel/scheme deltas | Sonnet 5 | low | no |
| 4 | Statistical deep dive (EPA, etc.) | Sonnet 5 | medium | no |
| 5 | Anomaly detection — Dark Side D | Opus 5 | high | **yes, parallel w/ 6** |
| 6 | Zero-turnover rate model | Opus 5 | high | **yes, parallel w/ 5** |
| 7 | Regression decomposition | Sonnet 5 | medium | no |
| 8 | Narrative writeup | Sonnet 5 | medium | no |
| 9 | Dashboard | Sonnet 5 | medium | no |
| 10 | Light QA pass | Sonnet 5 | low | no |
| 11 | README & polish | Sonnet 5 | low | no |

Run the session in `opusplan` mode if you want Opus reasoning on the trickier
design calls throughout without paying for it on the routine phases — or just
switch model/effort per-phase as above using `/model` and `/effort`.

---

## Phase-by-phase

### Phase 0 — Setup & feasibility check
**Model: Sonnet 5, low effort**
- **`cenv` session-start verification.** `cenv` is a separate session-history
  tracking tool (CLI at `~/.cargo/bin/cenv`) — not a Python environment.
  Do the normal first-reply work for the session as usual, then send this
  as your **second message** (the history export only appears once the
  first reply finishes):

  > Check that cenv is capturing this session. Run `~/.cargo/bin/cenv
  > doctor` and confirm it reports healthy. Then look under
  > `~/.local/share/cenv/history/` for a folder matching this project and
  > confirm its `INDEX.md` lists the current session. Also tell me whether
  > your session-start context included a note pointing to a cenv history
  > index. Report what you find plainly.

  A good result: `doctor` reports healthy; a history folder exists for the
  project with an `INDEX.md` row for this session; Claude confirms it saw a
  history pointer at session start. Report the actual findings rather than
  assuming success.

  **After this session ends,** to verify the end-of-session summary: start a
  new session in the same folder and ask Claude to open the latest session
  export in that history folder and report whether it contains a
  Haiku-written summary or only the heuristic one. If it's only heuristic,
  run `cenv analyze` from that folder to retry the summary.

- Set up the project's Python environment per the venv/conda/uv answer
  above — a normal Python virtual environment, unrelated to `cenv`.
- Scaffold the repo structure (including an empty `REFERENCES.md`) and
  `requirements.txt`: `pandas`, `nflreadpy` (or `nfl_data_py` if that's not
  a clean install — see Guardrails above), `numpy`, `scipy`, `scikit-learn`,
  `matplotlib`, `plotly`, `streamlit`.
- **Before building anything else**, do a small test pull (e.g. one season of
  schedules) to confirm the data package can actually fetch data in this
  environment — see the network-access note under Guardrails. If it fails,
  ask the user how to proceed (e.g. manually downloading CSVs from the
  nflverse GitHub releases) rather than guessing at a workaround.
- Ask the clarifying questions above if not already answered.

### Phase 1 — Data acquisition
**Model: Sonnet 5, medium effort**
- Pull play-by-play, schedules/results, and season-level team stats for the
  confirmed historical window plus 2024–2025.
- Cache everything to `/data/raw` (parquet) so later phases don't re-pull.
- Validate: confirm the 2024 and 2025 Seahawks records match the ground-truth
  table before moving on.
- Add an entry to `REFERENCES.md` for the nflverse/nflreadpy data itself
  (CC-BY-4.0; if FTN Data charting fields are pulled in, that subset needs
  separate attribution to "FTN Data via nflverse" under CC-BY-SA 4.0).

### Phase 2 — The puzzle: 2024 retrospective
**Model: Sonnet 5, low effort**
- Compute Pythagorean win expectation for the 2024 Seahawks from points
  for/against; compare to their actual 10–7 record.
- One short section + one chart.

### Phase 3 — Personnel/scheme deltas
**Model: Sonnet 5, low effort**
- Pull QB-level and team-level splits for 2024 vs. 2025 (completion %, yards,
  INT rate, pressure rate allowed under each OC).
- A comparison table + chart is enough here — this section is scene-setting,
  not the analytical core.

### Phase 4 — Statistical deep dive
**Model: Sonnet 5, medium effort**
- From play-by-play: EPA/play (offense & defense), turnover margin, red zone
  TD%, pressure rate — 2024 vs. 2025, plus a weekly rolling trend.
- Save engineered features to `/data/processed/team_week_features.csv` — this
  file feeds Phase 7 and the dashboard.
- Medium effort because EPA aggregation logic (down/distance filters, garbage
  time exclusion) is easy to get subtly wrong — worth a bit more care here
  than in Phases 2–3.

### Phase 5 — Anomaly detection: how extreme was the Dark Side defense?
**Model: Opus 5, high effort — dispatch as a subagent, parallel with Phase 6**
- Build the multi-season team-defense dataset (EPA/play allowed, sack rate,
  takeaway rate, points allowed/game) over the confirmed historical window.
- **Normalize each metric relative to that season's league average/SD before
  pooling across years** — this is the step most likely to go wrong if
  rushed, hence the higher effort budget.
- Compute per-metric z-scores for the 2025 (and 2024, for contrast) Seahawks
  defense, then a multivariate outlier score: Mahalanobis distance as the
  primary method, Isolation Forest as a secondary cross-check.
- Output: `/outputs/defense_anomaly_results.json` + a radar chart of z-scores
  and a historical-rank distribution chart.
- Only depends on Phase 1's cached data, so it can run independently.

### Phase 6 — Zero-turnover postseason as a rate-model problem
**Model: Opus 5, high effort — dispatch as a subagent, parallel with Phase 5**
- Compute Darnold's regular-season turnover rate (per drive and per
  dropback).
- Fit a rolling-window turnover rate across the season to check the claim
  that he improved in the second half.
- Model turnovers as a Poisson (or negative binomial) rate process; compute
  P(zero turnovers across ~3 playoff games' worth of plays) under the
  regular-season rate.
- **Explicitly note in code comments and the writeup that this is a
  discrete rate-model problem, not classic extreme value theory** (EVT is
  for continuous magnitudes) — that distinction is worth preserving for
  credibility.
- Output: `/outputs/turnover_rate_model_results.json` + a rolling-rate chart
  and a probability visualization.
- Only depends on Phase 1's cached data, so it can run independently of
  Phase 5.

### Phase 7 — Quantifying the jump: regression decomposition
**Model: Sonnet 5, medium effort**
- Depends on Phase 4's feature file.
- A regression or feature-importance model (linear regression, or XGBoost +
  SHAP if you want to lean into more ML-flavored output) explaining the
  2024→2025 shift in wins/point differential from the feature deltas.
- Output: `/outputs/decomposition_results.json` + a feature importance chart.

### Phase 8 — Synthesis / narrative writeup
**Model: Sonnet 5, medium effort**
- Depends on the output files from Phases 2–7.
- Write two clearly separated pieces from the same underlying results:
  - **Main narrative** — plain language, minimal jargon: the story of
    2024→2025 told through the findings ("the defense was a historical
    outlier," "the zero-turnover playoff run was extraordinarily unlikely
    given his regular season," etc.) without walking through the statistics
    themselves.
  - **Technical Details & Methodology** section — the actual methods:
    z-scores and Mahalanobis distance, the Poisson rate model and why it's
    the right tool here instead of classic EVT, the regression
    decomposition — for a reader who wants to evaluate the rigor.
- Pull numbers from the Phase 2–7 output files rather than re-deriving or
  estimating them. Paraphrase any facts drawn from news sources and log each
  source in `REFERENCES.md` as you use it.

### Phase 9 — Interactive dashboard
**Model: Sonnet 5, medium effort**
- Streamlit app reading from `/outputs` and `/data/processed` (don't
  recompute live).
- Structure as **two tabs**: a main tab with the plain-language story and
  headline charts (season trend explorer, the big-picture findings), and a
  separate **"Methodology" tab** with the statistical detail (z-scores/
  anomaly scores, the rate model and its assumptions, regression/
  feature-importance output).
- Team logos and broadcast-style photos are fine to include (see
  Guardrails) — source from team media pages, Wikimedia Commons, or
  broadcast screenshots. Claude Code will need working network access to
  fetch actual image files, or you can supply local image assets directly.

### Phase 10 — Light QA pass
**Model: Sonnet 5, low effort**
- Spot-check final numbers against the ground-truth table.
- A handful of unit tests on pure functions only (Pythagorean expectation,
  turnover probability calc).
- Smoke-test that the dashboard actually launches (`streamlit run app.py`).

### Phase 11 — README & polish
**Model: Sonnet 5, low effort**
- Setup instructions, project summary, a dashboard screenshot, pinned
  `requirements.txt`.
- Final pass on `REFERENCES.md`: confirm every external data source and any
  facts/quotes drawn from news articles along the way are credited.

---

## Subagent dispatch notes

Dispatch Phases 5 and 6 as two parallel subagents right after Phase 1
finishes — they're analytically independent and both only need the cached
base data. Give each a narrow prompt (exact input file paths, expected output
file path/schema) so they don't collide, and pin their config:

```
---
model: opus
effort: high
---
```

Keep the main/orchestrator thread on Sonnet 5 at its default effort for every
other phase — only Phases 5 and 6 need the extra reasoning budget.