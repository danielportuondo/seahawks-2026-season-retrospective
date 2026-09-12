# Seahawks Super Bowl LX Retrospective

A data science / statistics portfolio project on how the Seattle Seahawks
went from missing the playoffs in 2024 (10–7) to winning Super Bowl LX in
2025 (14–3) — Sam Darnold led the NFL in regular-season turnovers, then
posted zero across three playoff wins.

Full write-up: **[REPORT.md](REPORT.md)** — plain-language story plus a
"Technical Details & Methodology" section (Pythagorean win expectation,
z-scores/Mahalanobis distance, a Poisson rate model, and a regression
decomposition).

## Dashboard

![Dashboard screenshot](docs/dashboard_screenshot.png)

A two-tab Streamlit app: **The Story** (plain-language narrative, headline
charts, an interactive season-trend explorer) and **Methodology** (the
statistical detail behind it). Reads only from `outputs/` and
`data/processed/` — no live recomputation.

## Repo layout

- `data/raw/` — cached raw pull from nflverse (not committed; re-fetchable via `src/data_acquisition.py`)
- `data/processed/` — engineered feature files used by later analysis and the dashboard
- `src/` — pipeline and analysis code, one script per phase
- `outputs/` — analysis results (JSON) and charts
- `dashboard/` — Streamlit app (`app.py`)
- `tests/` — unit tests for the pure calculation functions (Pythagorean expectation, turnover-probability model, OLS decomposition)
- `docs/` — README assets
- `REFERENCES.md` — data source and citation log
- `HANDOFF.md` — the original phase-by-phase project plan

## Setup

Requires Python 3.12+.

```bash
uv venv
uv pip install -r requirements.txt
```

## Running the pipeline

Each phase is a standalone script; run in order (Phase 1 caches the raw
data everything downstream depends on):

```bash
uv run python src/feasibility_check.py     # Phase 0: confirm network access to nflverse
uv run python src/data_acquisition.py      # Phase 1: pull + cache raw data to data/raw/
uv run python src/phase2_puzzle.py         # Phase 2: 2024 Pythagorean win-expectation gap
uv run python src/phase3_personnel.py      # Phase 3: QB/OC personnel deltas
uv run python src/phase4_deep_dive.py      # Phase 4: EPA/turnover/red-zone/pressure features
uv run python src/phase5_anomaly_detection.py  # Phase 5: defense anomaly detection (z-scores, Mahalanobis)
uv run python src/phase6_turnover_rate_model.py # Phase 6: zero-turnover playoff probability
uv run python src/phase7_decomposition.py  # Phase 7: regression decomposition of the 2024->2025 jump
```

Phases 5 and 6 only depend on Phase 1's cached data and can run in either
order; Phases 2–4 and 7 depend on the outputs before them in the list.

## Running the dashboard

```bash
uv run streamlit run dashboard/app.py
```

## Tests and lint

```bash
uv run pytest
uvx ruff check .
```

## Status

All 11 phases complete. See `HANDOFF.md` for the original phase-by-phase
plan this project followed.
