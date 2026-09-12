# Seahawks Super Bowl LX Retrospective

A data science / statistics portfolio project analyzing how the Seattle
Seahawks went from missing the playoffs in 2024 (10–7) to winning Super Bowl
LX in 2025 (14–3).

**Status:** work in progress — Phase 0 (setup) complete. See `HANDOFF.md` for
the full project plan and phase-by-phase progress.

## Repo layout

- `data/raw/` — cached raw pull from nflverse (not committed; re-fetchable)
- `data/processed/` — engineered feature files used by later analysis and the dashboard
- `src/` — pipeline and analysis code
- `outputs/` — analysis results (JSON) and charts
- `dashboard/` — Streamlit app
- `REFERENCES.md` — data source and citation log

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

Full setup instructions, a project summary, and a dashboard screenshot will
be added in Phase 11.
