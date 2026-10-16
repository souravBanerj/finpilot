# Build guide — FinPilot from scratch, in the order it was built

Each step ends with a **checkpoint** you can run. Build in this order: every layer is testable before the next one exists.

## Step 0 — Setup
`python -m venv .venv && source .venv/bin/activate`; create the folders `app agents eval ui tests`; write `requirements.txt` (fastapi, openai, langgraph, apscheduler, pandas, numpy, …), `.gitignore` (**must include `.env`**), `.env.example`.
Checkpoint: `pip install -r requirements.txt` succeeds.

## Step 1 — Config, database, Mesh client (`app/config.py`, `app/db.py`, `app/llm.py`)
- Config reads env vars (`MESH_API_KEY`, thresholds). Base URL constant `https://api.meshapi.ai/v1`.
- SQLite tables: `transactions`, `decisions` (the audit trail: `reason`, `trigger_ref`, `evidence`), `goals`, `merchant_memory`, `settings`, `digests`.
- `llm.py` is the *only* file that talks to an LLM: `OpenAI(base_url=MESH_BASE_URL, api_key=MESH_API_KEY)`, plus `available()` so everything degrades gracefully without a key.
Checkpoint: `python -c "from app import db; db.init_db()"` creates `data/finpilot.db`.

## Step 2 — Categorizer (`agents/categorizer.py`)
1. `normalize_merchant`: strip rails (UPI/POS), refs, digits → `SWIGGY`.
2. Ordered keyword rules, **specific before generic** (`AMAZON PRIME` before `AMAZON`).
3. `detect_recurring`: ≥3 charges, ~monthly gaps, steady amount → unknown merchants become Subscriptions.
4. Batched Mesh LLM call for what is left; cache results into `merchant_memory`.
5. Fallback `Other` with low confidence. Always store `method` + `confidence`.
Checkpoint: smoke-test a few descriptions; later the eval gives accuracy.

## Step 3 — Shared helpers + Analyzer (`agents/common.py`, `agents/analyzer.py`)
Monthly category matrix over *complete* months only. Detectors, each emitting `{id, type, severity, title, evidence}`: category spike, robust-z outliers (median/MAD on log amounts), duplicates, subscription overlap/price-hike/unused, savings-rate drop. The `id` is what the advisor cites later.

## Step 4 — Forecaster + simulator (`agents/forecaster.py`)
Rent/Subscriptions = last month; Utilities = 3-mo mean; other categories = EWMA(α=0.4); income = 3-mo median. Sigma from in-sample one-step errors → 80% band that widens with √horizon. Then `simulate_purchase` (balance path − amount, buffer test on base *and* pessimistic case, "cut X/month to make it work"), `simulate_goal` (required vs projected per month, ETA), `goal_status` (schedule vs forecast). Add `backtest()` right away — it is how you know the model is any good.

## Step 5 — Synthetic data with ground truth (`eval/generate_data.py`)
Seeded personas; inject labelled events (spike, outliers, duplicates, price hike, overlapping + unknown subscriptions) and include hard cases (unknown merchants, `AMAZON FRESH`). Without labels you cannot compute precision/recall.

## Step 6 — Evaluation harness (`eval/run_eval.py`)
Categorization accuracy/F1/per-layer; anomaly precision/recall per detector; rolling-origin forecast backtest vs naive & mean-3 baselines; interval coverage. **Read the failures, not just the score** — each fix in the README came from a diagnosed failure.

## Step 7 — Advisor (`agents/advisor.py`)
Deterministic candidates (amount, reason, trigger_ref, evidence) from findings + forecast + goals; sorted by severity then impact. Optional Mesh rephrase with a **number-grounding guard**. `compose_digest` = one insight + one action.

## Step 8 — Orchestration, chat, API, scheduler, UI
`app/ingest.py` (messy CSVs) → `app/coordinator.py` (LangGraph graph `categorize→analyze→forecast→track_goals→advise`, with a sequential fallback; `handle()` routes simulate/chat/digest) → `agents/chat.py` (afford-intent → simulator) → `app/digest.py` + `app/scheduler.py` (APScheduler cron) → `app/main.py` (FastAPI) → `ui/streamlit_app.py`.
Checkpoint: `uvicorn app.main:app` then `POST /demo/load` and open `/docs`.

## Step 9 — Tests (`tests/test_core.py`)
Cover every layer plus an accuracy regression floor. `python -m unittest discover -s tests`.

## Step 10 — Ship
1. `python -m eval.run_eval` and commit `eval/results.*`.
2. Download the hackathon workflow from the URL in the brief into `.github/workflows/agent-arena-2026-checks.yml`.
3. Add repo secrets `MESH_API_KEY` and `SUBMISSION_TOKEN`. Confirm `.env` is **not** committed.
4. Push; check the Actions tab. Record a 2-minute demo: load demo → insights → "can I afford…?" → audit trail.
