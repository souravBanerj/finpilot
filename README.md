# FinPilot — a forward-looking, multi-agent money copilot

FinPilot ingests your transactions, understands your behaviour, **forecasts where your money is heading**, **simulates "what-if" decisions against your real data**, and proactively recommends *specific* actions — each with a recorded reason. It is not a budgeting dashboard: the point is the forward-looking part ("can I afford this?", "you will miss your goal by March").

All LLM calls go through **Mesh API** (`https://api.meshapi.ai/v1`, OpenAI-compatible) from a single module, `app/llm.py`.

## How the agents cooperate

```
 CSV / manual ─▶ COORDINATOR (LangGraph StateGraph; sequential fallback if not installed)
                     │
   categorize ─▶ analyze ─▶ forecast ─▶ track_goals ─▶ advise ─▶ decisions table (audit trail)
   Categorizer   Analyzer   Forecaster   Goal tracker   Advisor
   (rules+memory (pandas:    (EWMA+fixed   (schedule vs    (specific recs;
    +recurrence   spikes,     bills, 80%    forecast)       LLM only rephrases)
    +Mesh LLM)    outliers,   band, what-if
                  dupes, subs) simulator)
   Other routes via coordinator.handle(): simulate · chat · digest (APScheduler cron)
```

| Agent | Its own job | Its own logic/tools |
|---|---|---|
| **Categorizer** | label each transaction | merchant memory → ordered rules → recurrence detection → *one batched* Mesh LLM call for unknown merchants → `Other`. Stores `method` + `confidence` |
| **Analyzer** | find patterns/anomalies in *your* data | month-over-month spikes (volatility-aware), robust-z outliers, duplicate charges, overlapping/unused/price-hiked subscriptions, savings-rate drop |
| **Forecaster** | project the trajectory & simulate | per-category methods, 80% band, balance path, `simulate_purchase`, `simulate_goal`, goal tracking |
| **Advisor** | turn findings into actions | deterministic candidates with amounts + `reason` + `trigger_ref`; Mesh LLM may rephrase but a **number-grounding guard** rejects any invented figure |
| **Chat** | free-form questions | "can I afford…" → simulator (computed, not guessed); other questions → LLM over a compact JSON of *your* facts |

**Grounding principle:** numbers are computed by code; the LLM does fuzzy categorization and wording only. That is also what makes the accuracy numbers below measurable.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add MESH_API_KEY=rsk_...   (.env is gitignored)
uvicorn app.main:app --reload   # API + interactive docs at http://localhost:8000/docs
streamlit run ui/streamlit_app.py   # optional UI
```
Quick demo with no CSV: `curl -X POST "localhost:8000/demo/load?seed=1"`, then:
```bash
curl localhost:8000/insights
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"question":"Can I afford a ₹50,000 trip next month?"}'
curl -X POST localhost:8000/goals -H 'content-type: application/json' -d '{"name":"Goa trip","target_amount":150000,"deadline":"2027-06-30","saved_so_far":10000}'
curl localhost:8000/recommendations        # the audit trail: reason + trigger_ref + evidence for every recommendation
curl -X POST localhost:8000/digest/send    # same function the weekly scheduler job runs
```
Sample files to try: `sample_data/sample_transactions.csv` (and a debit/credit variant). Upload via `POST /upload` on `/docs`.

CSV format: `date,description,amount` (negative = expense) **or** `date,description,debit,credit`. ISO dates are exact; other formats are read day-first.

| Endpoint | Purpose |
|---|---|
| `POST /upload`, `POST /transactions` | import CSV / log one transaction (runs the pipeline) |
| `GET /insights`, `GET /forecast` | findings + profile; projected balance with low/base/high |
| `POST /simulate/purchase`, `/simulate/goal` | what-if against your trajectory |
| `POST/GET /goals` | goal tracking (advice adapts when you fall behind) |
| `GET /recommendations` | **audit trail** |
| `POST /chat` | conversational layer |
| `PATCH /transactions/{id}` | correct a category → merchant memory learns it |
| `POST /settings/subscription-usage` | optional `{merchant: last_used_date}` → enables "unused subscription" findings |

The weekly digest runs on a **real APScheduler cron job** (`DIGEST_CRON_*` in `.env`), delivered by Telegram if configured, otherwise written to `data/digests/`.

## Key design decisions
1. **Deterministic core, LLM at the edges.** Forecasts and findings are code, so they are testable. The LLM categorizes unknown merchants (batched, cached into memory) and phrases advice.
2. **Every recommendation is auditable.** `decisions` rows store `reason`, `trigger_ref` (e.g. `subscription_overlap:video streaming`, `forecast:low_buffer`, `goal:Goa trip`) and the numeric `evidence`. Simulations are logged too.
3. **Volatility-aware anomaly rules.** A spike must beat a % threshold, an absolute floor *and* the category's own normal swing, otherwise lumpy categories cry wolf.
4. **Honest "unused subscription".** Transactions alone cannot show usage, so we detect *overlap*, *price hikes*, and recurring charges of unknown merchants; true "unused" needs the optional usage input.
5. **Uncertainty is shown.** Forecasts carry an 80% band and the simulator checks the pessimistic case against a one-month-of-bills safety buffer.

## Self-evaluation (full tables: [`eval/results.md`](eval/results.md); reproduce: `python -m eval.run_eval`)
20 seeded synthetic personas × 12 months = 12,766 transactions with ground-truth labels and injected anomalies.

| What | Result |
|---|---|
| Categorization accuracy (offline: rules + recurrence, **no LLM**) | **91.7%**, macro-F1 **0.969** (rule layer 100%, recurrence layer 100%) |
| Remaining error | 8.3% unknown merchants fall to `Other` — the job of the Mesh LLM layer, **not measured here** (no key was available when results were generated; the harness measures it automatically when `MESH_API_KEY` is set) |
| Duplicate charges / subscription overlap / price hike / hidden subscriptions | 100% precision, 100% recall |
| Outlier transactions | 89% precision, 100% recall |
| Category-spike detection | 57% precision, 80% recall — a real trade-off (sensitivity sweep in results.md) |
| Forecast MAPE, total monthly spend | **14.9%** vs 18.5% naive (last month) and 16.1% (3-month mean) |
| 80% prediction-band coverage | 87.5% (slightly conservative) |

**What the evaluation taught us** (v1 → current): spike precision was 43% until we added the volatility check; 16 outlier false positives were all `SWIGGY INSTAMART` (groceries) mislabelled as delivery → added explicit grocery rules; duplicate "100%" was initially reported as 72% because of an *evaluation* bug (interchangeable same-day rows), fixed by pair matching; price-hike detection found nothing until it scanned the last 6 charges instead of the last 2.

**Caveats, stated plainly:** data is synthetic and written by us, so absolute numbers are optimistic versus real bank exports; some thresholds were tuned on it; per-category forecasts are weak for lumpy categories (Dining/Shopping MAPE > 100%) — total spend is the reliable number; Utilities is slightly worse than naive (seasonality is not modelled); the model is flat beyond month 1 (no seasonality/one-offs).

## Tests
`python -m unittest discover -s tests -v` — 24 tests: parser, categorizer layers, analyzer, simulator math, grounding guard, audit trail, goal adaptation, end-to-end pipeline on a temp DB, and an accuracy regression floor.

## Repo layout
```
app/        config, db (SQLite), llm (Mesh), ingest, coordinator (LangGraph), digest, scheduler, main (FastAPI)
agents/     categorizer, analyzer, forecaster, advisor, chat, common
eval/       generate_data (ground truth), run_eval, results.md/json
ui/         streamlit_app.py        tests/   test_core.py        docs/BUILD_GUIDE.md        sample_data/  example CSVs
```
