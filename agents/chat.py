"""Conversational layer. Money questions are answered from the user's real data:
  * 'can I afford X (next month)?'  -> routed to the forecaster's simulator (numbers computed, not guessed)
  * everything else                 -> LLM over a compact JSON of the user's findings/forecast/goals
                                       (system prompt forbids facts outside it); deterministic fallback offline.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import pandas as pd

from app import config, llm
from agents import advisor, forecaster

AFFORD_RX = re.compile(r"afford|can i (buy|spend|book|get|take)|should i (buy|book|get)|\bbuy\b|purchase|trip|vacation|laptop|phone", re.I)
AMOUNT_RX = re.compile(r"(?:₹|rs\.?|inr|\$)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|lakhs?|lacs?|l)?\b", re.I)


def parse_amount(q: str) -> Optional[float]:
    best = None
    for m in AMOUNT_RX.finditer(q):
        v = float(m.group(1).replace(",", ""))
        suf = (m.group(2) or "").lower()
        if suf == "k":
            v *= 1_000
        elif suf in ("l", "lakh", "lakhs", "lac", "lacs"):
            v *= 100_000
        if v >= 100 and (best is None or v > best):
            best = v
    return best


def parse_month_offset(q: str, fc: dict, today: pd.Timestamp) -> int:
    """Index into the forecast trajectory (1 = first forecast month)."""
    rel = 0
    ql = q.lower()
    if "next month" in ql:
        rel = 1
    elif (m := re.search(r"in (\d+) months?", ql)):
        rel = int(m.group(1))
    target = pd.Period(today, "M") + rel
    first = pd.Period(fc["trajectory"][0]["month"], "M")
    return int(min(max((target - first).n + 1, 1), len(fc["trajectory"])))


def _afford_text(res: dict) -> str:
    m = advisor.money
    head = {"yes": "Yes, you can afford it.", "tight": "Possible, but it would be tight.", "no": "I would not do it right now."}[res["verdict"]]
    s = (f"{head} {res['explanation']} If you buy {res['name']} ({m(res['amount'])}) in {res['month']}, your lowest projected balance "
         f"goes from {m(res['min_balance_before'])} to {m(res['min_balance_after_base'])} (pessimistic case {m(res['min_balance_after_low'])}); "
         f"your safety buffer is {m(res['safety_buffer'])}.")
    if res["monthly_cut_to_make_it_work"]:
        s += f" To make it work, cut about {m(res['monthly_cut_to_make_it_work'])}/month from spending."
    for g in res.get("goal_impact", []):
        if not g["on_track_after"]:
            s += f" It would push your goal '{g['goal']}' off track (extra {m(g['extra_needed_per_month'])}/month needed)."
    return s


def compact_context(state: dict) -> dict:
    a, fc = state.get("analysis", {}), state.get("forecast", {})
    return {"profile": a.get("profile"),
            "findings": [{"id": f["id"], "title": f["title"], "evidence": f["evidence"]} for f in a.get("findings", [])[:12]],
            "monthly_by_category_last3": {c: dict(list(v.items())[-3:]) for c, v in a.get("monthly_by_category", {}).items()},
            "forecast": None if "error" in fc else {k: fc[k] for k in ("monthly_income", "monthly_expenses", "free_cash_flow",
                                                                         "fixed_obligations", "start_balance", "by_category")}
                         | {"trajectory": fc["trajectory"]},
            "goals": state.get("goals", []), "recommendations": [r["message"] for r in state.get("recs", [])[:6]]}


def answer(question: str, state: dict, today: Optional[pd.Timestamp] = None) -> dict:
    today = today or pd.Timestamp.today().normalize()
    fc = state.get("forecast", {})
    if "error" in fc:
        return {"answer": fc["error"], "route": "none"}
    q = question.strip()
    if AFFORD_RX.search(q):
        amt = parse_amount(q)
        if amt is None:
            return {"answer": "How much is it? Tell me the amount and I will check it against your forecast.", "route": "clarify"}
        offset = parse_month_offset(q, fc, today)
        res = forecaster.simulate_purchase(fc, amt, offset, name="this purchase")
        res["goal_impact"] = []
        from app import db
        for g in db.list_goals():
            s = forecaster.simulate_goal(fc, g["target_amount"], g["deadline"], g["saved_so_far"], extra_spend=amt)
            if "error" not in s:
                res["goal_impact"].append({"goal": g["name"], "on_track_after": s["on_track"], "extra_needed_per_month": s["extra_to_save_per_month"]})
        text = _afford_text(res)
        return {"answer": text, "route": "simulation", "data": res}
    ctx = compact_context(state)
    if llm.available():
        try:
            txt = llm.chat("You are FinPilot, a personal-finance assistant. Answer ONLY from the JSON facts provided. "
                           "If the answer is not in them, say you do not have that data. Be specific and brief (max 120 words). "
                           f"Currency symbol: {config.CURRENCY}.", f"FACTS:\n{json.dumps(ctx, default=str)}\n\nQUESTION: {q}", max_tokens=350)
            return {"answer": txt.strip(), "route": "llm"}
        except Exception:
            pass
    return {"answer": _offline_answer(q, state), "route": "fallback"}


def _offline_answer(q: str, state: dict) -> str:
    a, fc, m = state.get("analysis", {}), state.get("forecast", {}), advisor.money
    ql = q.lower()
    if "subscri" in ql:
        s = [f for f in a.get("findings", []) if f["type"].startswith("subscription")]
        return "; ".join(f["title"] for f in s) or "No subscription issues found."
    if any(w in ql for w in ("goal", "target")):
        return " ".join(f"{g['goal']}: {g['schedule_status']}, saved {m(g['saved'])} of {m(g['target'])}." for g in state.get("goals", [])) or "You have no goals yet."
    if any(w in ql for w in ("forecast", "balance", "future", "month")):
        e = fc["trajectory"][-1]
        return f"At your current pace your balance reaches {m(e['balance_base'])} by {e['month']} (range {m(e['balance_low'])}–{m(e['balance_high'])})."
    top = a.get("profile", {}).get("top_categories_last3", {})
    return "Top spending (last 3 months): " + ", ".join(f"{c} {m(v)}" for c, v in top.items()) + "."
