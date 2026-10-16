"""AGENT 4 — Advisor.

Grounding contract (this is the anti-hallucination design):
  1. Candidate recommendations are built DETERMINISTICALLY from analyzer findings, the forecast and
     goal status. Each carries: specific items/amounts, `reason` (why), `trigger_ref` (the finding or
     forecast id that caused it) and `evidence` (the numbers). These are what get stored as the audit trail.
  2. The LLM (via Mesh) may only REPHRASE a recommendation into friendly language. A guard rejects any
     rewrite containing a number that is not in the facts it was given, and we fall back to the template.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from app import config

SEV = {"high": 0, "medium": 1, "low": 2}
DISCRETIONARY = ["Food Delivery", "Dining", "Shopping", "Entertainment", "Travel"]


def money(x: float) -> str:
    return f"{config.CURRENCY}{x:,.0f}"


def _rec(rec_type, title, action, impact, reason, trigger_ref, evidence, severity="medium"):
    return {"rec_type": rec_type, "title": title, "action": action, "impact_amount": round(impact or 0),
            "reason": reason, "trigger_ref": trigger_ref, "evidence": evidence, "severity": severity,
            "agent": "advisor", "message": f"{title}. {action}"}


def advise(analysis: dict, fc: Optional[dict] = None, goal_statuses: Optional[list[dict]] = None,
           max_recs: int = 10) -> list[dict]:
    recs: list[dict] = []
    lever_monthly = 0.0                                       # monthly savings available from other recs
    for f in analysis.get("findings", []):
        ev, t, sev = f["evidence"], f["type"], f["severity"]
        if t == "subscription_unused":
            recs.append(_rec("cancel_subscription", f"Cancel {f['merchant']}",
                             f"Cancel {f['merchant']} — saves {money(ev['monthly_cost'])}/month ({money(ev['saving_per_year'])}/year).",
                             ev["monthly_cost"],
                             f"Unused for {ev['days_unused']} days but still billing {money(ev['monthly_cost'])}/month.",
                             f["id"], ev, sev))
            lever_monthly += ev["monthly_cost"]
        elif t == "subscription_overlap":
            names = ", ".join(f"{m} ({money(c)})" for m, c in ev["costs"].items())
            recs.append(_rec("cancel_subscription", f"Trim overlapping {f['group']} subscriptions",
                             f"You pay for {len(ev['merchants'])} {f['group']} services: {names}. Keep the one you use most and cancel the rest "
                             f"to save up to {money(ev['saving_if_keep_one'])}/month ({money(ev['saving_if_keep_one']*12)}/year).",
                             ev["saving_if_keep_one"],
                             f"{len(ev['merchants'])} active subscriptions in the same group, together {money(ev['monthly_total'])}/month.",
                             f["id"], ev, sev))
            lever_monthly += ev["saving_if_keep_one"]
        elif t == "subscription_price_hike":
            recs.append(_rec("review_subscription", f"{f['merchant']} got more expensive",
                             f"{f['merchant']} rose from {money(ev['old'])} to {money(ev['new'])} (+{money(ev['extra_per_year'])}/year). "
                             f"Check for a cheaper plan or cancel if it is not worth it.",
                             ev["extra_per_year"] / 12, f"Latest charge is higher than the previous one since {ev['since']}.",
                             f["id"], ev, "low"))
        elif t == "duplicate_charge":
            recs.append(_rec("dispute_charge", f"Possible duplicate {ev['merchant']} charge",
                             f"{ev['merchant']} charged {money(ev['amount'])} twice ({ev['dates'][0]} and {ev['dates'][1]}). "
                             f"If you ordered once, ask for a refund of {money(ev['amount'])}.",
                             ev["amount"], "Same merchant and exact same amount within a few days.", f["id"], ev, "high"))
        elif t == "category_spike":
            cap = ev["baseline_3mo_avg"] * 1.1
            recs.append(_rec("reduce_spending", f"{f['category']} is up {ev['pct']:.0f}% this month",
                             f"{f['category']} hit {money(ev['current'])} vs your usual {money(ev['baseline_3mo_avg'])}. "
                             f"Cap it at about {money(cap)} next month to recover ~{money(ev['current'] - cap)}.",
                             ev["current"] - cap,
                             f"Latest month exceeds the 3-month average by {money(ev['increase'])} and your normal month-to-month swing.",
                             f["id"], ev, sev))
            lever_monthly += max(ev["current"] - cap, 0)
        elif t == "txn_outlier":
            recs.append(_rec("alert", f"Unusually large {f['category']} charge",
                             f"{ev['merchant']} on {f['date']} was {money(ev['amount'])} vs a typical {money(ev['typical'])}. "
                             f"If it was a one-off, no action — it will not be treated as your normal spending.",
                             0, f"Robust z-score {ev['robust_z']} within {f['category']}.", f["id"], ev, "low"))
        elif t == "savings_rate_drop":
            recs.append(_rec("alert", "Your savings rate dropped",
                             f"You saved {ev['current_rate']*100:.0f}% of income last month vs {ev['baseline_rate']*100:.0f}% before.",
                             0, "Savings rate fell compared with the previous 3-month average.", f["id"], ev, sev))

    if fc and "error" not in fc:
        fcf, buf = fc["free_cash_flow"], fc["fixed_obligations"]
        top = sorted(((c, v) for c, v in fc["by_category"].items() if c in DISCRETIONARY), key=lambda kv: -kv[1])[:2]
        if fcf < 0:
            recs.append(_rec("cashflow_warning", "You are on track to spend more than you earn",
                             f"Projected spending exceeds income by {money(-fcf)}/month. Trim "
                             + " and ".join(f"{c} ({money(v)}/mo)" for c, v in top) + f" by about {money(-fcf)} combined.",
                             -fcf, f"Forecast free cash flow is {money(fcf)}/month.", "forecast:negative_free_cash_flow",
                             {"free_cash_flow": fcf, "top_discretionary": dict(top)}, "high"))
        low = min(fc["trajectory"], key=lambda r: r["balance_low"])
        if low["balance_low"] < buf:
            recs.append(_rec("cashflow_warning", "Cash buffer may run thin",
                             f"In a high-spend case your balance could fall to {money(low['balance_low'])} by {low['month']}, "
                             f"below one month of fixed bills ({money(buf)}). Keep discretionary spending steady until then.",
                             0, "Pessimistic end of the 80% forecast band dips under the safety buffer.",
                             "forecast:low_buffer", {"month": low["month"], "balance_low": low["balance_low"], "buffer": buf}, "medium"))

    for gs in goal_statuses or []:
        if "error" in gs:
            continue
        if gs["schedule_status"] != "behind" and gs["forecast_on_track"]:
            continue
        if gs["forecast_on_track"]:
            # Behind schedule, but forecast cash flow is enough: the fix is behavioural (move the money).
            eta = gs["projected_completion_month"] or "the deadline"
            recs.append(_rec("goal_behind_schedule", f"'{gs['goal']}' is behind schedule",
                             f"You have {money(gs['saved'])} of {money(gs['target'])}, {money(-gs['gap_vs_schedule'])} behind a steady plan. "
                             f"Your forecast frees {money(gs['projected_per_month'])}/month, which is enough: set an automatic transfer of "
                             f"{money(gs['required_per_month'])}/month into this goal to finish by {eta}.",
                             gs["required_per_month"], f"Saved amount is below the straight-line schedule; forecast free cash flow "
                             f"({money(gs['projected_per_month'])}) covers the required {money(gs['required_per_month'])}/month.",
                             f"goal:{gs['goal']}", {**gs, "lever_monthly": round(lever_monthly)}, "medium"))
        else:
            extra = gs["extra_to_save_per_month"]
            cover = min(lever_monthly / extra * 100, 100) if extra else 100
            recs.append(_rec("goal_off_track", f"'{gs['goal']}' will be missed at the current pace",
                             f"You have {money(gs['saved'])} of {money(gs['target'])}; you need {money(gs['required_per_month'])}/month but the "
                             f"forecast frees only {money(gs['projected_per_month'])}. Find {money(extra)}/month more"
                             + (f" — the other actions here could cover about {cover:.0f}% of that." if lever_monthly else "."),
                             extra, f"Forecast free cash flow ({money(gs['projected_per_month'])}/month) is below the "
                             f"required pace ({money(gs['required_per_month'])}/month).", f"goal:{gs['goal']}",
                             {**gs, "lever_monthly": round(lever_monthly)}, "high"))
    recs.sort(key=lambda r: (SEV[r["severity"]], -r["impact_amount"]))
    return recs[:max_recs]


# ------------------------------------------------------------------------------ LLM phrasing
_NUM = re.compile(r"\d[\d,]*\.?\d*")


def numbers_grounded(text: str, facts: str) -> bool:
    """Every number >12 in `text` must appear in `facts` (commas stripped)."""
    allowed = {n.replace(",", "").rstrip(".") for n in _NUM.findall(facts)}
    for n in _NUM.findall(text):
        v = n.replace(",", "").rstrip(".")
        if v and float(v) > 12 and v not in allowed:
            return False
    return True


def polish(recs: list[dict]) -> list[dict]:
    """Optional Mesh rewrite of rec['message'] into one friendly sentence pair. Safe fallback."""
    from app import llm
    if not recs or not llm.available():
        return recs
    system = ("You are a concise personal-finance coach. Rewrite each recommendation in 1-2 friendly sentences. "
              "Use ONLY the facts given; never add or change numbers or merchants. "
              'Return JSON {"messages": {"<idx>": "<text>"}}.')
    facts = json.dumps([{"idx": i, "title": r["title"], "action": r["action"], "reason": r["reason"]}
                        for i, r in enumerate(recs)], default=str)
    try:
        out = llm.chat_json(system, facts, max_tokens=900).get("messages", {})
    except Exception:
        return recs
    for i, r in enumerate(recs):
        msg = out.get(str(i))
        if msg and numbers_grounded(msg, json.dumps(r, default=str)):
            r["message"] = msg
    return recs


def compose_digest(recs: list[dict], fc: Optional[dict], profile: dict, goal_statuses: list[dict]) -> str:
    """Scheduled digest: ONE key insight + ONE action. Deterministic skeleton, optional LLM polish."""
    lines = ["FinPilot weekly digest"]
    if fc and "error" not in fc:
        end = fc["trajectory"][-1]
        lines.append(f"Outlook: at your current pace your balance reaches {money(end['balance_base'])} by {end['month']} "
                     f"(likely range {money(end['balance_low'])}–{money(end['balance_high'])}); "
                     f"free cash flow is {money(fc['free_cash_flow'])}/month.")
    if recs:
        top = recs[0]
        lines.append(f"Key insight: {top['title']}. Why: {top['reason']}")
        lines.append(f"Do this now: {top['action']}")
    else:
        lines.append("No issues found this week — nothing to change.")
    for gs in goal_statuses[:2]:
        if "error" not in gs:
            lines.append(f"Goal '{gs['goal']}': {gs['schedule_status'].replace('_', ' ')}; saved {money(gs['saved'])} of {money(gs['target'])}.")
    text = "\n".join(lines)
    from app import llm
    if llm.available():
        try:
            new = llm.chat("Rewrite this finance digest as a short, warm message (max 90 words). Keep every number and "
                           "fact exactly; add none. Keep the 'Key insight' and 'Do this now' parts.", text, max_tokens=300)
            if new and numbers_grounded(new, text):
                return new.strip()
        except Exception:
            pass
    return text
