"""COORDINATOR — routes work to the specialised agents and assembles the result.

Flow (LangGraph StateGraph when installed, otherwise an equivalent sequential runner):

    categorize ─▶ analyze ─▶ forecast ─▶ track_goals ─▶ advise ─▶ END

Each node is one agent with its own job; every step appends to `trace` so you can see who did what.
Other tasks are routed by `handle()`: pipeline | simulate | chat | digest.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

import pandas as pd

from app import db
from agents import advisor, analyzer, categorizer, forecaster


class State(TypedDict, total=False):
    user_id: str
    use_llm: bool
    persist: bool
    df: pd.DataFrame
    analysis: dict
    forecast: dict
    goals: list
    recs: list
    trace: list
    empty: bool


def _log(state: State, agent: str, t0: float, summary: str) -> list:
    return state.get("trace", []) + [{"agent": agent, "ms": round((time.time() - t0) * 1000), "summary": summary}]


# ------------------------------------------------------------------------------ nodes (= agents)
def categorize_node(state: State) -> State:
    t0, uid = time.time(), state["user_id"]
    df = db.load_transactions(uid)
    if df.empty:
        return {"empty": True, "df": df, "trace": _log(state, "categorizer", t0, "no transactions")}
    out = categorizer.categorize(df, memory=db.get_memory(uid), use_llm=state.get("use_llm", True))
    if state.get("persist", True):
        db.update_categories(out, uid)
        for r in out[(out["method"] == "llm") & (out["confidence"] >= 0.7)].drop_duplicates("merchant").itertuples():
            db.remember(r.merchant, r.category, uid)            # LLM answers are cached → fewer calls next time
    methods = out["method"].value_counts().to_dict()
    return {"df": out, "empty": False, "trace": _log(state, "categorizer", t0, f"{len(out)} txns, methods={methods}")}


def analyze_node(state: State) -> State:
    t0 = time.time()
    if state.get("empty"):
        return {"analysis": {"findings": [], "profile": {}}, "trace": state.get("trace", [])}
    a = analyzer.analyze(state["df"], usage=db.get_setting("subscription_usage", {}, state["user_id"]))
    return {"analysis": a, "trace": _log(state, "analyzer", t0, f"{len(a['findings'])} findings")}


def forecast_node(state: State) -> State:
    t0 = time.time()
    if state.get("empty"):
        return {"forecast": {"error": "No data yet."}, "trace": state.get("trace", [])}
    fc = forecaster.forecast(state["df"], horizon=6,
                             opening_balance=db.get_setting("opening_balance", 0.0, state["user_id"]))
    msg = fc.get("error") or f"FCF {fc['free_cash_flow']}/mo, 6-mo horizon"
    return {"forecast": fc, "trace": _log(state, "forecaster", t0, msg)}


def goals_node(state: State) -> State:
    t0 = time.time()
    fc = state["forecast"]
    gs = [forecaster.goal_status(g, fc) for g in db.list_goals(state["user_id"])] if "error" not in fc else []
    return {"goals": gs, "trace": _log(state, "goal_tracker", t0, f"{len(gs)} goals")}


def advise_node(state: State) -> State:
    t0, uid = time.time(), state["user_id"]
    recs = advisor.advise(state["analysis"], state["forecast"], state.get("goals", []))
    if state.get("use_llm", True):
        recs = advisor.polish(recs)
    if state.get("persist", True) and recs:
        seen = {(d["trigger_ref"], d["action"]) for d in db.list_decisions(uid, 500)}
        db.save_decisions([r for r in recs if (r["trigger_ref"], r["action"]) not in seen], uid)
    return {"recs": recs, "trace": _log(state, "advisor", t0, f"{len(recs)} recommendations")}


NODES = [("categorize", categorize_node), ("analyze", analyze_node), ("forecast", forecast_node),
         ("track_goals", goals_node), ("advise", advise_node)]


def build_graph():
    """Explicit LangGraph flow. Returns None if langgraph is not installed."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None
    g = StateGraph(State)
    for name, fn in NODES:
        g.add_node(name, fn)
    g.set_entry_point(NODES[0][0])
    for (a, _), (b, _) in zip(NODES, NODES[1:]):
        g.add_edge(a, b)
    g.add_edge(NODES[-1][0], END)
    return g.compile()


_GRAPH = None


def run_pipeline(user_id: str = "demo", use_llm: bool = True, persist: bool = True) -> State:
    global _GRAPH
    state: State = {"user_id": user_id, "use_llm": use_llm, "persist": persist, "trace": []}
    if _GRAPH is None:
        _GRAPH = build_graph() or False
    if _GRAPH:
        return _GRAPH.invoke(state)
    for _, fn in NODES:                       # sequential fallback with identical semantics
        state.update(fn(state))
    return state


# ------------------------------------------------------------------------------ task routing
def simulate(user_id: str, kind: str, **p) -> dict:
    state = run_pipeline(user_id, use_llm=False, persist=False)
    fc = state.get("forecast", {})
    if "error" in fc:
        return fc
    if kind == "purchase":
        res = forecaster.simulate_purchase(fc, p["amount"], p.get("month_offset", 1), p.get("name", "purchase"))
        goals = [forecaster.simulate_goal(fc, g["target_amount"], g["deadline"], g["saved_so_far"], extra_spend=p["amount"])
                 for g in db.list_goals(user_id)]
        res["goal_impact"] = [{"goal": g["name"], "on_track_after": s["on_track"], "extra_needed_per_month": s["extra_to_save_per_month"]}
                              for g, s in zip(db.list_goals(user_id), goals) if "error" not in s]
        title, trig = f"Simulated purchase: {res['name']} ({res['amount']:,.0f})", "simulation:purchase"
        reason = f"User asked 'can I afford it?'. Verdict {res['verdict']}: {res['explanation']}"
    elif kind == "goal":
        res = forecaster.simulate_goal(fc, p["target"], p["deadline"], p.get("saved", 0.0))
        title, trig = f"Simulated goal {p['target']:,.0f} by {p['deadline']}", "simulation:goal"
        reason = f"Required {res['required_per_month']}/mo vs forecast {res['projected_per_month']}/mo."
    else:
        raise ValueError("kind must be 'purchase' or 'goal'")
    db.save_decisions([{"rec_type": "simulation", "title": title, "action": str(res.get("explanation", reason)),
                        "impact_amount": p.get("amount", p.get("target", 0)), "reason": reason, "trigger_ref": trig,
                        "evidence": res, "agent": "forecaster"}], user_id)
    return res


def handle(task: str, user_id: str = "demo", **kw) -> Any:
    if task == "pipeline":
        return run_pipeline(user_id, **kw)
    if task == "simulate":
        return simulate(user_id, **kw)
    if task == "chat":
        from agents import chat
        state = run_pipeline(user_id, use_llm=False, persist=False)
        return chat.answer(kw["question"], state)
    if task == "digest":
        from app import digest
        return digest.build_and_send(user_id)
    raise ValueError(f"unknown task {task}")
