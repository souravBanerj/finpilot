"""AGENT 3 — Forecaster + what-if simulator. Deterministic (so its accuracy is measurable).

Model (deliberately simple, transparent, and backtested — see eval/results.md):
  * Rent, Subscriptions     -> last month's value (they step, they don't drift)
  * Utilities               -> mean of last 3 months (seasonal wobble)
  * all other expense cats  -> EWMA (alpha=0.4) of the monthly totals
  * Income                  -> median of last 3 months (robust to a one-off bonus)
  * Uncertainty             -> std of in-sample one-step EWMA errors per category, combined as
                               sqrt(sum var); shown as an 80% band (±1.28 sigma), growing with sqrt(h)
Simulation: inject a hypothetical purchase / savings goal into the forecast balance path and report
the minimum balance, buffer breaches and goal-date shift.
Known limits (honest): no seasonality, no explicit one-off shocks, categories assumed independent.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from agents.common import FIXED, complete_months, monthly_matrix, monthly_series

ALPHA = 0.4
Z80 = 1.2816


def _ewma_level(x: list[float], alpha: float = ALPHA) -> tuple[float, list[float]]:
    level, resid = x[0], []
    for v in x[1:]:
        resid.append(v - level)
        level = alpha * v + (1 - alpha) * level
    return level, resid


def predict_next(series: list[float], category: str) -> tuple[float, float]:
    """(point forecast, sigma) for the next month of one category."""
    x = [float(v) for v in series]
    if not x:
        return 0.0, 0.0
    if category in ("Rent", "Subscriptions"):
        point = x[-1]
    elif category == "Utilities":
        point = float(np.mean(x[-3:]))
    else:
        point, _ = _ewma_level(x)
    _, resid = _ewma_level(x)
    sigma = float(np.std(resid, ddof=1)) if len(resid) >= 3 else 0.15 * point
    if category in ("Rent",):
        sigma = 0.0
    return max(point, 0.0), sigma


def forecast_next_month(mat: pd.DataFrame, income: pd.Series) -> dict:
    by_cat, var = {}, 0.0
    for cat in mat.index:
        p, s = predict_next(mat.loc[cat].tolist(), cat)
        by_cat[cat] = p
        var += s ** 2
    inc = float(np.median(income.iloc[-3:])) if len(income) else 0.0
    inc_sigma = float(income.iloc[-6:].std(ddof=1)) if len(income) >= 3 else 0.0
    var += (0.0 if math.isnan(inc_sigma) else inc_sigma) ** 2
    return {"by_category": by_cat, "expenses": sum(by_cat.values()), "income": inc, "sigma": math.sqrt(var)}


def forecast(df: pd.DataFrame, horizon: int = 6, opening_balance: float = 0.0,
             include_savings_contrib: bool = True) -> dict:
    """Project the next `horizon` months and the running balance (base / low / high)."""
    months = complete_months(df)
    if len(months) < 3:
        return {"error": "Need at least 3 complete months of data to forecast.", "months": months}
    mat = monthly_matrix(df, months)
    income = monthly_series(df, months, "income")
    sav = monthly_series(df, months, "savings")
    f1 = forecast_next_month(mat, income)
    sav_next = float(sav.iloc[-1]) if include_savings_contrib else 0.0

    d = df.copy()
    d["m"] = d["date"].dt.to_period("M").astype(str)
    start_balance = float(opening_balance) + float(d[d["m"].isin(months)]["amount"].sum())

    nxt = pd.Period(months[-1], "M")
    rows, bal = [], start_balance
    for h in range(1, horizon + 1):
        m = str(nxt + h)
        net = f1["income"] - f1["expenses"] - sav_next
        bal += net
        sd = f1["sigma"] * math.sqrt(h)                       # uncertainty compounds
        rows.append({"month": m, "income": round(f1["income"]), "expenses": round(f1["expenses"]),
                     "savings_contrib": round(sav_next), "net": round(net),
                     "balance_base": round(bal), "balance_low": round(bal - Z80 * sd),
                     "balance_high": round(bal + Z80 * sd)})
    fixed = sum(f1["by_category"].get(c, 0) for c in FIXED)
    return {"as_of_month": months[-1], "start_balance": round(start_balance), "horizon": horizon,
            "monthly_income": round(f1["income"]), "monthly_expenses": round(f1["expenses"]),
            "monthly_savings_contrib": round(sav_next), "free_cash_flow": round(f1["income"] - f1["expenses"] - sav_next),
            "fixed_obligations": round(fixed), "by_category": {k: round(v) for k, v in f1["by_category"].items()},
            "sigma_monthly": round(f1["sigma"]), "trajectory": rows,
            "assumptions": "Current behaviour continues; no seasonality; 80% band from historical volatility."}


# ----------------------------------------------------------------------------- simulation
def simulate_purchase(fc: dict, amount: float, month_offset: int = 1, name: str = "purchase",
                      buffer_months: float = 1.0) -> dict:
    """month_offset: 1 = first forecast month. Buffer = `buffer_months` of fixed obligations."""
    if "error" in fc:
        return fc
    traj = fc["trajectory"]
    buffer = fc["fixed_obligations"] * buffer_months
    base, low = [], []
    for i, r in enumerate(traj, start=1):
        hit = amount if i >= month_offset else 0
        base.append(r["balance_base"] - hit)
        low.append(r["balance_low"] - hit)
    min_base, min_low = min(base), min(low)
    before_min = min(r["balance_base"] for r in traj)
    if min_base < 0:
        verdict, why = "no", "Your projected balance goes negative."
    elif min_base < buffer:
        verdict, why = "no", f"Your balance would drop below a one-month safety buffer ({buffer:,.0f})."
    elif min_low < buffer:
        verdict, why = "tight", "Affordable on average, but in a high-spend month you would dip below your safety buffer."
    else:
        verdict, why = "yes", "Your balance stays above the safety buffer even in the pessimistic case."
    months_after = max(len(traj) - month_offset + 1, 1)
    cut = max(buffer - min_base, 0) / months_after if verdict == "no" else (
        max(buffer - min_low, 0) / months_after if verdict == "tight" else 0.0)
    return {"name": name, "amount": amount, "month": traj[min(month_offset, len(traj)) - 1]["month"],
            "verdict": verdict, "explanation": why, "safety_buffer": round(buffer),
            "min_balance_before": round(before_min), "min_balance_after_base": round(min_base),
            "min_balance_after_low": round(min_low), "balance_after": [round(b) for b in base],
            "monthly_cut_to_make_it_work": round(cut)}


def simulate_goal(fc: dict, target: float, deadline: str, saved: float = 0.0, extra_spend: float = 0.0) -> dict:
    if "error" in fc:
        return fc
    first = pd.Period(fc["trajectory"][0]["month"], "M")
    dl = pd.Period(pd.Timestamp(deadline), "M")
    months_left = max((dl - first).n + 1, 1)
    remaining = max(target - saved + extra_spend, 0)
    required = remaining / months_left
    fcf = fc["free_cash_flow"]
    projected = saved + fcf * months_left - extra_spend
    on_track = fcf >= required
    eta = None
    if fcf > 0:
        n = math.ceil(remaining / fcf)
        eta = str(first + max(n - 1, 0))
    return {"target": target, "saved": saved, "deadline": deadline, "months_left": months_left,
            "required_per_month": round(required), "projected_per_month": fcf,
            "projected_total_at_deadline": round(projected), "on_track": bool(on_track),
            "shortfall_at_deadline": round(max(target - projected, 0)),
            "extra_to_save_per_month": round(max(required - fcf, 0)), "projected_completion_month": eta}


def goal_status(goal: dict, fc: dict, today: Optional[pd.Timestamp] = None) -> dict:
    """Goal tracking: compare saved-so-far with a straight-line schedule AND the forecast."""
    today = today or pd.Timestamp.today().normalize()
    start = pd.Timestamp(goal.get("start_date") or today)
    end = pd.Timestamp(goal["deadline"])
    total_days = max((end - start).days, 1)
    elapsed = min(max((today - start).days / total_days, 0), 1)
    expected = goal["target_amount"] * elapsed
    saved = goal["saved_so_far"]
    sim = simulate_goal(fc, goal["target_amount"], goal["deadline"], saved)
    if "error" in sim:
        return sim
    gap = saved - expected
    status = "ahead" if gap >= 0.05 * goal["target_amount"] else (
        "behind" if gap <= -0.05 * goal["target_amount"] else "on_track")
    return {"goal": goal["name"], "target": goal["target_amount"], "saved": saved,
            "expected_by_now": round(expected), "schedule_status": status, "gap_vs_schedule": round(gap),
            "forecast_on_track": sim["on_track"], "required_per_month": sim["required_per_month"],
            "projected_per_month": sim["projected_per_month"],
            "extra_to_save_per_month": sim["extra_to_save_per_month"],
            "projected_completion_month": sim["projected_completion_month"]}


# ----------------------------------------------------------------------------- self-evaluation
def backtest(df: pd.DataFrame, min_train: int = 4) -> dict:
    """Rolling-origin backtest: train on months[:t], predict month t, compare to actual.
    Baselines for TOTAL monthly expenses: naive_last and mean3."""
    months = complete_months(df)
    mat_all = monthly_matrix(df, months)
    inc_all = monthly_series(df, months, "income")
    tot_all = mat_all.sum(axis=0)
    rec = []
    cat_err: dict[str, list] = {c: [] for c in mat_all.index}
    cat_naive: dict[str, list] = {c: [] for c in mat_all.index}
    for t in range(min_train, len(months)):
        mat = mat_all.iloc[:, :t]
        f = forecast_next_month(mat, inc_all.iloc[:t])
        actual = float(tot_all.iloc[t])
        naive = float(tot_all.iloc[t - 1])
        mean3 = float(tot_all.iloc[max(t - 3, 0):t].mean())
        lo, hi = f["expenses"] - Z80 * f["sigma"], f["expenses"] + Z80 * f["sigma"]
        rec.append({"month": months[t], "actual": actual, "model": f["expenses"], "naive_last": naive,
                    "mean3": mean3, "in_band": lo <= actual <= hi,
                    "income_actual": float(inc_all.iloc[t]), "income_pred": f["income"]})
        for c in mat_all.index:
            a = float(mat_all.loc[c].iloc[t])
            if a > 0:
                cat_err[c].append(abs(f["by_category"][c] - a) / a)
                cat_naive[c].append(abs(float(mat_all.loc[c].iloc[t - 1]) - a) / a)
    r = pd.DataFrame(rec)
    if r.empty:
        return {"error": "not enough months for a backtest"}
    mape = lambda p, a: float((abs(p - a) / a).mean() * 100)
    return {
        "folds": len(r),
        "total_expense_mape_model": round(mape(r["model"], r["actual"]), 2),
        "total_expense_mape_naive_last": round(mape(r["naive_last"], r["actual"]), 2),
        "total_expense_mape_mean3": round(mape(r["mean3"], r["actual"]), 2),
        "income_mape": round(mape(r["income_pred"], r["income_actual"]), 2),
        "interval_80_coverage": round(float(r["in_band"].mean()) * 100, 1),
        "per_category_mape_model": {c: round(float(np.mean(v)) * 100, 1) for c, v in cat_err.items() if v},
        "per_category_mape_naive": {c: round(float(np.mean(v)) * 100, 1) for c, v in cat_naive.items() if v},
        "rows": r.round(0).to_dict(orient="records"),
    }
