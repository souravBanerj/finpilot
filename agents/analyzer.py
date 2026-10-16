"""AGENT 2 — Analyzer (behaviour + anomaly detection). Deterministic pandas/numpy.

Job: study the user's OWN history and emit machine-readable `findings`. Every finding has a
stable `id` (the advisor cites it as its trigger) and an `evidence` dict with the numbers.
The LLM never invents findings; it only phrases them later.

Detectors
  category_spike        latest complete month vs mean of previous 3 months (+ pct and abs thresholds)
  txn_outlier           robust z-score (median/MAD on log-amount) within a category
  duplicate_charge      same merchant + same amount within N days
  subscription_overlap  several active subscriptions in the same service group (streaming, music...)
  subscription_unused   (needs optional usage data) active subscription with no recent use
  subscription_price_hike  latest recurring charge >= 8% above the previous one
  savings_rate_drop     latest savings rate well below trailing average
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app import config
from agents.common import (FIXED, complete_months, expenses, monthly_matrix, monthly_series, month_key)

SUB_GROUPS = {
    "video streaming": ["netflix", "hotstar", "prime video", "amazon prime", "disney", "hbo", "sonyliv",
                        "zee5", "crunchyroll", "youtube premium"],
    "music": ["spotify", "apple music", "jiosaavn", "gaana", "audible"],
    "cloud storage": ["icloud", "dropbox", "google one"],
    "fitness": ["gym", "cult"],
}
NO_OUTLIER_CATS = set(FIXED) | {"Income", "Savings", "Other"}


def _group_of(merchant: str) -> str | None:
    m = merchant.lower()
    for g, keys in SUB_GROUPS.items():
        if any(k in m for k in keys):
            return g
    return None


def active_subscriptions(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    s = df[(df["amount"] < 0) & ((df["category"] == "Subscriptions") | (df["is_recurring"] == 1))]
    s = s[s["category"] == "Subscriptions"]
    rows = []
    for m, g in s.groupby("merchant"):
        g = g.sort_values("date")
        if len(g) < 2:
            continue
        last = g.iloc[-1]
        if (as_of - last["date"]).days > 45:
            continue                                  # cancelled / no longer billing
        prev = g.iloc[-2]
        rows.append({"merchant": m, "monthly_cost": abs(last["amount"]), "prev_cost": abs(prev["amount"]),
                     "last_charge": last["date"], "months_seen": len(g), "group": _group_of(m)})
    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame, usage: dict | None = None) -> dict:
    """df must be categorized. usage = optional {merchant: 'YYYY-MM-DD' last used}."""
    findings: list[dict] = []
    if df.empty:
        return {"months": [], "findings": [], "profile": {}}
    months = complete_months(df)
    as_of = df["date"].max()
    mat = monthly_matrix(df, months)
    income = monthly_series(df, months, "income")
    exp_tot = monthly_series(df, months, "expense")
    sav = monthly_series(df, months, "savings")

    # ---- 1. category spikes ------------------------------------------------------------
    if len(months) >= 4:
        latest = months[-1]
        for cat in mat.index:
            if cat in ("Rent", "Other"):          # Other = unexplained junk drawer, not actionable
                continue
            hist = mat.loc[cat, months[-7:-1]]    # up to 6 prior months
            base = float(hist.iloc[-3:].mean())
            sd = float(hist.std(ddof=1)) if len(hist) >= 3 else 0.0
            cur = float(mat.loc[cat, latest])
            if base <= 0:
                continue
            pct = (cur - base) / base * 100
            # must beat the % threshold, the absolute floor AND this category's own normal volatility
            if pct >= config.MOM_INCREASE_PCT and (cur - base) >= config.MOM_MIN_ABS and cur > base + config.SPIKE_SIGMA * sd:
                findings.append({
                    "id": f"category_spike:{cat}:{latest}", "type": "category_spike",
                    "severity": "high" if pct >= 50 else "medium", "category": cat, "month": latest,
                    "title": f"{cat} spending is up {pct:.0f}% in {latest}",
                    "evidence": {"current": round(cur), "baseline_3mo_avg": round(base),
                                 "increase": round(cur - base), "pct": round(pct, 1),
                                 "normal_monthly_std": round(sd)}})

    # ---- 2. single-transaction outliers (robust z on log amount) -----------------------
    e = expenses(df)
    for cat, g in e.groupby("category"):
        if cat in NO_OUTLIER_CATS or len(g) < 8:
            continue
        la = np.log(g["amount"].abs().clip(lower=1))
        med = la.median()
        mad = (la - med).abs().median()
        if mad < 1e-6:
            continue
        z = 0.6745 * (la - med) / mad
        for idx in g.index[(z > config.ANOMALY_MAD_Z).values]:
            r = df.loc[idx]
            typical = float(np.exp(med))
            findings.append({
                "id": f"txn_outlier:{int(r['id'])}", "type": "txn_outlier", "severity": "medium",
                "category": cat, "txn_id": int(r["id"]), "date": str(r["date"].date()),
                "title": f"Unusually large {cat} charge: {r['merchant']} on {r['date'].date()}",
                "evidence": {"amount": round(abs(r["amount"])), "typical": round(typical),
                             "robust_z": round(float(z.loc[idx]), 1), "merchant": r["merchant"]}})

    # ---- 3. duplicate charges ----------------------------------------------------------
    for (m, a), g in e.groupby(["merchant", "amount"]):
        if len(g) < 2 or abs(a) < 200:
            continue
        g = g.sort_values(["date", "id"])
        for (_, a1), (_, a2) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            if (a2["date"] - a1["date"]).days <= config.DUPLICATE_WINDOW_DAYS and not a1["is_recurring"]:
                findings.append({
                    "id": f"duplicate_charge:{int(a2['id'])}", "type": "duplicate_charge", "severity": "high",
                    "category": a2["category"], "txn_id": int(a2["id"]), "date": str(a2["date"].date()),
                    "title": f"Possible duplicate charge: {m} {abs(a):.0f} twice within "
                             f"{(a2['date'] - a1['date']).days or 'under 1'} day(s)",
                    "evidence": {"merchant": m, "amount": round(abs(a)), "dates": [str(a1["date"].date()),
                                 str(a2["date"].date())], "txn_ids": [int(a1["id"]), int(a2["id"])]}})

    # ---- 4. subscriptions --------------------------------------------------------------
    subs = active_subscriptions(df, as_of)
    sub_total = float(subs["monthly_cost"].sum()) if len(subs) else 0.0
    if len(subs):
        for grp, g in subs.dropna(subset=["group"]).groupby("group"):
            if len(g) >= 2:
                keep_cost = float(g["monthly_cost"].max())
                findings.append({
                    "id": f"subscription_overlap:{grp}", "type": "subscription_overlap", "severity": "medium",
                    "category": "Subscriptions", "group": grp,
                    "title": f"{len(g)} overlapping {grp} subscriptions",
                    "evidence": {"merchants": g["merchant"].tolist(),
                                 "costs": {r.merchant: round(r.monthly_cost) for r in g.itertuples()},
                                 "monthly_total": round(float(g["monthly_cost"].sum())),
                                 "saving_if_keep_one": round(float(g["monthly_cost"].sum()) - keep_cost)}})
        for r in subs.itertuples():
            h = df[(df["merchant"] == r.merchant) & (df["category"] == "Subscriptions") & (df["amount"] < 0)]
            amts = h.sort_values("date")["amount"].abs().tolist()[-6:]
            dts = h.sort_values("date")["date"].tolist()[-6:]
            for i in range(len(amts) - 1, 0, -1):          # most recent change point in last 6 charges
                if amts[i] >= amts[i - 1] * 1.08:
                    findings.append({
                        "id": f"subscription_price_hike:{r.merchant}", "type": "subscription_price_hike",
                        "severity": "low", "category": "Subscriptions", "merchant": r.merchant,
                        "title": f"{r.merchant} raised its price from {amts[i-1]:.0f} to {amts[i]:.0f}",
                        "evidence": {"old": round(amts[i - 1]), "new": round(amts[i]), "since": str(dts[i].date()),
                                     "extra_per_year": round((amts[i] - amts[i - 1]) * 12)}})
                    break
            if usage and r.merchant in usage:
                days = (as_of - pd.Timestamp(usage[r.merchant])).days
                if days > 45:
                    findings.append({
                        "id": f"subscription_unused:{r.merchant}", "type": "subscription_unused",
                        "severity": "high", "category": "Subscriptions", "merchant": r.merchant,
                        "title": f"{r.merchant} not used for {days} days but still billing",
                        "evidence": {"days_unused": days, "monthly_cost": round(r.monthly_cost),
                                     "saving_per_year": round(r.monthly_cost * 12)}})

    # ---- 5. savings-rate drop ----------------------------------------------------------
    rate = ((income - exp_tot) / income.replace(0, np.nan)).fillna(0)
    if len(months) >= 4 and rate.iloc[-4:-1].mean() - rate.iloc[-1] >= 0.05:
        findings.append({
            "id": f"savings_rate_drop:{months[-1]}", "type": "savings_rate_drop", "severity": "high",
            "category": None, "month": months[-1],
            "title": f"Savings rate fell to {rate.iloc[-1]*100:.0f}% from a {rate.iloc[-4:-1].mean()*100:.0f}% average",
            "evidence": {"current_rate": round(float(rate.iloc[-1]), 3),
                         "baseline_rate": round(float(rate.iloc[-4:-1].mean()), 3)}})

    # ---- behaviour profile ---------------------------------------------------------------
    disc = ["Food Delivery", "Dining", "Shopping", "Entertainment", "Travel"]
    tot3 = float(exp_tot.iloc[-3:].sum()) or 1.0
    disc3 = float(mat.reindex(disc).fillna(0).loc[:, months[-3:]].sum().sum()) if len(months) >= 3 else 0
    wk = e[e["category"].isin(["Food Delivery", "Dining"])]
    profile = {
        "months_analyzed": len(months), "as_of": str(as_of.date()), "latest_month": months[-1],
        "avg_monthly_income": round(float(income.mean())), "avg_monthly_expenses": round(float(exp_tot.mean())),
        "latest_savings_rate": round(float(rate.iloc[-1]), 3),
        "discretionary_share_last3": round(disc3 / tot3, 3),
        "weekend_share_of_eating_out": round(float((wk["date"].dt.dayofweek >= 5).mean()), 3) if len(wk) else 0,
        "subscription_monthly_total": round(sub_total), "active_subscriptions": len(subs),
        "top_categories_last3": {c: round(float(v)) for c, v in
                                 mat.loc[:, months[-3:]].sum(axis=1).sort_values(ascending=False).head(5).items()},
    }
    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order[f["severity"]])
    return {"months": months, "findings": findings, "profile": profile,
            "monthly_by_category": mat.round(0).to_dict(orient="index"),
            "income_by_month": income.round(0).to_dict(), "expenses_by_month": exp_tot.round(0).to_dict()}
