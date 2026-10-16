"""Shared helpers for the deterministic agents."""
from __future__ import annotations

import pandas as pd

FIXED = ["Rent", "Utilities", "Subscriptions"]          # obligations that recur every month
NON_EXPENSE = ["Income", "Savings"]                      # money in / money moved to savings


def expenses(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["amount"] < 0) & (~df["category"].isin(NON_EXPENSE))].copy()


def month_key(s: pd.Series) -> pd.Series:
    return s.dt.to_period("M").astype(str)


def complete_months(df: pd.DataFrame) -> list[str]:
    """All months in the data, dropping the last one if the data stops well before month end."""
    if df.empty:
        return []
    months = sorted(month_key(df["date"]).unique())
    last_date = df["date"].max()
    if last_date.day < last_date.days_in_month - 4:
        months = months[:-1]
    full = pd.period_range(months[0], months[-1], freq="M").astype(str).tolist() if months else []
    return full


def monthly_matrix(df: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """rows = expense categories, cols = months, values = positive spend."""
    e = expenses(df)
    e["month"] = month_key(e["date"])
    e = e[e["month"].isin(months)]
    m = e.pivot_table(index="category", columns="month", values="amount", aggfunc=lambda x: -x.sum(), fill_value=0.0)
    return m.reindex(columns=months, fill_value=0.0)


def monthly_series(df: pd.DataFrame, months: list[str], kind: str) -> pd.Series:
    """kind: income | savings | expense (positive numbers)."""
    d = df.copy()
    d["month"] = month_key(d["date"])
    d = d[d["month"].isin(months)]
    if kind == "income":
        s = d[d["amount"] > 0].groupby("month")["amount"].sum()
    elif kind == "savings":
        s = -d[(d["amount"] < 0) & (d["category"] == "Savings")].groupby("month")["amount"].sum()
    else:
        e = expenses(d)
        s = -e.groupby("month")["amount"].sum()
    return s.reindex(months, fill_value=0.0).astype(float)
