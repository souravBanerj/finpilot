"""AGENT 1 — Categorizer.

Job: turn raw bank descriptions into (merchant, category, confidence, method).
Output depends on the transaction through four layers, tried in order:

  1. memory      merchant→category learned from earlier data / user corrections
  2. rule        ordered keyword rules (specific before generic: "AMAZON PRIME" -> Subscriptions
                 beats "AMAZON" -> Shopping)
  3. recurrence  an unknown merchant that bills every ~month at a steady amount is almost surely
                 a subscription/bill, even if we have never seen its name
  4. llm         ONE batched Mesh call for the remaining unknown merchants (cached into memory)
  5. fallback    "Other" with low confidence

Every row records which layer decided (`method`) so accuracy can be measured per layer.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

CATEGORIES = [
    "Income", "Rent", "Groceries", "Food Delivery", "Dining", "Transport", "Subscriptions",
    "Utilities", "Shopping", "Entertainment", "Health", "Travel", "Savings", "Other",
]

# Order matters: first match wins. Specific patterns come before generic ones.
RULES: list[tuple[str, str]] = [
    ("Income", r"salary|payroll|stipend|freelance|interest credit|refund|cashback|dividend"),
    ("Rent", r"\brent\b|landlord|housing society|property mgmt"),
    ("Savings", r"\bsip\b|mutual fund|zerodha|groww|recurring deposit|fixed deposit|ppf|nps"),
    # Quick-commerce / marketplace grocery lines must beat the generic delivery & shopping rules
    ("Groceries", r"instamart|amazon fresh|swiggy instamart|bigbasket|blinkit|zepto"),
    ("Subscriptions", r"netflix|spotify|hotstar|youtube premium|disney|hbo|prime video|amazon prime|"
                      r"apple music|icloud|dropbox|adobe|audible|notion|chatgpt|openai|linkedin premium|"
                      r"cult\.?fit|gym|crunchyroll|sonyliv|zee5|jiosaavn|gaana|google one|microsoft 365"),
    ("Utilities", r"electricity|power bill|tata power|bescom|water bill|gas bill|broadband|airtel|"
                  r"\bjio\b|vodafone|\bvi\b|mobile recharge|internet bill|act fibernet"),
    ("Food Delivery", r"swiggy|zomato|ubereats|uber eats|doordash|grubhub|deliveroo|foodpanda|dominos|domino's"),
    ("Groceries", r"bigbasket|blinkit|zepto|dmart|d-mart|reliance fresh|grocery|supermarket|more retail|"
                  r"kroger|walmart grocery|spencer"),
    ("Dining", r"restaurant|cafe|café|starbucks|mcdonald|kfc|burger|bistro|diner|dhaba|coffee|chai|"
               r"barbeque|pizza hut|subway"),
    ("Transport", r"\buber\b|\bola\b|lyft|rapido|metro|fuel|petrol|diesel|shell|irctc|parking|fastag|toll|bmtc"),
    ("Travel", r"makemytrip|goibibo|airbnb|booking\.com|indigo|air india|vistara|airlines|hotel|"
               r"resort|expedia|cleartrip"),
    ("Entertainment", r"bookmyshow|pvr|inox|cinema|steam|playstation|xbox|concert|nintendo|event ticket"),
    ("Health", r"pharmacy|apollo|hospital|clinic|medplus|1mg|netmeds|dental|doctor|diagnostic|lab test"),
    ("Shopping", r"amazon|amzn|flipkart|myntra|ajio|ikea|zara|h&m|nykaa|decathlon|croma|meesho|"
                 r"shopping|mall|store"),
]
_COMPILED = [(c, re.compile(p, re.I)) for c, p in RULES]

_PREFIX = re.compile(
    r"^(upi|pos|ach|neft|imps|ecs|nach|debit card|dc|card|paypal \*|sq \*|tst\*|pur)\b[\s\-/*:]*", re.I)


def normalize_merchant(description: str) -> str:
    """'UPI-SWIGGY ORDER 88231@okaxis' -> 'SWIGGY ORDER'. Strips rails, refs, digits, punctuation."""
    s = str(description).upper().strip()
    for _ in range(2):
        s = _PREFIX.sub("", s)
    s = re.sub(r"\S*@\S*", " ", s)             # UPI handles
    s = re.sub(r"[#*/\-_.:,]+", " ", s)
    s = re.sub(r"\b[A-Z]*\d[A-Z\d]*\b", " ", s)   # reference codes / any token containing digits
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split() if t not in {"ORDER", "PAYMENT", "TXN", "PURCHASE", "BILL", "ONLINE", "INDIA"}]
    return " ".join(toks[:2]) if toks else s


def rule_category(description: str, amount: float) -> Optional[str]:
    text = str(description)
    if amount > 0:
        return "Income"           # money in is income regardless of keywords
    for cat, rx in _COMPILED:
        if cat == "Income":
            continue
        if rx.search(text):
            return cat
    return None


def detect_recurring(df: pd.DataFrame) -> pd.Series:
    """Per merchant: >=3 charges, median gap 25–35 days, amount CV < 0.15 (or <0.25 for bills)."""
    flag = pd.Series(False, index=df.index)
    exp = df[df["amount"] < 0]
    for _, g in exp.groupby("merchant"):
        if len(g) < 3:
            continue
        g = g.sort_values("date")
        gaps = g["date"].diff().dt.days.dropna()
        amt = g["amount"].abs()
        cv = amt.std(ddof=0) / amt.mean() if amt.mean() else 1
        if 25 <= gaps.median() <= 35 and cv < 0.25 and (gaps.between(20, 42).mean() >= 0.7):
            flag.loc[g.index] = True
    return flag


def _llm_categorize(unknown: dict[str, list[str]]) -> dict[str, tuple[str, float]]:
    """One batched Mesh call. `unknown` = {merchant: [example descriptions]}."""
    from app import llm
    if not unknown or not llm.available():
        return {}
    system = ("You categorize bank transactions. Allowed categories: "
              + ", ".join(c for c in CATEGORIES if c != "Income")
              + '. Return JSON: {"<merchant>": {"category": "<one allowed>", "confidence": 0-1}}. '
                "If truly unknown use Other with low confidence. Do not invent merchants.")
    user = "\n".join(f"- {m}: e.g. {', '.join(ex[:2])}" for m, ex in unknown.items())
    try:
        out = llm.chat_json(system, user, max_tokens=800)
    except Exception:
        return {}
    res = {}
    for m, v in out.items():
        if isinstance(v, dict) and v.get("category") in CATEGORIES:
            res[m] = (v["category"], float(min(max(v.get("confidence", 0.6), 0.0), 0.85)))
    return res


def categorize(df: pd.DataFrame, memory: Optional[dict[str, str]] = None, use_llm: bool = True) -> pd.DataFrame:
    """Returns a copy of df with merchant, category, confidence, method, is_recurring."""
    memory = dict(memory or {})
    out = df.copy()
    out["merchant"] = out["description"].map(normalize_merchant)
    cats, confs, methods = [], [], []
    for r in out.itertuples():
        if r.amount > 0:
            cats.append("Income"); confs.append(0.95); methods.append("rule")
            continue
        rc = rule_category(r.description, r.amount)
        if r.merchant in memory:             # user corrections / learned merchants win over rules
            cats.append(memory[r.merchant]); confs.append(0.95); methods.append("memory")
        elif rc:
            cats.append(rc); confs.append(0.9); methods.append("rule")
        else:
            cats.append(None); confs.append(0.0); methods.append("")
    out["category"], out["confidence"], out["method"] = cats, confs, methods
    out["is_recurring"] = detect_recurring(out)

    # Layer 3: recurrence — unknown but steady monthly charge
    mask = out["category"].isna() & out["is_recurring"]
    out.loc[mask, ["category", "confidence", "method"]] = ["Subscriptions", 0.6, "recurrence"]

    # Layer 4: batched LLM for what is still unknown
    unk = out[out["category"].isna()]
    if len(unk) and use_llm:
        groups = {m: g["description"].tolist() for m, g in unk.groupby("merchant")}
        llm_res = _llm_categorize(groups)
        for m, (cat, conf) in llm_res.items():
            idx = out.index[(out["merchant"] == m) & out["category"].isna()]
            out.loc[idx, ["category", "confidence", "method"]] = [cat, conf, "llm"]
    # Layer 5: fallback
    still = out["category"].isna()
    out.loc[still, ["category", "confidence", "method"]] = ["Other", 0.3, "fallback"]
    out["confidence"] = out["confidence"].astype(float)
    return out
