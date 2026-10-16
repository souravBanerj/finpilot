"""Seeded synthetic persona generator WITH ground truth.

Why synthetic: we need to know the true category of every transaction and exactly which anomalies
were injected, otherwise accuracy/precision/recall are unmeasurable.

Each persona = 12 complete months of a salaried user. Injected, labelled events:
  * Food Delivery spike in the LAST month (more orders)
  * 2 single-transaction outliers (a big Shopping and a big Dining charge) in random earlier months
  * 2 duplicate charges (same merchant, same amount, <=2 days apart)
  * Netflix price hike partway through
  * Overlapping subscriptions (3 video-streaming, 2 music) and 2 unknown-name subscriptions
Hard cases are included on purpose (unknown merchants, 'AMAZON FRESH', 'SWIGGY INSTAMART') so the
reported accuracy is honest rather than flattering.
"""
from __future__ import annotations

import calendar
import numpy as np
import pandas as pd

START = "2025-10-01"
N_MONTHS = 12

# category -> (txns per month, median amount, sigma, [(merchant, kind)]) ; kind: known|unknown|hard
POOLS = {
    "Groceries": (6, 1200, 0.35, [("BIGBASKET", "known"), ("DMART AVENUE", "known"), ("BLINKIT", "known"),
                                   ("RELIANCE FRESH", "known"), ("ZEPTO", "known"), ("NATURES BASKET", "unknown"),
                                   ("FRESHTOHOME", "unknown"), ("SHREE KIRANA", "unknown"),
                                   ("AMAZON FRESH", "hard"), ("SWIGGY INSTAMART", "hard")]),
    "Food Delivery": (8, 420, 0.30, [("SWIGGY", "known"), ("ZOMATO", "known"), ("DOMINOS PIZZA", "known"),
                                      ("EATSURE", "unknown")]),
    "Dining": (4, 900, 0.40, [("STARBUCKS", "known"), ("CAFE COFFEE DAY", "known"), ("MCDONALDS", "known"),
                               ("TANDOOR TALES", "unknown"), ("BIRYANI BLUES", "unknown"), ("PUNJABI TADKA", "unknown")]),
    "Transport": (14, 180, 0.35, [("UBER", "known"), ("OLA CABS", "known"), ("RAPIDO", "known"),
                                   ("HP PETROL PUMP", "known"), ("METRO CARD", "known"), ("NAMMA YATRI", "unknown")]),
    "Shopping": (3, 1800, 0.50, [("AMAZON", "known"), ("FLIPKART", "known"), ("MYNTRA", "known"),
                                  ("NYKAA", "known"), ("BOAT LIFESTYLE", "unknown"), ("WESTSIDE", "unknown")]),
    "Entertainment": (2, 700, 0.35, [("BOOKMYSHOW", "known"), ("PVR CINEMAS", "known"), ("TIMEZONE GAMES", "unknown")]),
    "Health": (0.5, 900, 0.5, [("APOLLO PHARMACY", "known"), ("MEDPLUS", "known"), ("KUMAR MEDICALS", "unknown")]),
    "Travel": (0.15, 6000, 0.4, [("MAKEMYTRIP", "known"), ("INDIGO AIRLINES", "known")]),
}
PREFIXES = ["UPI-", "POS ", "", "CARD ", "UPI-"]
SUBS = [  # merchant, amount, day, true_category
    ("NETFLIX", 649, 7), ("SPOTIFY", 119, 12), ("HOTSTAR", 299, 15), ("AMAZON PRIME", 299, 18),
    ("JIOSAAVN", 99, 21), ("CULT.FIT GYM", 1500, 9),
    ("PIXELCLOUD STORAGE", 130, 24), ("ZENITH LEARN", 499, 27),      # unknown names: must be caught by recurrence
]


def _desc(rng, name):
    pre = PREFIXES[rng.integers(len(PREFIXES))]
    ref = int(rng.integers(10000, 99999))
    suffix = ["", f" {ref}", f" {ref}@okaxis", f" TXN{ref}"][rng.integers(4)]
    return f"{pre}{name}{suffix}".strip()


def generate_persona(seed: int = 0) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    months = pd.period_range(START, periods=N_MONTHS, freq="M")
    rows = []

    def add(date, name, amount, cat, tag=None):
        rows.append({"date": pd.Timestamp(date), "description": _desc(rng, name), "amount": float(round(amount, 2)),
                     "true_category": cat, "tag": tag})

    pay_base = float(rng.choice([65000, 85000, 120000]))
    rent = float(round(pay_base * rng.uniform(0.26, 0.32), -3))
    hike_month = int(rng.integers(7, 10))
    out_months = rng.choice(np.arange(2, N_MONTHS - 2), size=2, replace=False)
    dup_months = rng.choice(np.arange(2, N_MONTHS - 1), size=2, replace=False)
    unknown_boost = 0.12

    for mi, p in enumerate(months):
        dim = calendar.monthrange(p.year, p.month)[1]
        d = lambda day: p.to_timestamp() + pd.Timedelta(days=int(min(day, dim)) - 1)
        add(d(1), "SALARY ACME CORP", pay_base, "Income")
        if rng.random() < 0.2:
            add(d(int(rng.integers(10, 25))), "FREELANCE PAYMENT", rng.uniform(8000, 15000), "Income")
        add(d(3), "RENT TRANSFER LANDLORD", -rent, "Rent")
        add(d(5), "ZERODHA SIP", -10000, "Savings")
        season = 1 + 0.25 * np.sin((p.month - 3) / 12 * 2 * np.pi)
        add(d(8), "TATA POWER ELECTRICITY BILL", -rng.normal(2200, 150) * season, "Utilities")
        add(d(10), "ACT FIBERNET BROADBAND", -999, "Utilities")
        add(d(14), "AIRTEL MOBILE RECHARGE", -399, "Utilities")
        for name, amt, day in SUBS:
            price = 799 if (name == "NETFLIX" and mi >= hike_month) else amt
            add(d(day + int(rng.integers(-1, 2))), name, -price, "Subscriptions",
                "price_hike" if (name == "NETFLIX" and mi == hike_month) else None)
        for cat, (n_mean, med, sig, merchants) in POOLS.items():
            n = rng.poisson(n_mean)
            if cat == "Food Delivery" and mi == N_MONTHS - 1:
                n = 15                                           # injected spike
            for _ in range(n):
                known = [m for m in merchants if m[1] == "known"]
                other = [m for m in merchants if m[1] != "known"]
                pick = other[rng.integers(len(other))] if (other and rng.random() < unknown_boost) else known[rng.integers(len(known))]
                amt = float(rng.lognormal(np.log(med), sig))
                add(d(int(rng.integers(1, dim + 1))), pick[0], -amt, cat)
        # injected outliers / duplicates
        if mi == out_months[0]:
            add(d(int(rng.integers(5, 25))), "CROMA ELECTRONICS", -rng.uniform(32000, 45000), "Shopping", "outlier")
        if mi == out_months[1]:
            add(d(int(rng.integers(5, 25))), "TAJ RESTAURANT", -rng.uniform(9000, 14000), "Dining", "outlier")
        if mi == dup_months[0]:
            day = int(rng.integers(5, 25)); amt = round(float(rng.uniform(450, 700)))
            add(d(day), "SWIGGY", -amt, "Food Delivery", "dup_first"); add(d(day), "SWIGGY", -amt, "Food Delivery", "dup_second")
            rows[-1]["pair"] = rows[-2]["pair"] = f"s{mi}"
        if mi == dup_months[1]:
            day = int(rng.integers(5, 24)); amt = round(float(rng.uniform(2000, 3500)))
            add(d(day), "AMAZON", -amt, "Shopping", "dup_first"); add(d(day + 2), "AMAZON", -amt, "Shopping", "dup_second")
            rows[-1]["pair"] = rows[-2]["pair"] = f"a{mi}"

    for r in rows:
        r.setdefault("pair", None)
    df = pd.DataFrame(rows).sort_values(["date", "description"]).reset_index(drop=True)
    df["id"] = np.arange(1, len(df) + 1)
    last_month = str(months[-1])
    truth = {
        "seed": seed, "last_month": last_month,
        "spike": {"category": "Food Delivery", "month": last_month},
        "outlier_ids": df.loc[df["tag"] == "outlier", "id"].tolist(),
        "duplicate_pairs": [sorted(g["id"].tolist()) for _, g in df[df["pair"].notna()].groupby("pair")],
        "price_hike_merchant": "NETFLIX",
        "overlap_groups": ["video streaming", "music"],
        "hidden_subscriptions": ["PIXELCLOUD STORAGE", "ZENITH LEARN"],
        "subscription_total": sum(799 if n == "NETFLIX" else a for n, a, _ in SUBS),
    }
    return df, truth
