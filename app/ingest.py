"""CSV / manual ingestion. Accepts messy bank exports: flexible column names, signed amounts OR
separate debit/credit columns, ISO or day-first dates."""
from __future__ import annotations

import io
import pandas as pd

DATE_COLS = ["date", "transaction date", "txn date", "posted date", "value date", "posting date"]
DESC_COLS = ["description", "narration", "details", "particulars", "merchant", "memo", "transaction details", "remarks"]
DEBIT_COLS = ["debit", "withdrawal", "withdrawals", "debit amount", "dr"]
CREDIT_COLS = ["credit", "deposit", "deposits", "credit amount", "cr"]


def _pick(cols, options):
    return next((c for c in options if c in cols), None)


def parse_transactions_csv(content: bytes | str) -> tuple[list[dict], dict]:
    raw = pd.read_csv(io.BytesIO(content) if isinstance(content, bytes) else io.StringIO(content))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    cols = list(raw.columns)
    dcol, tcol = _pick(cols, DATE_COLS), _pick(cols, DESC_COLS)
    if not dcol or not tcol:
        raise ValueError(f"CSV needs a date column and a description column. Found: {cols}")
    if "amount" in cols:
        amt = pd.to_numeric(raw["amount"].astype(str).str.replace(r"[,₹$\s]", "", regex=True), errors="coerce")
        if "type" in cols:                       # amounts unsigned + a DR/CR column
            is_debit = raw["type"].astype(str).str.lower().str.startswith(("d", "w"))
            amt = amt.abs() * is_debit.map({True: -1, False: 1})
    else:
        dr, cr = _pick(cols, DEBIT_COLS), _pick(cols, CREDIT_COLS)
        if not dr and not cr:
            raise ValueError("CSV needs an 'amount' column, or debit/credit columns.")
        num = lambda c: pd.to_numeric(raw[c].astype(str).str.replace(r"[,₹$\s]", "", regex=True), errors="coerce").fillna(0) if c else 0
        amt = num(cr) - num(dr)
    # ISO (YYYY-MM-DD) is unambiguous. Anything else (03/04/2026) is read day-first, the norm for
    # Indian/UK/EU bank exports; US-style month-first files should be converted to ISO before upload.
    sample = raw[dcol].astype(str).str.strip()
    iso = sample.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").mean() > 0.5
    dates = pd.to_datetime(sample, errors="coerce", dayfirst=not iso)
    out = pd.DataFrame({"date": dates, "description": raw[tcol].astype(str).str.strip(), "amount": amt})
    bad = out["date"].isna() | out["amount"].isna() | (out["amount"] == 0) | (out["description"] == "")
    good = out[~bad]
    rows = [{"date": r.date.strftime("%Y-%m-%d"), "description": r.description, "amount": float(r.amount)}
            for r in good.itertuples()]
    return rows, {"rows_in": len(raw), "rows_ok": len(rows), "rows_skipped": int(bad.sum())}
