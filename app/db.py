"""SQLite persistence. Plain sqlite3 so there is nothing to install or migrate."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'demo',
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    merchant TEXT,
    amount REAL NOT NULL,              -- negative = expense, positive = income
    category TEXT,
    confidence REAL,
    method TEXT,                       -- rule | memory | llm | fallback
    is_recurring INTEGER DEFAULT 0,
    UNIQUE(user_id, date, description, amount)
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'demo',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    rec_type TEXT NOT NULL,
    title TEXT NOT NULL,
    action TEXT NOT NULL,
    impact_amount REAL,
    reason TEXT NOT NULL,              -- WHY: the pattern / forecast that triggered it
    trigger_ref TEXT NOT NULL,         -- machine id of the triggering finding
    evidence TEXT,                     -- JSON with the numbers used
    agent TEXT DEFAULT 'advisor'
);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'demo',
    name TEXT NOT NULL,
    target_amount REAL NOT NULL,
    deadline TEXT NOT NULL,            -- YYYY-MM-DD
    saved_so_far REAL NOT NULL DEFAULT 0,
    start_date TEXT
);
CREATE TABLE IF NOT EXISTS merchant_memory (
    user_id TEXT NOT NULL DEFAULT 'demo',
    merchant TEXT NOT NULL,
    category TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (user_id, merchant)
);
CREATE TABLE IF NOT EXISTS settings (
    user_id TEXT NOT NULL DEFAULT 'demo',
    key TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY (user_id, key)
);
CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'demo',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    body TEXT NOT NULL,
    channel TEXT NOT NULL
);
"""


@contextmanager
def connect(path: str | None = None):
    p = path or config.DB_PATH
    if p != ":memory:":
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(path: str | None = None) -> None:
    with connect(path) as con:
        con.executescript(SCHEMA)


def reset_user(user_id: str = "demo") -> None:
    with connect() as con:
        for t in ("transactions", "decisions", "goals", "settings", "digests"):
            con.execute(f"DELETE FROM {t} WHERE user_id=?", (user_id,))


def insert_transactions(rows: Iterable[dict], user_id: str = "demo") -> int:
    n = 0
    with connect() as con:
        for r in rows:
            cur = con.execute(
                """INSERT OR IGNORE INTO transactions
                   (user_id,date,description,merchant,amount,category,confidence,method,is_recurring)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (user_id, r["date"], r["description"], r.get("merchant"), r["amount"],
                 r.get("category"), r.get("confidence"), r.get("method"),
                 int(r.get("is_recurring", 0))),
            )
            n += cur.rowcount
    return n


def load_transactions(user_id: str = "demo") -> pd.DataFrame:
    with connect() as con:
        df = pd.read_sql_query(
            "SELECT * FROM transactions WHERE user_id=? ORDER BY date, id", con, params=(user_id,))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def update_categories(df: pd.DataFrame, user_id: str = "demo") -> None:
    with connect() as con:
        for r in df.itertuples():
            con.execute(
                """UPDATE transactions SET merchant=?, category=?, confidence=?, method=?, is_recurring=?
                   WHERE id=? AND user_id=?""",
                (r.merchant, r.category, float(r.confidence), r.method, int(r.is_recurring), int(r.id), user_id))


def get_memory(user_id: str = "demo") -> dict[str, str]:
    with connect() as con:
        rows = con.execute("SELECT merchant, category FROM merchant_memory WHERE user_id=?", (user_id,)).fetchall()
    return {r["merchant"]: r["category"] for r in rows}


def remember(merchant: str, category: str, user_id: str = "demo") -> None:
    with connect() as con:
        con.execute(
            """INSERT INTO merchant_memory(user_id,merchant,category,hits) VALUES(?,?,?,1)
               ON CONFLICT(user_id,merchant) DO UPDATE SET category=excluded.category, hits=hits+1""",
            (user_id, merchant, category))


def save_decisions(recs: list[dict], user_id: str = "demo") -> None:
    with connect() as con:
        for r in recs:
            con.execute(
                """INSERT INTO decisions(user_id,rec_type,title,action,impact_amount,reason,trigger_ref,evidence,agent)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (user_id, r["rec_type"], r["title"], r["action"], r.get("impact_amount"),
                 r["reason"], r["trigger_ref"], json.dumps(r.get("evidence", {}), default=str),
                 r.get("agent", "advisor")))


def list_decisions(user_id: str = "demo", limit: int = 50) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM decisions WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_setting(key: str, default: Any = None, user_id: str = "demo") -> Any:
    with connect() as con:
        r = con.execute("SELECT value FROM settings WHERE user_id=? AND key=?", (user_id, key)).fetchone()
    return json.loads(r["value"]) if r else default


def set_setting(key: str, value: Any, user_id: str = "demo") -> None:
    with connect() as con:
        con.execute(
            """INSERT INTO settings(user_id,key,value) VALUES(?,?,?)
               ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value""",
            (user_id, key, json.dumps(value)))


def add_goal(name: str, target: float, deadline: str, saved: float = 0.0,
             start_date: str | None = None, user_id: str = "demo") -> int:
    start_date = start_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    with connect() as con:
        cur = con.execute(
            "INSERT INTO goals(user_id,name,target_amount,deadline,saved_so_far,start_date) VALUES(?,?,?,?,?,?)",
            (user_id, name, target, deadline, saved, start_date))
        return cur.lastrowid


def list_goals(user_id: str = "demo") -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM goals WHERE user_id=?", (user_id,)).fetchall()
    return [dict(r) for r in rows]


def save_digest(body: str, channel: str, user_id: str = "demo") -> None:
    with connect() as con:
        con.execute("INSERT INTO digests(user_id,body,channel) VALUES(?,?,?)", (user_id, body, channel))
