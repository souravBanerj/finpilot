"""FastAPI entrypoint.  Run:  uvicorn app.main:app --reload   (docs at /docs)"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app import coordinator, db
from app.ingest import parse_transactions_csv


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    sched = None
    try:
        from app.scheduler import start_scheduler
        sched = start_scheduler()
    except Exception as e:                      # scheduler is a bonus; never block the API
        print("scheduler not started:", e)
    yield
    if sched:
        sched.shutdown(wait=False)


app = FastAPI(title="FinPilot", version="1.0", lifespan=lifespan)
USER = "demo"


class Txn(BaseModel):
    date: str
    description: str
    amount: float                    # negative = expense, positive = income


class PurchaseReq(BaseModel):
    amount: float
    name: str = "purchase"
    month_offset: int = 1            # 1 = first forecast month


class GoalSimReq(BaseModel):
    target: float
    deadline: str
    saved: float = 0.0


class GoalReq(BaseModel):
    name: str
    target_amount: float
    deadline: str
    saved_so_far: float = 0.0


class ChatReq(BaseModel):
    question: str


class Correction(BaseModel):
    category: str


@app.get("/health")
def health():
    from app import llm
    return {"status": "ok", "llm_available": llm.available()}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        rows, report = parse_transactions_csv(await file.read())
    except Exception as e:
        raise HTTPException(400, str(e))
    inserted = db.insert_transactions(rows, USER)
    st = coordinator.run_pipeline(USER)
    return {**report, "inserted": inserted, "trace": st.get("trace"),
            "findings": len(st["analysis"].get("findings", [])), "recommendations": len(st.get("recs", []))}


@app.post("/transactions")
def add_transaction(t: Txn):
    db.insert_transactions([t.model_dump()], USER)
    st = coordinator.run_pipeline(USER)
    return {"ok": True, "recommendations": st.get("recs", [])}


@app.get("/transactions")
def transactions(limit: int = 100):
    df = db.load_transactions(USER).tail(limit)
    if df.empty:
        return []
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")   # NaN is not valid JSON


@app.patch("/transactions/{txn_id}")
def correct(txn_id: int, c: Correction):
    """User correction → merchant memory, so the categorizer learns."""
    df = db.load_transactions(USER)
    row = df[df["id"] == txn_id]
    if row.empty:
        raise HTTPException(404, "transaction not found")
    from agents.categorizer import normalize_merchant
    db.remember(normalize_merchant(row.iloc[0]["description"]), c.category, USER)
    coordinator.run_pipeline(USER, use_llm=False)
    return {"ok": True}


@app.get("/insights")
def insights():
    st = coordinator.run_pipeline(USER, use_llm=False, persist=False)
    return {"profile": st["analysis"].get("profile"), "findings": st["analysis"].get("findings"),
            "monthly_by_category": st["analysis"].get("monthly_by_category")}


@app.get("/forecast")
def get_forecast(months: int = 6):
    from agents import forecaster
    st = coordinator.run_pipeline(USER, use_llm=False, persist=False)
    if st.get("empty"):
        raise HTTPException(400, "Upload transactions first")
    return forecaster.forecast(st["df"], months, db.get_setting("opening_balance", 0.0, USER))


@app.post("/simulate/purchase")
def sim_purchase(r: PurchaseReq):
    return coordinator.handle("simulate", USER, kind="purchase", amount=r.amount, name=r.name, month_offset=r.month_offset)


@app.post("/simulate/goal")
def sim_goal(r: GoalSimReq):
    return coordinator.handle("simulate", USER, kind="goal", target=r.target, deadline=r.deadline, saved=r.saved)


@app.post("/goals")
def create_goal(g: GoalReq):
    return {"id": db.add_goal(g.name, g.target_amount, g.deadline, g.saved_so_far, user_id=USER)}


@app.get("/goals")
def goals():
    st = coordinator.run_pipeline(USER, use_llm=False, persist=False)
    return st.get("goals", [])


@app.post("/settings/balance")
def set_balance(opening_balance: float):
    db.set_setting("opening_balance", opening_balance, USER)
    return {"opening_balance": opening_balance}


@app.post("/settings/subscription-usage")
def set_usage(usage: dict[str, str]):
    """{'NETFLIX': '2026-06-01'} = last date you used it. Enables 'unused subscription' detection."""
    db.set_setting("subscription_usage", usage, USER)
    return {"ok": True}


@app.post("/recommendations/run")
def run_recs():
    st = coordinator.run_pipeline(USER)
    return {"trace": st["trace"], "recommendations": st.get("recs", [])}


@app.get("/recommendations")
def audit_trail(limit: int = 50):
    """The decision trail: every recommendation + WHY (trigger_ref, reason, evidence)."""
    return db.list_decisions(USER, limit)


@app.post("/chat")
def chat(r: ChatReq):
    return coordinator.handle("chat", USER, question=r.question)


@app.post("/digest/send")
def send_digest():
    return coordinator.handle("digest", USER)


@app.post("/demo/load")
def load_demo(seed: int = 1, opening_balance: float = 0.0):
    """Load a synthetic persona so you can demo instantly without a CSV."""
    from eval.generate_data import generate_persona
    df, _ = generate_persona(seed)
    db.reset_user(USER)
    db.insert_transactions([{"date": r.date.strftime("%Y-%m-%d"), "description": r.description, "amount": r.amount}
                            for r in df.itertuples()], USER)
    db.set_setting("opening_balance", opening_balance, USER)
    st = coordinator.run_pipeline(USER)
    return {"loaded": len(df), "trace": st["trace"], "recommendations": len(st.get("recs", []))}
