"""Build and deliver the proactive digest (Telegram if configured, else file). Used by the scheduler."""
from __future__ import annotations

import os
from datetime import datetime

from app import config, db
from agents import advisor


def deliver(text: str) -> str:
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": text}, timeout=15)
        if r.ok:
            return "telegram"
    config.DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    (config.DIGEST_DIR / f"digest_{datetime.now():%Y%m%d_%H%M%S}.md").write_text(text, encoding="utf-8")
    return "file"


def build_and_send(user_id: str = "demo") -> dict:
    from app.coordinator import run_pipeline          # lazy: avoid circular import
    st = run_pipeline(user_id, use_llm=True, persist=True)
    if st.get("empty"):
        return {"sent": False, "reason": "no transactions"}
    text = advisor.compose_digest(st.get("recs", []), st.get("forecast"), st["analysis"].get("profile", {}), st.get("goals", []))
    channel = deliver(text)
    db.save_digest(text, channel, user_id)
    return {"sent": True, "channel": channel, "digest": text}
