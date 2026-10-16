"""The ONLY place LLM calls are made. Everything goes through the Mesh API gateway
(OpenAI-compatible): base_url=https://api.meshapi.ai/v1, key from MESH_API_KEY.

If no key is configured (tests, offline eval) `available()` is False and callers use their
deterministic fallback, so the system degrades gracefully instead of crashing.
"""
import json
import re
from typing import Optional

from app import config

_client = None


def available() -> bool:
    if not config.MESH_API_KEY:
        return False
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=config.MESH_BASE_URL, api_key=config.MESH_API_KEY)
    return _client


def chat(system: str, user: str, *, model: Optional[str] = None, temperature: float = 0.2,
         max_tokens: int = 600) -> str:
    resp = _get_client().chat.completions.create(
        model=model or config.MESH_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def chat_json(system: str, user: str, **kw) -> dict:
    """Ask for JSON only; tolerate code fences / stray prose."""
    raw = chat(system + "\nRespond with a single JSON object and nothing else.", user, **kw)
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0) if m else raw)
