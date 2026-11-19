"""Generative-AI features, powered by the Claude API (Anthropic)."""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
MAX_INPUT_CHARS = 150_000  # keep requests well inside the context window

_client: Optional[Anthropic] = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to a .env file (see .env.example).")
        _client = Anthropic()
    return _client


def _ask(system: str, user: str, max_tokens: int, model: str) -> str:
    response = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def _document(text: str) -> str:
    return f"<document>\n{text[:MAX_INPUT_CHARS]}\n</document>"


def summarize_text(text: str, model: str = DEFAULT_MODEL) -> str:
    system = (
        "You summarize documents for busy professionals. Write a short paragraph with the main point, "
        "then 3-5 bullet points with the most important details. Use the same language as the document. "
        "Do not add information that is not in the document."
    )
    return _ask(system, _document(text), 700, model)


def answer_question(context: str, question: str, model: str = DEFAULT_MODEL) -> str:
    system = (
        "Answer the user's question using only the document. If the document does not contain the answer, "
        "say so clearly instead of guessing. Answer in the language of the question."
    )
    return _ask(system, f"{_document(context)}\n\n<question>{question}</question>", 500, model)


KEY_INFO_FIELDS = ("people", "organizations", "dates", "amounts", "locations", "action_items")


def extract_key_info(text: str, model: str = DEFAULT_MODEL) -> dict:
    """Pull structured facts out of a document (contracts, invoices, reports, emails)."""
    system = (
        "Extract key facts from the document. Return only a JSON object with these keys, each a list of short "
        f"strings: {', '.join(KEY_INFO_FIELDS)}. 'amounts' should include the currency and what the amount is for. "
        "'action_items' are tasks, deadlines or obligations. Use empty lists when nothing is found. "
        "Copy names and figures exactly as written."
    )
    raw = _ask(system, _document(text), 1500, model)
    return parse_json_object(raw)


def parse_json_object(raw: str) -> dict:
    """Parse the model's JSON even if it is wrapped in a code fence or has text around it."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(match.group() if match else raw)
    return {key: [str(v) for v in data.get(key, []) if str(v).strip()] for key in KEY_INFO_FIELDS}
