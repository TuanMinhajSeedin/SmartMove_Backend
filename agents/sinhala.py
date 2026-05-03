"""Sinhala input helpers: on-screen keyboard rows + Singlish→Sinhala conversion."""

from __future__ import annotations

import hashlib
import os

from langchain_core.prompts import ChatPromptTemplate

from .llm import get_llm


# Compact Sinhala keyboard: independent letters + common vowel signs
# (the user composes syllables by combining consonants with vowel signs).
SINHALA_KEYBOARD_ROWS: list[list[str]] = [
    ["අ", "ආ", "ඇ", "ඈ", "ඉ", "ඊ", "උ", "ඌ", "එ", "ඒ", "ඔ", "ඕ"],
    ["ක", "ඛ", "ග", "ඝ", "ඞ", "ඟ", "ච", "ඡ", "ජ", "ඣ", "ඤ", "ඥ"],
    ["ට", "ඨ", "ඩ", "ඪ", "ණ", "ත", "ථ", "ද", "ධ", "න"],
    ["ප", "ඵ", "බ", "භ", "ම", "ඹ", "ය", "ර", "ල", "ව"],
    ["ශ", "ෂ", "ස", "හ", "ළ", "ෆ", "ං", "ඃ", "ඳ"],
    ["්", "ා", "ැ", "ෑ", "ි", "ී", "ු", "ූ", "ෘ", "ෲ"],
    ["ෙ", "ේ", "ෛ", "ො", "ෝ", "ෞ"],
]


def _llm_singlish_to_sinhala(text: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You convert Singlish (romanized Sinhala typed in Latin letters) into proper Sinhala Unicode.\n"
                "Rules:\n"
                "- Output ONLY Sinhala script — no Latin, no explanation, no quotes.\n"
                "- Preserve numbers, times, and place names read naturally in Sinhala.\n"
                "- Keep punctuation that makes sense in Sinhala (e.g. ? !).\n"
                "- If the input is empty, output nothing.",
            ),
            ("human", "{text}"),
        ]
    )
    out = (prompt | get_llm()).invoke({"text": text}).content
    return (out or "").strip()


def singlish_to_sinhala(text: str) -> str:
    """Convert Singlish to Sinhala. Uses OpenAI when `OPENAI_API_KEY` is set; else returns text unchanged."""
    raw = (text or "").strip()
    if not raw:
        return ""
    if not os.getenv("OPENAI_API_KEY"):
        return raw
    try:
        return _llm_singlish_to_sinhala(raw)
    except Exception:
        return raw


def singlish_cache_key(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]
