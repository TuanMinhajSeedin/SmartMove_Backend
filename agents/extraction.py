"""Field extraction: regex heuristics, LLM extraction, normalization, and merge.

Inputs are an English (or translated) user query; outputs are partial dicts that
can be merged into `SmartMoveState`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from .llm import get_llm
from .places import canonical_place
from .state import STATE_FIELDS, SmartMoveState


# ---------------------------------------------------------------------------
# Place + datetime helpers
# ---------------------------------------------------------------------------


def normalize_datetime(raw_text: str) -> str | None:
    """Pull a normalized departure-time phrase out of a free-text query."""
    if not raw_text:
        return None
    rel = re.search(r"\b(after|before)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", raw_text, re.IGNORECASE)
    if rel:
        return f"{rel.group(1).lower()} {rel.group(2).lower()}"
    m = re.search(r"(\d{1,2}\s+[A-Za-z]+\s+\d{4}(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)?)", raw_text)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)", raw_text)
    if m2:
        return m2.group(1)
    t = re.search(r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", raw_text, re.IGNORECASE)
    if t:
        return t.group(1).lower()
    return None


def _clean_place(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().lower()
    cleaned = re.sub(r"^(?:the\s+)?city\s+of\s+", "", cleaned, flags=re.IGNORECASE)
    while True:
        next_cleaned = re.sub(
            r"^(?:need\s+to\s+|go\s+to\s+|go\s+|travel\s+to\s+|travel\s+|reach\s+|get\s+to\s+|get\s+)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
    return cleaned.strip() or None


def _looks_like_place_name(raw: str | None) -> bool:
    """Reject generic travel phrases and non-location text; keep plausible place tokens."""
    if not raw:
        return False
    t = re.sub(r"\s+", " ", raw.strip().lower())
    if len(t) < 2 or len(t) > 80:
        return False

    if re.fullmatch(
        r"(on\s+)?a\s+trip|on\s+a\s+trip|trip|trips?|vacation|holiday|journey|"
        r"plan\s+a\s+trip|go\s+on\s+a\s+trip|for\s+a\s+trip|take\s+a\s+trip|"
        r"travel(ing)?|touring|sightseeing|outing|abroad|home|there|here|work|office|"
        r"go|going|get|getting|come|coming|leave|leaving|back|away",
        t,
    ):
        return False

    if re.search(
        r"\b(on\s+a\s+trip|a\s+trip\b|plan\s+a\s+trip|go\s+on\s+a\s+trip|for\s+a\s+trip)\b",
        t,
    ):
        return False

    words = t.split()
    if len(words) > 5:
        return False

    junk_only = {"on", "a", "the", "to", "for", "in", "at", "my", "your", "our", "an", "and", "or", "of", "is", "it"}
    trip_words = {"trip", "trips", "travel", "vacation", "holiday", "journey", "tour", "tours", "planning"}
    if len(words) >= 2 and all(w in junk_only | trip_words for w in words):
        return False

    return True


def _sanitize_place_field(value: str | None) -> str | None:
    """Clean + canonicalize a place token.

    A canonical match (e.g. "මහනුවර" / "mahanuwara" / "nuwara" -> "kandy") wins
    even when the raw token is short / non-Latin and would otherwise fail the
    English-only `_looks_like_place_name` heuristic.
    """
    if value is None:
        return None
    canon = canonical_place(value)
    if canon:
        return canon
    cleaned = _clean_place(value)
    if cleaned is None:
        return None
    canon = canonical_place(cleaned)
    if canon:
        return canon
    return cleaned if _looks_like_place_name(cleaned) else None


def _extract_travel_date(text: str) -> str | None:
    """Date-only part (not the same as clock time). Complements departure_time."""
    if not text:
        return None
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)
    m2 = re.search(r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b", text)
    if m2:
        return m2.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Fare helpers
# ---------------------------------------------------------------------------


def extract_fare_from_query(text: str) -> str | None:
    """Infer fare / budget / price intent from English (or translated) query text."""
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()

    if re.search(r"\b(cheapest|lowest\s+(?:price|fare)|economy(?:\s+option)?)\b", low):
        return "cheapest"
    if re.search(r"\b(any\s+price|don'?t\s+care|no\s+budget|no\s+limit|price\s+doesn'?t\s+matter)\b", low):
        return "any"

    m = re.search(
        r"\b(?:under|below|less\s+than|max(?:imum)?|up\s*to|upto|<=)\s*(?:lkr|rs\.?|rupees?)?\s*([\d,]+(?:\.\d+)?)\b",
        low,
    )
    if m:
        return f"max LKR {m.group(1).replace(',', '')}"

    m2 = re.search(r"\bbudget\s*(?:of|:)?\s*(?:lkr|rs\.?)?\s*([\d,]+(?:\.\d+)?)\b", low)
    if m2:
        return f"budget LKR {m2.group(1).replace(',', '')}"

    m3 = re.search(r"\b(?:lkr|rs\.?)\s*([\d,]+(?:\.\d+)?)\b(?:\s*(?:max|budget|limit))?\b", low)
    if m3 and re.search(r"\b(?:budget|max|under|below|limit|fare|price|cost)\b", low):
        return f"budget LKR {m3.group(1).replace(',', '')}"

    if re.search(
        r"\b(how\s+much|what'?s?\s+the\s+fare|what\s+is\s+the\s+fare|ticket\s+price|cost\s+of|fare\s+for|prices?\s+for|include\s+(?:the\s+)?(?:fare|price|cost))\b",
        low,
    ):
        return "include_prices"

    return None


def _heuristic_fare_from_query(user_query: str) -> str | None:
    """When LLM omits fare: infer generic fare interest or regex preference."""
    q = (user_query or "").strip()
    if not q:
        return None
    pref = extract_fare_from_query(q)
    if pref:
        return pref
    low = q.lower()
    if re.search(
        r"\b(fares?\b|ticket\s+prices?|ticket\s+cost|bus\s+fare|train\s+fare|"
        r"price\s+and\s+schedule|schedule\s+and\s+fare|fare\s+and\s+schedule|"
        r"how\s+much\b.*\b(fare|ticket|cost|price)|"
        r"\b(cost|pricing)\b.*\b(bus|train|ticket))",
        low,
    ):
        return "yes"
    return None


# ---------------------------------------------------------------------------
# Regex-only extraction (fallback / heuristic enrichment)
# ---------------------------------------------------------------------------


def extract_transport_fields(user_query: str) -> dict[str, str | None]:
    """Pure-regex extraction used as a fallback when no LLM key is present."""
    text = user_query or ""

    transport_type = None
    for t in ["bus", "train", "car", "taxi", "metro", "flight", "ferry"]:
        if re.search(rf"\b{t}\b", text, re.IGNORECASE):
            transport_type = t
            break

    origin = None
    destination = None

    go_from = re.search(
        r"(?:go(?:\s+to)?|travel(?:\s+to)?|reach|get(?:\s+to)?)\s+([A-Za-z\s]+?)\s+from\s+([A-Za-z\s]+?)(?:$|\s+(?:on|at|by|for|after|before)\b)",
        text,
        re.IGNORECASE,
    )
    if go_from:
        destination = go_from.group(1).strip()
        origin = go_from.group(2).strip()

    if not (origin and destination):
        from_to = re.search(
            r"from\s+([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+?)(?:$|\s+(?:on|at|by|for|after|before)\b)",
            text,
            re.IGNORECASE,
        )
        if from_to:
            origin = from_to.group(1).strip()
            destination = from_to.group(2).strip()

    if not (origin and destination):
        to_from = re.search(
            r"\bto\s+([A-Za-z\s]+?)\s+from\s+([A-Za-z\s]+?)(?:$|\s+(?:on|at|by|for|after|before)\b)",
            text,
            re.IGNORECASE,
        )
        if to_from:
            destination = to_from.group(1).strip()
            origin = to_from.group(2).strip()

    if not destination:
        transport_to = re.search(
            r"\b(?:a\s+)?(?:bus|train|car|taxi|metro|flight|ferry)\s+to\s+([A-Za-z][A-Za-z\s]*?)(?:$|\s+(?:on|at|by|for|after|before|from)\b)",
            text,
            re.IGNORECASE,
        )
        if transport_to:
            destination = transport_to.group(1).strip()

    if not destination:
        to_match = re.search(
            r"(?:go|travel|reach|get)\s+to\s+([A-Za-z\s]+?)(?:$|\s+(?:on|at|by|for|after|before)\b)",
            text,
            re.IGNORECASE,
        )
        if to_match:
            destination = to_match.group(1).strip()

    origin = _sanitize_place_field(origin)
    destination = _sanitize_place_field(destination)

    departure_time = normalize_datetime(text)
    date_val = _extract_travel_date(text)
    if not date_val and departure_time:
        dm = re.match(r"^(\d{4}-\d{2}-\d{2})(?:\s+\d{2}:\d{2})?$", departure_time.strip())
        if dm:
            date_val = dm.group(1)
    if not date_val:
        date_val = departure_time

    fare = extract_fare_from_query(text)

    return {
        "origin": origin,
        "destination": destination,
        "departure_time": departure_time,
        "date": date_val,
        "transport_type": transport_type,
        "fare": fare,
    }


# ---------------------------------------------------------------------------
# JSON / value helpers
# ---------------------------------------------------------------------------


def _strip_json_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def _parse_json_obj(raw: str) -> dict[str, Any]:
    text = _strip_json_fence(raw)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _value_present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and not v.strip():
        return False
    if isinstance(v, str) and v.strip().lower() in ("null", "none", "n/a"):
        return False
    return True


def normalize_extraction_output(data: dict[str, Any]) -> dict[str, Any]:
    """Map LLM keys to state fields + fare / fare_preference merge rules."""
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for k in STATE_FIELDS:
        if k == "fare":
            continue
        if k in data and data[k] is not None:
            out[k] = data[k]

    fp = data.get("fare_preference")
    fv = data.get("fare")
    wants = data.get("wants_fare_info")

    if _value_present(fp):
        out["fare"] = str(fp).strip()
    elif _value_present(fv):
        out["fare"] = fv
    elif wants is True:
        out["fare"] = "yes"
    elif isinstance(fv, str) and fv.strip().lower() in ("null", "none"):
        pass
    else:
        out.pop("fare", None)

    fare = out.get("fare")
    if isinstance(fare, str):
        fl = fare.strip().lower()
        if fl in ("cheap", "cheaper", "low cost", "affordable"):
            out["fare"] = "cheapest"
    return out


def _coerce_extracted_value(field: str, v: Any) -> str | None:
    if not _value_present(v):
        return None
    s = str(v).strip()
    if field in ("origin", "destination"):
        p = _sanitize_place_field(s)
        if p:
            return p
        c = _clean_place(s)
        return c if c else s.lower()
    if field == "departure_time":
        return normalize_datetime(s) or s.lower()
    if field == "fare":
        sl = s.lower().strip()
        if sl in ("yes", "true", "1"):
            return "yes"
        if sl in ("no", "false", "0", "none", "skip", "not interested", "no thanks"):
            return "no"
        normalized = extract_fare_from_query(s)
        if normalized:
            return normalized
        return s
    if field == "transport_type":
        return s.lower()
    if field == "date":
        return s
    return s


def merge_extracted_into_state(state: SmartMoveState, extracted: dict[str, Any]) -> dict[str, str | None]:
    """New extraction overrides prior state when present; supports partial updates."""
    norm = normalize_extraction_output(extracted)
    merged: dict[str, str | None] = {}
    for field in STATE_FIELDS:
        v = norm.get(field)
        if _value_present(v):
            merged[field] = _coerce_extracted_value(field, v)
        else:
            merged[field] = state.get(field)  # type: ignore[assignment]
    return merged


# ---------------------------------------------------------------------------
# LLM extraction prompts
# ---------------------------------------------------------------------------


def llm_extract_transport_from_query(user_query: str) -> dict[str, Any]:
    """Single-shot LLM extraction; returns the raw JSON dict."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract transportation details from the user query (English).\n"
                "Return JSON ONLY with these keys:\n"
                "- origin\n"
                "- destination\n"
                "- departure_time\n"
                "- date\n"
                "- transport_type\n"
                "- fare_preference\n"
                "- wants_fare_info (boolean)\n"
                "- fare (optional; use only when needed — see rules)\n\n"
                "Fare rules (important):\n"
                "- fare_preference: use when the user expresses a *specific* price intent — e.g. cheapest, "
                "lowest fare, under/below/max LKR amount, budget range, 'between X and Y', 'not too expensive'.\n"
                "  Put a short normalized English phrase (e.g. cheapest, max LKR 2000).\n"
                "- wants_fare_info: true if they ask for fares/prices/ticket cost/schedule that clearly includes "
                "fare or pricing information in a general way (e.g. 'bus fare and schedule', 'ticket prices', "
                "'how much is the fare') and they did NOT give a specific fare_preference constraint.\n"
                "- fare: use the literal string \"yes\" ONLY for a generic fare/price inquiry when wants_fare_info "
                "would be true and fare_preference is null. If you set fare_preference, you may omit fare or set it null.\n"
                "- If the user does not care about fare at all, set wants_fare_info false and fare_preference null.\n\n"
                "Other rules:\n"
                "- Only extract what is clearly stated or reasonably implied for travel planning.\n"
                "- Do NOT invent specific places if the user did not give them.\n"
                "- Natural language times (e.g. after lunch, tomorrow morning) go in departure_time or date.\n"
                "- Use null for unknown fields.\n"
                "Respond with a single JSON object only. No markdown fences.",
            ),
            ("human", "{query}"),
        ]
    )
    raw = (prompt | get_llm()).invoke({"query": user_query}).content
    return _parse_json_obj(raw)


def followup_llm_extract(state: SmartMoveState, user_reply: str) -> dict[str, str | None]:
    """LLM extraction for follow-up replies; fills only keys still missing."""
    missing = state.get("missing_fields") or []
    if not missing or not (user_reply or "").strip() or not os.getenv("OPENAI_API_KEY"):
        return {}
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract ONLY transportation values from the user reply (English).\n"
                "Return JSON only with keys among:\n"
                "origin, destination, departure_time, date, transport_type, "
                "fare_preference, wants_fare_info (boolean), fare (string; use \"yes\" only for generic fare interest)\n\n"
                "Fare:\n"
                "- fare_preference for cheapest, max LKR, budget range, between amounts.\n"
                "- wants_fare_info true if they ask for fare/prices without a specific constraint.\n"
                "- fare: \"yes\" for generic fare inquiry when appropriate.\n\n"
                "Rules:\n"
                "- Include only what the reply clearly provides.\n"
                "- Do not guess missing endpoints.\n"
                "- Use null for absent fields.\n"
                "Single JSON object only. No markdown.",
            ),
            ("human", "Still needed: {missing}\nUser reply: {reply}"),
        ]
    )
    raw = (prompt | get_llm()).invoke(
        {"missing": ", ".join(missing), "reply": user_reply.strip()}
    ).content
    parsed = _parse_json_obj(raw)
    data = normalize_extraction_output(parsed)
    out: dict[str, str | None] = {}
    for field in missing:
        if field == "fare":
            v = data.get("fare")
            if not _value_present(v):
                v = data.get("fare_preference")
            if not _value_present(v) and parsed.get("wants_fare_info") is True:
                v = "yes"
        else:
            v = data.get(field)
        if _value_present(v):
            coerced = _coerce_extracted_value("fare" if field == "fare" else field, v)
            if coerced:
                out[field] = coerced
    return out
