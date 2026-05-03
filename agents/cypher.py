"""Cypher generation for the (:Place)-[:Schedule]->(:Place) + (:Place)-[:Fare]->(:Place) graph.

Both a deterministic fallback (`generate_cypher_for_transport`) and an LLM-driven
agent (`cypher_generation_agent`) are exposed. The agent normalizes the agent
state (Title-Case place names, parsed time constraint, structured fare intent)
before handing it to the LLM, so the model only has to fill a stable template.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from .llm import get_llm
from .state import SmartMoveState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _title_case_place(value: str | None) -> str:
    """Normalize a place name to the Title Case form stored in Neo4j (e.g. "Colombo")."""
    if not value:
        return ""
    cleaned = re.sub(r"\s+", " ", value.strip())
    if not cleaned:
        return ""
    return " ".join(part[:1].upper() + part[1:].lower() for part in cleaned.split(" "))


def _parse_clock_to_24h(text: str) -> str | None:
    """Parse a clock token (8pm, 8:30 am, 20:00, 8) into HH:MM 24-hour, or None."""
    if not text:
        return None
    t = text.strip().lower().replace(".", ":")

    m = re.match(r"^(\d{1,2})(?::(\d{1,2}))?\s*(am|pm|a\.m\.|p\.m\.)$", t)
    if m:
        h = int(m.group(1))
        mn = int(m.group(2) or 0)
        ap = m.group(3).replace(".", "")
        if ap.startswith("p") and h < 12:
            h += 12
        if ap.startswith("a") and h == 12:
            h = 0
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return f"{h:02d}:{mn:02d}"
        return None

    m2 = re.match(r"^(\d{1,2}):(\d{2})$", t)
    if m2:
        h = int(m2.group(1))
        mn = int(m2.group(2))
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return f"{h:02d}:{mn:02d}"

    m3 = re.match(r"^(\d{1,2})$", t)
    if m3:
        h = int(m3.group(1))
        if 0 <= h <= 23:
            return f"{h:02d}:00"

    return None


def _parse_departure_constraint(raw: str | None) -> tuple[str | None, str | None]:
    """Map a departure_time phrase to (sql-style operator, HH:MM string)."""
    if not raw:
        return None, None
    s = raw.strip().lower()

    m = re.match(r"^\s*(after|from|>=)\s+(.+)$", s)
    if m:
        return ">=", _parse_clock_to_24h(m.group(2))

    m = re.match(r"^\s*(before|until|<=)\s+(.+)$", s)
    if m:
        return "<=", _parse_clock_to_24h(m.group(2))

    m = re.match(r"^\s*at\s+(.+)$", s)
    if m:
        return "=", _parse_clock_to_24h(m.group(1))

    direct = _parse_clock_to_24h(s)
    if direct:
        return "=", direct
    return None, None


_FARE_BUDGET_RE = re.compile(
    r"(?:max(?:imum)?|under|below|less\s+than|up\s*to|upto|<=|budget(?:\s+of)?)\s*"
    r"(?:lkr|rs\.?|rupees?)?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_fare_intent(fare: str | None) -> dict[str, Any]:
    """Classify the fare string into a structured intent.

    Returns one of:
        {"mode": "skip"}              -> fare = "no"; do not MATCH Fare, hide prices
        {"mode": "any"}               -> include fare info, no filter, no ordering
        {"mode": "cheapest"}          -> ORDER BY f.fare ASC
        {"mode": "budget", "max": N}  -> WHERE f.fare <= N, ORDER BY f.fare ASC
    """
    f = (fare or "").strip().lower()
    if not f or f in {"no", "false", "0", "skip", "none"}:
        return {"mode": "skip"}

    if "cheap" in f or "lowest" in f or "economy" in f:
        return {"mode": "cheapest"}

    m = _FARE_BUDGET_RE.search(f)
    if m:
        try:
            return {"mode": "budget", "max": float(m.group(1))}
        except ValueError:
            pass

    if f in {"any", "include_prices", "yes", "true", "1"}:
        return {"mode": "any"}

    return {"mode": "any"}


# Query-shape classifier --------------------------------------------------
# We pick between three Cypher shapes from the user's *actual words*:
#   "fare_only"     -> the user only wants prices/fares; skip Schedule.
#   "schedule_only" -> the user only wants times/departures; skip Fare.
#   "both"          -> combine Schedule + Fare (default).

# Words that signal a FARE-only ask. English + romanised Sinhala + Sinhala +
# Tamil. We deliberately keep the regex broad — false positives on the fare
# side are cheap; false negatives are what hurt the user (hallucinated
# departure_time gets baked into the query).
_FARE_INTENT_PATTERNS = [
    # English
    r"\bfares?\b",
    r"\bticket\s+(?:price|cost|fee|charge)s?\b",
    r"\bticket\s+rates?\b",
    r"\bhow\s+much\b",
    r"\bcost(?:s|ing|\s+of)?\b",
    r"\bprices?(?:ing)?\b",
    r"\bcharge[ds]?\b",
    r"\brates?\b",
    r"\bbudget\b",
    r"\bcheapest\b",
    r"\blowest\s+(?:fare|price)\b",
    r"\beconomy\b",
    # Sinhala
    r"ගාස්තු",
    r"මිලක්|මිලට|මිලවල්|මිල\b",
    r"මුදල",
    r"කීයද|කීයකටද|කීයටද|කොච්චරද",
    # Tamil
    r"கட்டண",
    r"விலை",
    r"எவ்வளவு",
]
_FARE_INTENT_RE = re.compile("|".join(_FARE_INTENT_PATTERNS), re.IGNORECASE)

# Words that signal a SCHEDULE / time ask.
_SCHEDULE_INTENT_PATTERNS = [
    # English
    r"\bwhen\b",
    r"\bwhat\s+time\b",
    r"\bat\s+what\s+time\b",
    r"\bschedules?\b",
    r"\btimetables?\b",
    r"\btimings?\b",
    r"\bdepart(?:s|ing|ure)?\b",
    r"\bleaves?\b|\bleaving\b",
    r"\barriv(?:e|al|ing|es)\b",
    r"\bnext\s+(?:bus|train|ferry|flight|service)\b",
    r"\bearliest\b",
    r"\blatest\b",
    r"\b(?:after|before)\s+\d",
    r"\b(?:morning|noon|afternoon|evening|night|tonight|midnight|dawn|dusk)\b",
    # Sinhala
    r"වේලාව|වේලාවට|වේලාවන්|වේලාවක්",
    r"පිටත්වෙන|පිටත්වීම|පිටත්වෙනවා",
    r"පැමිණෙන|පැමිණීම|පැමිණෙනවා",
    r"කවදද|කවදාද|කවදට",
    r"ඊළඟ|මීළඟ",
    r"උදේ|රෑ|රාත්‍රී|සවස|දහවල්",
    # Tamil
    r"நேரம்|நேரங்கள்",
    r"அடுத்த",
    r"காலை|மாலை",
]
_SCHEDULE_INTENT_RE = re.compile("|".join(_SCHEDULE_INTENT_PATTERNS), re.IGNORECASE)


def classify_query_shape(state: SmartMoveState) -> str:
    """Classify the user's question into "fare_only" | "schedule_only" | "both".

    Look at BOTH the English-translated query and the original (so Sinhala /
    Tamil words still match). When neither side of the keyword set fires, we
    fall back to "both" — the safe combined shape.
    """
    en = (state.get("user_query") or "").strip()
    orig = (state.get("user_query_original") or "").strip()
    text = (en + " " + orig).strip()
    if not text:
        return "both"

    has_fare = bool(_FARE_INTENT_RE.search(text))
    has_sched = bool(_SCHEDULE_INTENT_RE.search(text))

    if has_fare and not has_sched:
        return "fare_only"
    if has_sched and not has_fare:
        return "schedule_only"
    return "both"


def _strip_cypher_fence(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# Deterministic fallback Cypher
# ---------------------------------------------------------------------------


def _build_fare_only_cypher(state: SmartMoveState) -> str:
    """Standalone Fare-only query. Reusable as a fallback for "both" shape."""
    origin = _title_case_place(state.get("origin")) or "Unknown"
    destination = _title_case_place(state.get("destination")) or "Unknown"
    fare_intent = _parse_fare_intent(state.get("fare"))
    fi = fare_intent if fare_intent["mode"] != "skip" else {"mode": "any"}

    lines: list[str] = [
        f'MATCH (from:Place {{name: "{origin}"}})-[f:Fare]->(to:Place {{name: "{destination}"}})',
    ]
    if fi["mode"] == "budget":
        lines.append(f"WHERE f.fare <= {fi['max']}")
    lines.append(
        "RETURN from.name AS origin, to.name AS destination, f.fare AS fare"
    )
    lines.append("ORDER BY f.fare ASC")
    lines.append("LIMIT 5")
    return "\n".join(lines)


def _build_fare_only_cypher_reversed(state: SmartMoveState) -> str | None:
    """Fare-only query with origin and destination swapped.

    Sri Lankan transport fares are typically symmetric — the same value is
    stored for one direction only. So if the user asks "B -> A" and the graph
    only has the "A -> B" fare edge, this query lets us still answer.
    """
    origin = state.get("origin")
    destination = state.get("destination")
    if not origin or not destination:
        return None
    swapped: SmartMoveState = {
        **state,
        "origin": destination,
        "destination": origin,
    }
    return _build_fare_only_cypher(swapped)


def _build_schedule_only_cypher(state: SmartMoveState) -> str:
    """Standalone Schedule-only query. Reusable as a fallback for "both" shape."""
    origin = _title_case_place(state.get("origin")) or "Unknown"
    destination = _title_case_place(state.get("destination")) or "Unknown"
    transport = (state.get("transport_type") or "").strip().lower()
    op, time_24 = _parse_departure_constraint(state.get("departure_time"))

    lines: list[str] = [
        f'MATCH (from:Place {{name: "{origin}"}})-[s:Schedule]->(to:Place {{name: "{destination}"}})',
    ]
    where_parts: list[str] = []
    if op and time_24:
        where_parts.append(f's.departure {op} "{time_24}"')
    if transport:
        where_parts.append(
            f"(toLower(coalesce(s.transport_type, s.service_type, '')) CONTAINS '{transport}')"
        )
    if where_parts:
        lines.append("WHERE " + " AND ".join(where_parts))
    lines.append(
        "RETURN from.name AS origin, to.name AS destination, "
        "s.departure AS departure_time, s.arrival AS arrival_time, "
        "s.route_type AS route_type, s.service_type AS service_type"
    )
    lines.append("ORDER BY s.departure")
    lines.append("LIMIT 5")
    return "\n".join(lines)


def _build_combined_cypher(state: SmartMoveState) -> str:
    """Combined Schedule + Fare query (the "both" shape)."""
    origin = _title_case_place(state.get("origin")) or "Unknown"
    destination = _title_case_place(state.get("destination")) or "Unknown"
    transport = (state.get("transport_type") or "").strip().lower()
    op, time_24 = _parse_departure_constraint(state.get("departure_time"))
    fare_intent = _parse_fare_intent(state.get("fare"))

    lines = [
        f'MATCH (from:Place {{name: "{origin}"}})-[s:Schedule]->(to:Place {{name: "{destination}"}})',
    ]
    where_parts = []
    if op and time_24:
        where_parts.append(f's.departure {op} "{time_24}"')
    if transport:
        where_parts.append(
            f"(toLower(coalesce(s.transport_type, s.service_type, '')) CONTAINS '{transport}')"
        )
    if where_parts:
        lines.append("WHERE " + " AND ".join(where_parts))

    return_cols = [
        "from.name AS origin",
        "to.name AS destination",
        "s.departure AS departure_time",
        "s.arrival AS arrival_time",
    ]
    if fare_intent["mode"] != "skip":
        lines.append("MATCH (from)-[f:Fare]->(to)")
        if fare_intent["mode"] == "budget":
            lines.append(f"WITH from, to, s, f WHERE f.fare <= {fare_intent['max']}")
            lines.append("ORDER BY f.fare ASC")
        elif fare_intent["mode"] == "cheapest":
            lines.append("WITH from, to, s, f")
            lines.append("ORDER BY f.fare ASC")
        else:
            lines.append("WITH from, to, s, f")
        return_cols.append("f.fare AS fare")
    else:
        lines.append("ORDER BY s.departure")

    lines.append("RETURN " + ", ".join(return_cols))
    lines.append("LIMIT 5")
    return "\n".join(lines)


def generate_cypher_for_transport(state: SmartMoveState) -> str:
    """Deterministic Cypher generator (no LLM). Routes by shape."""
    shape = classify_query_shape(state)
    if shape == "fare_only":
        return _build_fare_only_cypher(state)
    if shape == "schedule_only":
        return _build_schedule_only_cypher(state)
    return _build_combined_cypher(state)


def generate_cypher_bundle(state: SmartMoveState) -> dict[str, str | None]:
    """Generate every Cypher query the executor might need for this turn.

    Returned keys:
      - `combined`     -> primary query (combined Schedule+Fare for "both",
                          or the single-edge query for fare_only/schedule_only)
      - `schedule`     -> Schedule-only fallback (only for shape == "both")
      - `fare`         -> Fare-only fallback (only for shape == "both")
      - `fare_reverse` -> Fare-only with origin/destination swapped, used as a
                          last-resort retry for any shape that involves fares
                          (since SL fares are typically symmetric)
    """
    shape = classify_query_shape(state)
    if shape == "fare_only":
        return {
            "combined": _build_fare_only_cypher(state),
            "schedule": None,
            "fare": None,
            "fare_reverse": _build_fare_only_cypher_reversed(state),
        }
    if shape == "schedule_only":
        return {
            "combined": _build_schedule_only_cypher(state),
            "schedule": None,
            "fare": None,
            "fare_reverse": None,
        }
    return {
        "combined": _build_combined_cypher(state),
        "schedule": _build_schedule_only_cypher(state),
        "fare": _build_fare_only_cypher(state),
        "fare_reverse": _build_fare_only_cypher_reversed(state),
    }


# ---------------------------------------------------------------------------
# LLM-driven Cypher agent
# ---------------------------------------------------------------------------


_CYPHER_SYSTEM_PROMPT = """You are a Neo4j Cypher expert for the SmartMove Sri Lankan transport graph.

GRAPH SCHEMA (this is the ONLY schema — do not invent labels or relationships):

  (:Place {name})
      `name` is in Title Case, e.g. "Colombo", "Kandy", "Galle", "Negombo".

  (:Place)-[:Schedule {
      departure,                 // "HH:MM" 24-hour string, e.g. "06:15", "20:00"
      arrival,                   // "HH:MM" 24-hour string
      departure_from_terminal,   // optional, "HH:MM"
      arrival_to_terminal,       // optional, "HH:MM"
      route_type,                // optional, e.g. "Expressway", "Normal Route"
      service_type,              // optional, e.g. "Luxury"
      route_no                   // optional
  }]->(:Place)

  (:Place)-[:Fare {
      fare,                      // numeric LKR
      route_id                   // optional
  }]->(:Place)

QUERY SHAPE — `query_shape` is the MOST IMPORTANT input. Pick exactly ONE template:

  shape = "fare_only"     // user is asking ONLY about price/fare/cost.
    MATCH (from:Place {name: "<Origin>"})-[f:Fare]->(to:Place {name: "<Destination>"})
    [WHERE f.fare <= <budget>]                  // only when fare_intent.mode = "budget"
    RETURN from.name AS origin, to.name AS destination, f.fare AS fare
    ORDER BY f.fare ASC
    LIMIT 5
    RULES:
      - DO NOT MATCH :Schedule.
      - DO NOT include departure / arrival / route_type / service_type in RETURN.
      - Ignore departure_constraint completely (the user said nothing about time).
      - Always ORDER BY f.fare ASC and always include f.fare, even if fare_intent.mode = "skip".

  shape = "schedule_only" // user is asking ONLY about times / next service.
    MATCH (from:Place {name: "<Origin>"})-[s:Schedule]->(to:Place {name: "<Destination>"})
    [WHERE s.departure <op> "<HH:MM>" [AND <transport filter>]]
    RETURN from.name AS origin, to.name AS destination,
           s.departure AS departure_time, s.arrival AS arrival_time,
           s.route_type AS route_type, s.service_type AS service_type
    ORDER BY s.departure
    LIMIT 5
    RULES:
      - DO NOT MATCH :Fare. DO NOT include f.fare in RETURN even if fare_intent says so.

  shape = "both"          // combined route+fare lookup (default).
    MATCH (from:Place {name: "<Origin>"})-[s:Schedule]->(to:Place {name: "<Destination>"})
    [WHERE s.departure <op> "<HH:MM>" [AND <transport filter>]]
    [MATCH (from)-[f:Fare]->(to)]                  // when fare_intent.mode != "skip"
    [WITH from, to, s, f [WHERE f.fare <= <budget>]]
    [ORDER BY f.fare ASC]                          // when fare_intent.mode in {cheapest, budget}
    RETURN
      from.name AS origin, to.name AS destination,
      s.departure AS departure_time, s.arrival AS arrival_time
      [, f.fare AS fare]                           // only when fare is matched
    LIMIT 5

NORMALIZATION RULES (already applied for you in `inputs`):
  - Place names are pre-converted to Title Case (Neo4j-stored form).
  - `departure_constraint` is {"op": ">="|"<="|"=", "time": "HH:MM"} or null.
  - `fare_intent.mode` is one of: "skip" | "any" | "cheapest" | "budget".

OUTPUT:
- Return ONLY the Cypher query (no markdown fences, no commentary, no $params).
- Single-line clauses separated by newlines.
- Always end with `LIMIT 5`.
"""


_CYPHER_FEW_SHOT = """EXAMPLES

# 1) Schedule + fare combined
inputs:
{
  "origin": "Colombo",
  "destination": "Kandy",
  "departure_constraint": {"op": ">=", "time": "20:00"},
  "transport_type": null,
  "fare_intent": {"mode": "cheapest"},
  "query_shape": "both"
}
output:
MATCH (from:Place {name: "Colombo"})-[s:Schedule]->(to:Place {name: "Kandy"})
WHERE s.departure >= "20:00"
MATCH (from)-[f:Fare]->(to)
WITH from, to, s, f
ORDER BY f.fare ASC
RETURN from.name AS origin, to.name AS destination, s.departure AS departure_time, s.arrival AS arrival_time, f.fare AS fare
LIMIT 5

# 2) Fare-only lookup ("what are the fares from Anuradhapura to Colombo?")
inputs:
{
  "origin": "Anuradhapura",
  "destination": "Colombo",
  "departure_constraint": null,
  "transport_type": null,
  "fare_intent": {"mode": "any"},
  "query_shape": "fare_only"
}
output:
MATCH (from:Place {name: "Anuradhapura"})-[f:Fare]->(to:Place {name: "Colombo"})
RETURN from.name AS origin, to.name AS destination, f.fare AS fare
ORDER BY f.fare ASC
LIMIT 5

# 3) Schedule-only lookup ("when is the next bus from Galle to Matara?")
inputs:
{
  "origin": "Galle",
  "destination": "Matara",
  "departure_constraint": null,
  "transport_type": "bus",
  "fare_intent": {"mode": "skip"},
  "query_shape": "schedule_only"
}
output:
MATCH (from:Place {name: "Galle"})-[s:Schedule]->(to:Place {name: "Matara"})
WHERE (toLower(coalesce(s.transport_type, s.service_type, '')) CONTAINS 'bus')
RETURN from.name AS origin, to.name AS destination, s.departure AS departure_time, s.arrival AS arrival_time, s.route_type AS route_type, s.service_type AS service_type
ORDER BY s.departure
LIMIT 5
"""


def _llm_cypher_for_shape(state: SmartMoveState, shape: str) -> str:
    """Ask the LLM for a single-shape Cypher query, falling back to the deterministic builder."""
    op, time_24 = _parse_departure_constraint(state.get("departure_time"))
    fare_intent = _parse_fare_intent(state.get("fare"))
    if shape == "fare_only":
        op, time_24 = None, None

    if shape == "fare_only":
        deterministic = _build_fare_only_cypher(state)
    elif shape == "schedule_only":
        deterministic = _build_schedule_only_cypher(state)
    else:
        deterministic = _build_combined_cypher(state)

    if not os.getenv("OPENAI_API_KEY"):
        return deterministic

    inputs: dict[str, Any] = {
        "origin": _title_case_place(state.get("origin")) or None,
        "destination": _title_case_place(state.get("destination")) or None,
        "departure_constraint": (
            {"op": op, "time": time_24} if op and time_24 else None
        ),
        "transport_type": state.get("transport_type"),
        "fare_intent": fare_intent,
        "query_shape": shape,
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _CYPHER_SYSTEM_PROMPT + "\n" + _CYPHER_FEW_SHOT),
            ("human", "inputs:\n{inputs_json}\noutput:"),
        ]
    )
    try:
        response = (prompt | get_llm()).invoke(
            {"inputs_json": json.dumps(inputs, indent=2, default=str)}
        ).content
        cypher = _strip_cypher_fence(response)
        if not cypher or "MATCH" not in cypher.upper():
            cypher = deterministic
    except Exception:
        cypher = deterministic
    return cypher


def cypher_generation_agent(state: SmartMoveState) -> SmartMoveState:
    """Build the primary Cypher plus, for combined queries, two single-edge fallbacks.

    Output keys:
      - `cypher_query`           -> the primary query (combined when shape == "both")
      - `cypher_query_schedule`  -> Schedule-only fallback (None for non-"both")
      - `cypher_query_fare`      -> Fare-only fallback (None for non-"both")
    """
    shape = classify_query_shape(state)

    primary = _llm_cypher_for_shape(state, shape)
    schedule_q: str | None = None
    fare_q: str | None = None
    fare_reverse_q: str | None = None
    if shape == "both":
        # Build the two single-edge fallbacks deterministically — they are
        # short, unambiguous, and don't need an LLM round-trip. We still want
        # the LLM-quality combined query as the primary attempt.
        schedule_q = _build_schedule_only_cypher(state)
        fare_q = _build_fare_only_cypher(state)
        fare_reverse_q = _build_fare_only_cypher_reversed(state)
    elif shape == "fare_only":
        # Same direction as primary, but with origin/destination swapped.
        fare_reverse_q = _build_fare_only_cypher_reversed(state)

    return {
        **state,
        "cypher_query": primary,
        "cypher_query_schedule": schedule_q,
        "cypher_query_fare": fare_q,
        "cypher_query_fare_reverse": fare_reverse_q,
    }


# Backwards-compatible alias
cypher_generator_node = cypher_generation_agent
