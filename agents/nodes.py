"""LangGraph nodes that orchestrate the SmartMove pipeline."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt

from .cypher import (
    _parse_departure_constraint,
    _parse_fare_intent,
    _title_case_place,
    classify_query_shape,
    cypher_generation_agent,
)
from .extraction import (
    _coerce_extracted_value,
    _heuristic_fare_from_query,
    _sanitize_place_field,
    _value_present,
    extract_fare_from_query,
    extract_transport_fields,
    followup_llm_extract,
    llm_extract_transport_from_query,
    merge_extracted_into_state,
    normalize_datetime,
    normalize_extraction_output,
)
from .i18n import SUPPORTED_LANGS, _t, detect_language, to_english
from .llm import get_llm
from .neo4j_client import execute_neo4j_query
from .state import SmartMoveState


# ---------------------------------------------------------------------------
# Pre-extraction nodes
# ---------------------------------------------------------------------------


def language_detection_node(state: SmartMoveState) -> SmartMoveState:
    original = state.get("user_query_original") or state.get("user_query") or ""
    lang = state.get("language") or detect_language(original)
    user_query_en = to_english(original, lang)
    return {
        **state,
        "language": lang,
        "user_query_original": original,
        "user_query": user_query_en or original,
    }


def intent_detection_node(state: SmartMoveState) -> SmartMoveState:
    query = state.get("user_query") or ""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are SmartMove intent router.\n"
                "Classify user message into one of: greeting, transport, fallback.\n"
                "Return only one label.",
            ),
            ("human", "{query}"),
        ]
    )
    intent = (prompt | get_llm()).invoke({"query": query}).content.strip().lower()
    if intent not in {"greeting", "transport", "fallback"}:
        intent = "fallback"
    return {**state, "intent": intent}


def greeting_node(state: SmartMoveState) -> SmartMoveState:
    lang = state.get("language") or "en"
    response = _t(lang, "greeting")
    return {
        **state,
        "response": response,
        "messages": state.get("messages", []) + [AIMessage(content=response)],
    }


def fallback_node(state: SmartMoveState) -> SmartMoveState:
    lang = state.get("language") or "en"
    response = _t(lang, "fallback")
    return {
        **state,
        "response": response,
        "messages": state.get("messages", []) + [AIMessage(content=response)],
    }


# ---------------------------------------------------------------------------
# Field extraction nodes
# ---------------------------------------------------------------------------


def llm_extract_node(state: SmartMoveState) -> SmartMoveState:
    q = (state.get("user_query") or "").strip()
    extracted: dict[str, Any] = {}
    if os.getenv("OPENAI_API_KEY") and q:
        try:
            extracted = llm_extract_transport_from_query(q)
        except Exception:
            extracted = {}
    if not extracted:
        extracted = extract_transport_fields(q)
    norm = normalize_extraction_output(extracted)
    if not _value_present(norm.get("fare")):
        hint = _heuristic_fare_from_query(q)
        if hint:
            norm = normalize_extraction_output({**extracted, "fare": hint})
    return {**state, "extracted_data": norm}


def merge_state_node(state: SmartMoveState) -> SmartMoveState:
    ext = state.get("extracted_data") or {}
    merged_fields = merge_extracted_into_state(state, ext)
    return {**state, **merged_fields}


# ---------------------------------------------------------------------------
# Validation + follow-up
# ---------------------------------------------------------------------------


def validate_mandatory_fields(state: SmartMoveState) -> list[str]:
    """Origin + destination always required.

    `departure_time` is only required when the user actually asked about
    schedules / times. `fare` is only required when the user did NOT ask a
    pure schedule question (so we can still apply the fare toggle).
    """
    shape = classify_query_shape(state)
    missing: list[str] = []
    if not state.get("origin"):
        missing.append("origin")
    if not state.get("destination"):
        missing.append("destination")
    if shape != "fare_only" and not state.get("departure_time"):
        missing.append("departure_time")
    if shape != "schedule_only" and not state.get("fare"):
        missing.append("fare")
    return missing


def missing_info_validator_node(state: SmartMoveState) -> SmartMoveState:
    return {**state, "missing_fields": validate_mandatory_fields(state)}


def _regex_followup_updates(state: SmartMoveState, missing: list[str], user_input: Any) -> dict[str, str | None]:
    """Original dict / regex follow-up path (fallback)."""
    lang = state.get("language") or "en"
    if isinstance(user_input, dict):
        updates: dict[str, str | None] = {}
        for field in missing:
            if state.get(field):
                continue
            v = user_input.get(field)
            if isinstance(v, str) and v.strip():
                v_en = to_english(v, lang)
                if field in {"origin", "destination"}:
                    p = _sanitize_place_field(v_en)
                    if p:
                        updates[field] = p
                elif field == "departure_time":
                    updates[field] = normalize_datetime(v_en) or v_en.strip().lower()
                elif field == "fare":
                    coerced = _coerce_extracted_value("fare", v_en)
                    updates[field] = coerced or v_en.strip()
                else:
                    updates[field] = v_en.strip()
        return updates

    user_text = str(user_input).strip()
    user_text_en = to_english(user_text, lang)
    extracted = extract_transport_fields(user_text_en)
    updates2: dict[str, str | None] = {}
    for field in missing:
        if state.get(field):
            continue
        value = extracted.get(field)
        if not value:
            if field in {"origin", "destination"}:
                value = _sanitize_place_field(user_text_en)
            elif field == "departure_time":
                value = normalize_datetime(user_text_en) or user_text_en.strip().lower() or None
            elif field == "fare":
                value = extract_fare_from_query(user_text_en) or (user_text_en.strip() or None)
        if value:
            updates2[field] = value
    return updates2


def _extract_missing_field_updates(state: SmartMoveState, missing: list[str], user_input: Any) -> dict[str, str | None]:
    regex_updates = _regex_followup_updates(state, missing, user_input)
    if isinstance(user_input, dict):
        return regex_updates
    user_text = str(user_input).strip()
    llm_updates: dict[str, str | None] = {}
    if os.getenv("OPENAI_API_KEY") and user_text:
        try:
            llm_updates = followup_llm_extract(state, user_text)
        except Exception:
            llm_updates = {}
    merged = dict(regex_updates)
    for k, v in llm_updates.items():
        if v:
            merged[k] = v
    return merged


def follow_up_question_node(state: SmartMoveState) -> SmartMoveState:
    missing = state.get("missing_fields") or []
    lang = state.get("language") or "en"
    field_map = {
        "origin": _t(lang, "origin"),
        "destination": _t(lang, "destination"),
        "departure_time": _t(lang, "departure_time"),
        "fare": _t(lang, "fare"),
    }
    missing_text = ", ".join(field_map.get(f, f) for f in missing)
    response = f"{_t(lang, 'follow_up_prefix')}: {missing_text}."

    human_reply = interrupt(
        {"kind": "follow_up_question", "question": response, "missing_fields": missing}
    )
    updates = _extract_missing_field_updates(state, missing, human_reply)

    messages = state.get("messages", []) + [AIMessage(content=response)]
    messages = messages + [HumanMessage(content=str(human_reply).strip())]

    return {
        **state,
        **updates,
        "follow_up_question": response,
        "response": response,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Cypher / Neo4j / response
# ---------------------------------------------------------------------------


def cypher_generator_node(state: SmartMoveState) -> SmartMoveState:
    """Thin wrapper to expose the cypher agent under a stable node name."""
    return cypher_generation_agent(state)


_NEO4J_ERROR_PREFIXES = ("Neo4j query error", "Neo4j not configured", "No Cypher query")


def _is_neo4j_error(raw: str | None) -> bool:
    if not raw:
        return False
    return str(raw).strip().startswith(_NEO4J_ERROR_PREFIXES)


def _looks_empty(raw: str | None) -> bool:
    """JSON `[]` (or missing payload) is "empty". Errors aren't empty."""
    if raw is None:
        return True
    s = str(raw).strip()
    if not s or s == "[]":
        return True
    if _is_neo4j_error(s):
        return False
    try:
        data = json.loads(s)
        return isinstance(data, list) and len(data) == 0
    except Exception:
        return False


def _is_usable(raw: str | None) -> bool:
    """A result is "usable" when it parses to a non-empty list (no errors)."""
    if raw is None:
        return False
    if _is_neo4j_error(raw):
        return False
    if _looks_empty(raw):
        return False
    return True


def neo4j_query_node(state: SmartMoveState) -> SmartMoveState:
    """Run the primary Cypher and cascade through every fallback we have whenever
    the primary is *unusable* — i.e. empty OR a transient Neo4j error like
    "Failed to obtain connection towards 'WRITE' server.". Even if the primary
    errored, the schedule / fare / fare-reverse queries may still succeed on a
    different driver session, so we give them a chance.

    Cascade outcomes (`result_source`):
      - "combined"      -> primary OR a fare-reverse retry returned rows.
      - "split"         -> "both" shape, primary unusable, schedule+fare run separately.
                           At least one of schedule/fare is usable.
      - "primary_error" -> nothing usable came back. We surface the most
                           informative error string we received.

    `fare_reversed` is True when the executor substituted the reversed fare
    query for the forward one.
    """
    primary_q = state.get("cypher_query") or ""
    schedule_q = state.get("cypher_query_schedule") or ""
    fare_q = state.get("cypher_query_fare") or ""
    fare_reverse_q = state.get("cypher_query_fare_reverse") or ""

    primary_result = execute_neo4j_query(primary_q)
    out: dict[str, Any] = {
        **state,
        "result": primary_result,
        "result_schedule": None,
        "result_fare": None,
        "fare_reversed": False,
    }

    if _is_usable(primary_result):
        out["result_source"] = "combined"
        return out

    has_combined_fallback = bool(schedule_q or fare_q)

    # ---- "both" shape: run schedule + fare separately, with fare-reverse retry ----
    if has_combined_fallback:
        sched_res = execute_neo4j_query(schedule_q) if schedule_q else None
        fare_res = execute_neo4j_query(fare_q) if fare_q else None
        if not _is_usable(fare_res) and fare_reverse_q:
            rev = execute_neo4j_query(fare_reverse_q)
            if _is_usable(rev):
                fare_res = rev
                out["fare_reversed"] = True
        out["result_schedule"] = sched_res
        out["result_fare"] = fare_res

        if _is_usable(sched_res) or _is_usable(fare_res):
            out["result_source"] = "split"
        else:
            # Surface the most informative error we have (primary -> schedule -> fare).
            for cand in (primary_result, sched_res, fare_res):
                if _is_neo4j_error(cand):
                    out["result"] = cand
                    break
            out["result_source"] = "primary_error"
        return out

    # ---- pure fare-only shape: try the reversed fare ----
    if fare_reverse_q:
        rev = execute_neo4j_query(fare_reverse_q)
        if _is_usable(rev):
            out["result"] = rev
            out["fare_reversed"] = True
            out["result_source"] = "combined"
            return out
        if _is_neo4j_error(rev) and not _is_neo4j_error(primary_result):
            out["result"] = rev
        out["result_source"] = (
            "primary_error" if _is_neo4j_error(out["result"]) else "combined"
        )
        return out

    # No fallbacks available.
    out["result_source"] = (
        "primary_error" if _is_neo4j_error(primary_result) else "combined"
    )
    return out


_RESPONSE_FORMATTER_SYSTEM = """You are SmartMove's Response Formatter Agent operating in RAG mode.

ROLE
You receive the user's question plus a small set of retrieved Neo4j rows
(`context.rows`) that the upstream Cypher agent already filtered and ordered.
Treat those rows as your ONLY ground truth — exactly like a RAG retrieval set.
Compose a final answer that is what the user actually asked for, in their
language, honouring their preferences.

LANGUAGE LOCK
- Respond ONLY in {lang_name} ({lang_code}). Never mix languages.
- Mirror the user's tone (formal vs casual) from `user_query`.
- Place names follow the source script (e.g. "Colombo" -> "කොළඹ" in Sinhala);
  keep clock times as numerals like 06:15 / 20:30.

DEDUPLICATION (CRITICAL)
- `context.rows` is ALREADY deduplicated upstream. Every row you see is a
  distinct option — never restate the same fare / departure / route twice.
- If multiple rows still share the SAME displayable fact (e.g. all have
  fare = LKR 1195 across different schedules), do NOT list it multiple times.
  Mention the value ONCE. You may add a brief qualifier like
  "(consistent across services)" if appropriate.
- Count distinct facts before you write. A list of 4 identical fares should
  collapse into a single sentence, not 4 bullet points.

ANSWER SHAPE — pick whichever serves the user's intent AND the row count:
A. SINGLE-FACT shape (rows collapse to ONE distinct fact, OR user asked
   "single best option" — cheapest / earliest / latest / specific fare lookup):
   - One or two direct sentences naming the fact.
   - Examples (Sinhala): "අනුරාධපුර සිට කොළඹට ගාස්තුව **LKR 1195** වේ."
   - Examples (English): "The fare from Anuradhapura to Colombo is **LKR 1195**."
   - For a winning schedule row: "The earliest bus is **06:15 → 09:30**, fare `LKR 450`."
   - NO bullet list. NO repetition.
B. BROWSE shape (rows contain MULTIPLE genuinely-distinct facts):
   - One short intro sentence (mention origin -> destination).
   - Markdown numbered list (`1.` / `2.` / ...) — at most {max_rows} entries,
     in the order rows are given. Each line:
       "**HH:MM → HH:MM** — <qualifiers>"
     Append fare when allowed: " — `LKR <fare>`".
   - Each list item must convey something the previous one didn't.
   - Optional one-line closing tip (e.g. "Cheapest option listed first.").

FARE PREFERENCE — fare_intent.mode = `{fare_mode}`:
- "cheapest" -> Always cite fare per option. Call out the top row as the cheapest.
- "budget"   -> Always cite fare per option. Flag the cheapest in-budget pick.
- "any"      -> Cite fare per option, neutrally. Don't editorialise.
- "skip"     -> NEVER mention price, fare, cost, LKR, or budget anywhere.

EMPTY ROWS
- Apologise briefly and suggest ONE concrete tweak (a different time,
  transport, or fare preference). Do NOT invent rows.

SPLIT CONTEXT (`context.source == "split"`)
- This means we couldn't find a SINGLE service that has BOTH a matching
  schedule AND a fare record, so we ran the schedule-only and fare-only
  queries separately. `context.schedule_rows` and `context.fare_rows` carry
  those two result sets.
- Open with one short sentence explaining that no combined record was found
  but you have schedules and fares separately (in the user's language).
- Present TWO labelled sections with markdown headings:
    "**Schedules**" / "**ගමන් වේලාවන්**" / "**அட்டவணைகள்**"
    "**Fares**" / "**ගාස්තු**" / "**கட்டணங்கள்**"
- Each section uses the BROWSE shape rules (numbered list, deduped, concise).
- If one of the two lists is empty, say so briefly in that section instead of
  inventing rows.
- Close with ONE short tip suggesting the user can pick a schedule and ask
  about its fare specifically.

REVERSE-DIRECTION FARE (`context.fare_reversed == true`)
- Fares in our graph are typically stored in only one direction. When the
  user asked "B -> A" but only "A -> B" had fare data, the fare values you
  see were retrieved from the reverse direction. They are still correct.
- Continue presenting the user's requested direction (origin -> destination)
  in the answer text. Optionally add ONE short parenthetical note in the
  user's language, e.g.
    EN: "(fares are typically the same in either direction)"
    SI: "(ගාස්තු සාමාන්‍යයෙන් දෙපැත්තටම සමාන වේ)"
    TA: "(கட்டணம் இரு திசையிலும் பெரும்பாலும் ஒன்றாகவே இருக்கும்)"
- Do NOT lead with the note, do NOT apologise, and do NOT repeat the note
  more than once.

RETRIEVAL ERROR (`context.error` not null)
- Apologise briefly and explain the issue in plain language. Do NOT show
  the raw error or any Cypher / stack traces.

ABSOLUTE RULES
- Use ONLY facts present in `context.rows`. No fabrication of places, times,
  fares, or transport types.
- No JSON, no Cypher, no markdown fences around the whole response,
  no "as an AI" preambles.
- Keep it tight — concise is better than complete.
"""


_RESPONSE_FORMATTER_USER = """user_query (LANGUAGE={lang_code}):
{user_query_original}

user_query_english:
{user_query_english}

preferences:
{preferences_json}

context:
{context_json}
"""


def _parse_result_payload(raw: str | None) -> tuple[list[dict[str, Any]], str | None]:
    """Convert the JSON-string returned by `execute_neo4j_query` into rows + optional error."""
    if raw is None:
        return [], "No result available."
    s = str(raw).strip()
    if not s:
        return [], "No result available."
    if s.startswith(("Neo4j query error", "Neo4j not configured", "No Cypher query")):
        return [], s
    try:
        data = json.loads(s)
    except Exception:
        return [], s
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)], None
    return [], "Unexpected result shape."


_ROW_FIELDS = (
    "origin",
    "destination",
    "departure_time",
    "arrival_time",
    "departure",
    "arrival",
    "departure_from_terminal",
    "arrival_to_terminal",
    "route_type",
    "service_type",
    "route_no",
    "fare",
)


def _row_signature(row: dict[str, Any]) -> tuple:
    """Stable signature of a row across the fields the user actually sees.

    Two rows with the same origin/destination/departure/arrival/route info AND
    the same fare collapse to one entry — so we never tell the user the same
    "LKR 1195" four times in a row.
    """
    norm = []
    for k in _ROW_FIELDS:
        v = row.get(k)
        if isinstance(v, str):
            norm.append(v.strip().lower())
        elif isinstance(v, (int, float)):
            norm.append(float(v))
        else:
            norm.append(v)
    return tuple(norm)


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order-preserving dedupe by `_row_signature`."""
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        sig = _row_signature(r)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(r)
    return out


def _trim_rows_for_prompt(
    rows: list[dict[str, Any]], max_rows: int = 8
) -> list[dict[str, Any]]:
    """Project to display fields, dedupe, and cap row count.

    Dedupe BEFORE the cap so a Neo4j result of 5 identical fares doesn't kick
    out a different (unique) option that came after.
    """
    projected: list[dict[str, Any]] = []
    for row in rows:
        item = {k: row.get(k) for k in _ROW_FIELDS if row.get(k) not in (None, "")}
        if item:
            projected.append(item)
    deduped = _dedupe_rows(projected)
    return deduped[:max_rows]


def _build_preferences(state: SmartMoveState) -> dict[str, Any]:
    """Structured preference snapshot the formatter LLM consumes as part of the RAG prompt."""
    op, time_24 = _parse_departure_constraint(state.get("departure_time"))
    fare_intent = _parse_fare_intent(state.get("fare"))
    return {
        "origin": _title_case_place(state.get("origin")) or None,
        "destination": _title_case_place(state.get("destination")) or None,
        "departure_constraint": (
            {"op": op, "time": time_24} if op and time_24 else None
        ),
        "departure_time_raw": state.get("departure_time"),
        "transport_type": state.get("transport_type"),
        "fare_raw": state.get("fare"),
        "fare_intent": fare_intent,
    }


def response_formatter_node(state: SmartMoveState) -> SmartMoveState:
    """RAG-style synthesis: rows are context, the user's query is the question.

    Honors `result_source`:
      - "combined" / "primary_error" / unset -> use `result` (default behaviour).
      - "split"                              -> use `result_schedule` + `result_fare`
                                                and ask the formatter to present
                                                them as two separate sections.
    """
    user_query_original = (
        state.get("user_query_original") or state.get("user_query") or ""
    )
    user_query_english = state.get("user_query") or user_query_original
    raw_result = state.get("result") or ""
    lang_code = state.get("language") or "en"
    lang_name = SUPPORTED_LANGS.get(lang_code, "English")
    result_source = state.get("result_source") or "combined"

    preferences = _build_preferences(state)
    fare_mode = preferences["fare_intent"].get("mode", "skip")

    rows, error = _parse_result_payload(raw_result)
    trimmed = _trim_rows_for_prompt(rows, max_rows=8)

    schedule_rows: list[dict[str, Any]] = []
    fare_rows: list[dict[str, Any]] = []
    if result_source == "split":
        sched_raw = state.get("result_schedule") or ""
        fare_raw = state.get("result_fare") or ""
        s_rows, _ = _parse_result_payload(sched_raw)
        f_rows, _ = _parse_result_payload(fare_raw)
        schedule_rows = _trim_rows_for_prompt(s_rows, max_rows=8)
        fare_rows = _trim_rows_for_prompt(f_rows, max_rows=8)

    context_block: dict[str, Any] = {
        "source": result_source,
        "rows": trimmed,
        "schedule_rows": schedule_rows,
        "fare_rows": fare_rows,
        "error": error,
        "fare_reversed": bool(state.get("fare_reversed")),
    }

    prompt = ChatPromptTemplate.from_messages(
        [("system", _RESPONSE_FORMATTER_SYSTEM), ("human", _RESPONSE_FORMATTER_USER)]
    )

    response = ""
    if os.getenv("OPENAI_API_KEY"):
        try:
            response = (prompt | get_llm()).invoke(
                {
                    "lang_name": lang_name,
                    "lang_code": lang_code,
                    "fare_mode": fare_mode,
                    "max_rows": 8,
                    "user_query_original": user_query_original,
                    "user_query_english": user_query_english,
                    "preferences_json": json.dumps(
                        preferences, ensure_ascii=False, indent=2
                    ),
                    "context_json": json.dumps(
                        context_block, ensure_ascii=False, indent=2, default=str
                    ),
                }
            ).content
        except Exception:
            response = ""

    if not (response or "").strip():
        if result_source == "split" and (schedule_rows or fare_rows):
            response = _fallback_split_response(
                lang_code,
                fare_mode,
                preferences.get("origin") or state.get("origin") or "?",
                preferences.get("destination") or state.get("destination") or "?",
                schedule_rows,
                fare_rows,
                fare_reversed=bool(state.get("fare_reversed")),
            )
        else:
            response = _fallback_response(
                lang_code,
                fare_mode,
                preferences.get("origin") or state.get("origin") or "?",
                preferences.get("destination") or state.get("destination") or "?",
                trimmed,
                error,
                fare_reversed=bool(state.get("fare_reversed")),
            )

    return {
        **state,
        "response": response,
        "messages": state.get("messages", []) + [AIMessage(content=response)],
    }


_FARE_REVERSED_NOTE = {
    "en": "_(fares are typically the same in either direction)_",
    "si": "_(ගාස්තු සාමාන්‍යයෙන් දෙපැත්තටම සමාන වේ)_",
    "ta": "_(கட்டணம் இரு திசையிலும் பெரும்பாலும் ஒன்றாகவே இருக்கும்)_",
}


def _fare_reversed_suffix(lang_code: str) -> str:
    return " " + _FARE_REVERSED_NOTE.get(lang_code, _FARE_REVERSED_NOTE["en"])


def _fallback_response(
    lang_code: str,
    fare_mode: str,
    origin: str,
    destination: str,
    rows: list[dict[str, Any]],
    error: str | None,
    fare_reversed: bool = False,
) -> str:
    """Deterministic Markdown fallback when the LLM is unavailable or empty."""
    if error and not rows:
        prefix = {
            "en": "Sorry, I couldn't fetch results",
            "si": "කණගාටුයි, ප්‍රතිඵල ලබා ගත නොහැකි විය",
            "ta": "மன்னிக்கவும், முடிவுகளை பெற முடியவில்லை",
        }.get(lang_code, "Sorry, I couldn't fetch results")
        return f"{prefix}: {error}"

    if not rows:
        msg = {
            "en": (
                f"No matching options were found from **{origin}** to **{destination}**. "
                "Try a different departure time, transport type, or fare preference."
            ),
            "si": (
                f"**{origin}** සිට **{destination}** දක්වා ගැලපෙන විකල්ප හමු නොවීය. "
                "වෙනත් වේලාවක්, ප්‍රවාහන වර්ගයක්, හෝ ගාස්තු මනාපයක් උත්සාහ කරන්න."
            ),
            "ta": (
                f"**{origin}** -> **{destination}** ஐக் கொண்டு பொருந்தும் "
                "விருப்பங்கள் எதுவும் இல்லை. வேறு நேரம், போக்குவரத்து, அல்லது "
                "கட்டண விருப்பத்தை முயற்சிக்கவும்."
            ),
        }
        return msg.get(lang_code, msg["en"])

    has_schedule = any(
        r.get("departure_time") or r.get("departure") for r in rows
    )

    # Single-fact case ----------------------------------------------------
    # When upstream dedupe has already collapsed everything to one row, OR the
    # rows differ only in a field the user didn't ask about, give a tight
    # one-sentence answer.
    if len(rows) == 1:
        row = rows[0]
        if not has_schedule and fare_mode != "skip" and row.get("fare") is not None:
            single = {
                "en": f"The fare from **{origin}** to **{destination}** is `LKR {row['fare']}`.",
                "si": f"**{origin}** සිට **{destination}** දක්වා ගාස්තුව **LKR {row['fare']}** වේ.",
                "ta": f"**{origin}** -> **{destination}** கட்டணம் **LKR {row['fare']}** ஆகும்.",
            }
            text = single.get(lang_code, single["en"])
            if fare_reversed:
                text += _fare_reversed_suffix(lang_code)
            return text
        if has_schedule:
            dep = row.get("departure_time") or row.get("departure") or "?"
            arr = row.get("arrival_time") or row.get("arrival") or "?"
            qual_parts = [
                str(v)
                for v in (row.get("route_type"), row.get("service_type"), row.get("route_no"))
                if v
            ]
            qualifiers = f" ({' / '.join(qual_parts)})" if qual_parts else ""
            fare_suffix = ""
            if fare_mode != "skip" and row.get("fare") is not None:
                fare_suffix = f", `LKR {row['fare']}`"
            single_sched = {
                "en": f"From **{origin}** to **{destination}**: **{dep} → {arr}**{qualifiers}{fare_suffix}.",
                "si": f"**{origin}** සිට **{destination}**: **{dep} → {arr}**{qualifiers}{fare_suffix}.",
                "ta": f"**{origin}** -> **{destination}**: **{dep} → {arr}**{qualifiers}{fare_suffix}.",
            }
            return single_sched.get(lang_code, single_sched["en"])

    # Browse / multiple-fact case ---------------------------------------
    intro = {
        "en": f"Here are options from **{origin}** to **{destination}**:",
        "si": f"**{origin}** සිට **{destination}** දක්වා විකල්ප මෙන්න:",
        "ta": f"**{origin}** -> **{destination}** விருப்பங்கள்:",
    }.get(lang_code, f"Here are options from **{origin}** to **{destination}**:")

    bullets: list[str] = []
    for i, row in enumerate(rows):
        if has_schedule:
            dep = row.get("departure_time") or row.get("departure") or "?"
            arr = row.get("arrival_time") or row.get("arrival") or "?"
            qual_parts = [
                str(v)
                for v in (row.get("route_type"), row.get("service_type"), row.get("route_no"))
                if v
            ]
            qualifiers = f" — {' / '.join(qual_parts)}" if qual_parts else ""
            line = f"{i + 1}. **{dep} → {arr}**{qualifiers}"
        else:
            line = f"{i + 1}."

        if fare_mode != "skip" and row.get("fare") is not None:
            tag = ""
            if i == 0 and fare_mode in ("cheapest", "budget"):
                tag = (
                    " _(cheapest)_"
                    if fare_mode == "cheapest"
                    else " _(cheapest within budget)_"
                )
            sep = " — " if has_schedule else " "
            line += f"{sep}`LKR {row['fare']}`{tag}"
        bullets.append(line.strip())

    return "\n".join([intro, *bullets])


def _format_schedule_bullets(rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for i, row in enumerate(rows):
        dep = row.get("departure_time") or row.get("departure") or "?"
        arr = row.get("arrival_time") or row.get("arrival") or "?"
        qual_parts = [
            str(v)
            for v in (row.get("route_type"), row.get("service_type"), row.get("route_no"))
            if v
        ]
        qualifiers = f" — {' / '.join(qual_parts)}" if qual_parts else ""
        out.append(f"{i + 1}. **{dep} → {arr}**{qualifiers}")
    return out


def _format_fare_bullets(rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for i, row in enumerate(rows):
        if row.get("fare") is None:
            continue
        tag = " _(cheapest)_" if i == 0 else ""
        out.append(f"{i + 1}. `LKR {row['fare']}`{tag}")
    return out


def _fallback_split_response(
    lang_code: str,
    fare_mode: str,
    origin: str,
    destination: str,
    schedule_rows: list[dict[str, Any]],
    fare_rows: list[dict[str, Any]],
    fare_reversed: bool = False,
) -> str:
    """Deterministic fallback for the "split" path: schedule + fare separately."""
    intro = {
        "en": (
            f"I couldn't find a single service from **{origin}** to **{destination}** that has both a schedule and a fare on file. "
            "Here's what I do have, listed separately:"
        ),
        "si": (
            f"**{origin}** සිට **{destination}** දක්වා ගමන් වේලාවක් සහ ගාස්තුවක් එකම ලේඛනයක තිබෙන සේවාවක් සොයාගත නොහැකි විය. "
            "මට ඇති දත්ත වෙන වෙනම මෙසේ දැක්වේ:"
        ),
        "ta": (
            f"**{origin}** -> **{destination}** க்கு அட்டவணையும் கட்டணமும் சேர்ந்த ஒரே சேவை கிடைக்கவில்லை. "
            "எனக்குக் கிடைத்த தகவல்களை தனித்தனியாக கீழே தருகிறேன்:"
        ),
    }.get(lang_code, "")

    sched_heading = {"en": "**Schedules**", "si": "**ගමන් වේලාවන්**", "ta": "**அட்டவணைகள்**"}.get(
        lang_code, "**Schedules**"
    )
    fare_heading = {"en": "**Fares**", "si": "**ගාස්තු**", "ta": "**கட்டணங்கள்**"}.get(
        lang_code, "**Fares**"
    )
    none_msg = {
        "en": "_No data available._",
        "si": "_දත්ත නැත._",
        "ta": "_தரவு இல்லை._",
    }.get(lang_code, "_No data available._")
    closing = {
        "en": "Pick a schedule above and I can confirm its specific fare on a follow-up.",
        "si": "ඉහත වේලාවක් තෝරාගෙන, එහි නිශ්චිත ගාස්තුව මට පසුව තහවුරු කළ හැක.",
        "ta": "மேலே உள்ள ஒரு அட்டவணையை தேர்ந்தெடுங்கள், அதன் குறிப்பிட்ட கட்டணத்தை அடுத்த கேள்வியில் உறுதிப்படுத்த முடியும்.",
    }.get(
        lang_code,
        "Pick a schedule above and I can confirm its specific fare on a follow-up.",
    )

    sched_lines = _format_schedule_bullets(schedule_rows) if schedule_rows else [none_msg]
    fare_lines = _format_fare_bullets(fare_rows) if fare_rows else [none_msg]
    if fare_reversed and fare_rows:
        fare_heading = f"{fare_heading} {_FARE_REVERSED_NOTE.get(lang_code, _FARE_REVERSED_NOTE['en'])}"

    parts = [intro, "", sched_heading, *sched_lines, "", fare_heading, *fare_lines]
    if fare_mode != "skip":
        parts.extend(["", closing])
    return "\n".join(p for p in parts if p is not None)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------


def route_intent(state: SmartMoveState) -> Literal["greeting", "transport", "fallback"]:
    return state.get("intent") or "fallback"


def route_missing_info(state: SmartMoveState) -> Literal["follow_up", "continue"]:
    return "follow_up" if (state.get("missing_fields") or []) else "continue"
