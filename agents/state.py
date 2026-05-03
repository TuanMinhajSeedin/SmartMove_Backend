"""Shared LangGraph state schema for SmartMove agents."""

from __future__ import annotations

from typing import Any

from typing_extensions import Annotated, TypedDict

from langgraph.graph.message import add_messages


STATE_FIELDS: tuple[str, ...] = (
    "origin",
    "destination",
    "departure_time",
    "date",
    "transport_type",
    "fare",
)


class SmartMoveState(TypedDict, total=False):
    """LangGraph state passed between every node in the SmartMove graph."""

    messages: Annotated[list[Any], add_messages]

    user_query: str
    user_query_original: str | None
    language: str | None
    intent: str | None

    origin: str | None
    destination: str | None
    departure_time: str | None
    date: str | None
    transport_type: str | None
    fare: str | None

    extracted_data: dict[str, Any] | None

    missing_fields: list[str] | None

    cypher_query: str | None
    cypher_query_schedule: str | None
    cypher_query_fare: str | None
    cypher_query_fare_reverse: str | None
    result: str | None
    result_schedule: str | None
    result_fare: str | None
    result_source: str | None
    fare_reversed: bool | None
    response: str | None
    follow_up_question: str | None
