"""LangGraph builder that wires every node into the SmartMove HITL pipeline."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    cypher_generator_node,
    fallback_node,
    follow_up_question_node,
    greeting_node,
    intent_detection_node,
    language_detection_node,
    llm_extract_node,
    merge_state_node,
    missing_info_validator_node,
    neo4j_query_node,
    response_formatter_node,
    route_intent,
    route_missing_info,
)
from .state import SmartMoveState


def build_app():
    """Return a compiled LangGraph app with an in-memory checkpoint saver."""
    graph = StateGraph(SmartMoveState)
    graph.add_node("language_detection", language_detection_node)
    graph.add_node("intent_detection", intent_detection_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("llm_extract", llm_extract_node)
    graph.add_node("merge_state", merge_state_node)
    graph.add_node("missing_info_validator", missing_info_validator_node)
    graph.add_node("follow_up_question", follow_up_question_node)
    graph.add_node("cypher_generator", cypher_generator_node)
    graph.add_node("neo4j_query", neo4j_query_node)
    graph.add_node("response_formatter", response_formatter_node)

    graph.add_edge(START, "language_detection")
    graph.add_edge("language_detection", "intent_detection")
    graph.add_conditional_edges(
        "intent_detection",
        route_intent,
        {"greeting": "greeting", "transport": "llm_extract", "fallback": "fallback"},
    )
    graph.add_edge("greeting", END)
    graph.add_edge("fallback", END)

    graph.add_edge("llm_extract", "merge_state")
    graph.add_edge("merge_state", "missing_info_validator")
    graph.add_conditional_edges(
        "missing_info_validator",
        route_missing_info,
        {"follow_up": "follow_up_question", "continue": "cypher_generator"},
    )
    graph.add_edge("follow_up_question", "missing_info_validator")

    graph.add_edge("cypher_generator", "neo4j_query")
    graph.add_edge("neo4j_query", "response_formatter")
    graph.add_edge("response_formatter", END)

    return graph.compile(checkpointer=MemorySaver())
