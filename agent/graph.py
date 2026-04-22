"""
agent/graph.py — AutoStream Lead Agent LangGraph StateGraph
Wires all nodes together with conditional routing logic.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    intent_classifier_node,
    lead_capture_node,
    lead_collection_trigger_node,
    rag_retriever_node,
    response_generator_node,
)
from agent.state import AgentState

def _route_after_intent(state: AgentState) -> str:
    """
    Decide where to go after the intent has been classified.

    Rules:
      - "greeting"             → response_generator  (direct reply)
      - "product_inquiry"      → rag_retriever        (fetch context first)
      - "high_intent"
          + not collecting     → lead_collection_trigger
          + already collecting → response_generator   (keep collecting fields)
    """
    intent = state.get("intent", "greeting")
    collecting = state.get("collecting_lead", False)

    if intent == "product_inquiry":
        return "rag_retriever"
    elif intent == "high_intent":
        if collecting:
            return "response_generator"
        return "lead_collection_trigger"
    else:  # greeting or unknown
        return "response_generator"


# Conditional routing after response_generator

def _route_after_response(state: AgentState) -> str:
    """
    After the response is generated, check whether all lead fields have been
    collected and the lead has NOT yet been formally captured.

    If so, route to lead_capture; otherwise finish.
    """
    if state.get("lead_captured"):
        return END  

    lead_info = state.get("lead_info", {})
    all_collected = all(
        lead_info.get(f) for f in ("name", "email", "platform")
    )
    collecting = state.get("collecting_lead", False)

    if collecting and all_collected:
        return "lead_capture"

    return END


# Build the compiled graph

def build_graph() -> StateGraph:
    """Construct and compile the AutoStream LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("intent_classifier", intent_classifier_node)
    builder.add_node("rag_retriever", rag_retriever_node)
    builder.add_node("lead_collection_trigger", lead_collection_trigger_node)
    builder.add_node("response_generator", response_generator_node)
    builder.add_node("lead_capture", lead_capture_node)

    # Entry point
    builder.add_edge(START, "intent_classifier")

    # Intent classifier → branching
    builder.add_conditional_edges(
        "intent_classifier",
        _route_after_intent,
        {
            "rag_retriever": "rag_retriever",
            "lead_collection_trigger": "lead_collection_trigger",
            "response_generator": "response_generator",
        },
    )

    # RAG retriever → response generator
    builder.add_edge("rag_retriever", "response_generator")

    # Lead collection trigger → response generator
    builder.add_edge("lead_collection_trigger", "response_generator")

    # Response generator → conditional (lead_capture or END)
    builder.add_conditional_edges(
        "response_generator",
        _route_after_response,
        {
            "lead_capture": "lead_capture",
            END: END,
        },
    )

    # Lead capture → END
    builder.add_edge("lead_capture", END)

    return builder.compile()


# Singleton compiled graph — import this in main.py
graph = build_graph()
