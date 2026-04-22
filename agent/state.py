"""
agent/state.py — AutoStream Lead Agent State Definition
Defines the AgentState TypedDict shared across all LangGraph nodes.
"""

from __future__ import annotations

from typing import Dict, List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state passed between every node in the LangGraph StateGraph.

    Fields
    ------
    messages : List[BaseMessage]
        Full conversation history (HumanMessage + AIMessage objects).
    intent : str
        Classified intent of the latest user message.
        One of: "greeting", "product_inquiry", "high_intent"
    lead_info : dict
        Captured lead data.  Keys: "name", "email", "platform".
        Values are empty strings until collected.
    lead_captured : bool
        True once mock_lead_capture() has been successfully called.
    collecting_lead : bool
        True once the lead-collection flow has been triggered.
    context : str
        RAG-retrieved context string from the knowledge base.
    """

    messages: List[BaseMessage]
    intent: str
    lead_info: Dict[str, str]
    lead_captured: bool
    collecting_lead: bool
    context: str
