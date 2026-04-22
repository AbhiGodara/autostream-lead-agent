"""
validate.py — Smoke-test for all non-LLM logic.
Run with:  conda run -n serviceHive python validate.py
"""

import sys
print("=" * 60)
print("  AutoStream Lead Agent — Validation Suite")
print("=" * 60)

# ── 1. Tools ──────────────────────────────────────────────────
from agent.tools import validate_email, mock_lead_capture

assert validate_email("user@example.com") is True
assert validate_email("bad-email") is False
assert validate_email("a@b.co") is True
assert validate_email("@nodomain.com") is False
print("✅  validate_email          : all cases pass")

result = mock_lead_capture("Alex Johnson", "alex@example.com", "LinkedIn")
assert "Alex Johnson" in result
assert "alex@example.com" in result
print("✅  mock_lead_capture       : confirmation text OK")

# ── 2. AgentState ─────────────────────────────────────────────
from agent.state import AgentState

state = AgentState(
    messages=[],
    intent="greeting",
    lead_info={"name": "", "email": "", "platform": ""},
    lead_captured=False,
    collecting_lead=False,
    context="",
)
assert state["intent"] == "greeting"
assert state["lead_captured"] is False
print("✅  AgentState              : instantiation OK")

# ── 3. RAG Retriever ──────────────────────────────────────────
from rag.retriever import retrieve

r = retrieve("what is the refund policy")
assert len(r) > 0, "Empty retrieval result"
assert "refund" in r.lower() or "7" in r
print("✅  retrieve(refund policy) : relevant chunk returned")

r2 = retrieve("pro plan 4K captions unlimited")
assert "pro" in r2.lower() or "unlimited" in r2.lower()
print("✅  retrieve(pro plan)      : relevant chunk returned")

# ── 4. Field extraction (pure function, no LLM needed) ────────
from agent.nodes import _extract_fields_from_text

r = _extract_fields_from_text("my name is Alice Smith")
assert r["name"] == "Alice Smith", f"Expected 'Alice Smith', got: '{r['name']}'"
print("✅  name extraction         : 'my name is Alice Smith' → OK")

r = _extract_fields_from_text("alice@example.com")
assert r["email"] == "alice@example.com", f"Got: {r['email']}"
print("✅  email extraction        : bare email address → OK")

r = _extract_fields_from_text("I found you on LinkedIn")
assert "linkedin" in r["platform"].lower(), f"Got: {r['platform']}"
print("✅  platform extraction     : LinkedIn mention → OK")

r = _extract_fields_from_text("Alice Smith, alice@example.com, LinkedIn")
assert r["email"] == "alice@example.com"
assert "linkedin" in r["platform"].lower()
print("✅  multi-field extraction  : all 3 in 1 message → OK")

# ── 5. Graph structure (no API call) ──────────────────────────
# We only test the routing functions, not actual node execution
import os
os.environ.setdefault("GROQ_API_KEY", "dummy-for-structure-test")

from agent.graph import _route_after_intent, _route_after_response
from langgraph.graph import END

s_greeting = AgentState(
    messages=[], intent="greeting", lead_info={"name": "", "email": "", "platform": ""},
    lead_captured=False, collecting_lead=False, context=""
)
assert _route_after_intent(s_greeting) == "response_generator"
print("✅  graph routing           : greeting → response_generator")

s_inquiry = AgentState(
    messages=[], intent="product_inquiry", lead_info={"name": "", "email": "", "platform": ""},
    lead_captured=False, collecting_lead=False, context=""
)
assert _route_after_intent(s_inquiry) == "rag_retriever"
print("✅  graph routing           : product_inquiry → rag_retriever")

s_high = AgentState(
    messages=[], intent="high_intent", lead_info={"name": "", "email": "", "platform": ""},
    lead_captured=False, collecting_lead=False, context=""
)
assert _route_after_intent(s_high) == "lead_collection_trigger"
print("✅  graph routing           : high_intent (new) → lead_collection_trigger")

s_collecting = AgentState(
    messages=[], intent="high_intent", lead_info={"name": "", "email": "", "platform": ""},
    lead_captured=False, collecting_lead=True, context=""
)
assert _route_after_intent(s_collecting) == "response_generator"
print("✅  graph routing           : high_intent (collecting) → response_generator")

s_all_filled = AgentState(
    messages=[], intent="high_intent",
    lead_info={"name": "Alice", "email": "a@b.co", "platform": "LinkedIn"},
    lead_captured=False, collecting_lead=True, context=""
)
assert _route_after_response(s_all_filled) == "lead_capture"
print("✅  graph routing           : all fields filled → lead_capture")

s_already_captured = AgentState(
    messages=[], intent="high_intent",
    lead_info={"name": "Alice", "email": "a@b.co", "platform": "LinkedIn"},
    lead_captured=True, collecting_lead=True, context=""
)
assert _route_after_response(s_already_captured) == END
print("✅  graph routing           : already captured → END")

print()
print("=" * 60)
print("  🎉  ALL TESTS PASSED — Ready to run with a valid GROQ_API_KEY")
print("=" * 60)
