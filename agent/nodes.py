"""
agent/nodes.py — AutoStream Lead Agent LangGraph Nodes
Each function accepts AgentState and returns a dict of state updates.
"""

from __future__ import annotations

import re
from typing import Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agent.state import AgentState
from agent.tools import mock_lead_capture, validate_email
from config import GROQ_API_KEY, MODEL_NAME
from rag.retriever import retrieve

# Lazy LLM singleton — created only on first use so that importing this
# module without a real API key (e.g. in unit tests) is safe.

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    """Return the shared ChatGroq instance, creating it on first call."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0.4,
        )
    return _llm


# Helper: get the latest human message text

def _latest_human_text(state: AgentState) -> str:
    """Return the text of the most recent HumanMessage in the conversation."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# Helper: extract lead fields from free-form text

def _extract_fields_from_text(text: str) -> Dict[str, str]:
    """
    Try to extract name, email, and platform from a user message using
    both regex heuristics and positional inference.

    Returns a dict with keys 'name', 'email', 'platform' (values may be empty).
    """
    found: Dict[str, str] = {"name": "", "email": "", "platform": ""}

    # --- Email (most reliable via regex) ---
    email_match = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
    )
    if email_match:
        found["email"] = email_match.group(0).strip()

    # --- Platform (keyword matching) ---
    platforms = [
        "linkedin", "twitter", "instagram", "facebook",
        "youtube", "tiktok", "reddit", "whatsapp", "telegram",
        "x.com", "x ", "web", "google", "website",
    ]
    text_lower = text.lower()
    for p in platforms:
        if p in text_lower:
            found["platform"] = p.replace("x ", "X / Twitter").title()
            break

    # --- Name: whatever remains after removing email / platform references ---
    # Simple heuristic: look for "my name is ...", "i am ...", or "i'm ..." patterns
    name_patterns = [
        r"(?:my name is|i am|i'm|name[:=\s]+)\s*([A-Za-z][A-Za-z\s\-\']{1,40})",
    ]
    for pat in name_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip(".,;")
            # Exclude platform words
            if not any(p in candidate.lower() for p in platforms):
                found["name"] = candidate
                break

    # --- Fallback name: if message has no email, no platform keyword, and
    #     is a short phrase of 1-3 capitalised words, treat it as a name. ---
    if not found["name"]:
        # Remove the email from consideration
        clean = text
        if found["email"]:
            clean = clean.replace(found["email"], "").strip()
        # Remove platform words
        for p in platforms:
            clean = re.sub(re.escape(p), "", clean, flags=re.IGNORECASE)
        clean = clean.strip(" ,.-|/")

        words = clean.split()
        if 1 <= len(words) <= 4 and all(
            re.match(r"^[A-Za-z\-\']+$", w) for w in words
        ):
            # Accept capitalised words as a name
            if any(w[0].isupper() for w in words if w):
                found["name"] = " ".join(words)

    return found


# NODE 1: Intent Classifier

def intent_classifier_node(state: AgentState) -> dict:
    """
    Classify the latest user message into one of three intent labels:
      - "greeting"         → small talk, hello, hi, thanks, etc.
      - "product_inquiry"  → general questions about AutoStream features/pricing
      - "high_intent"      → ready to buy, start trial, sign up, pricing details
    Uses ChatGroq for zero-shot classification.
    """
    user_text = _latest_human_text(state)

    system_prompt = (
        "You are an intent classifier for AutoStream, a SaaS company. "
        "Classify the user's message into EXACTLY ONE of these labels:\n"
        "  - greeting        : greetings, small talk, thanks, goodbyes\n"
        "  - product_inquiry : questions about features, pricing, plans, refunds, support\n"
        "  - high_intent     : user expresses clear interest in signing up, "
        "buying, starting a trial, or wants to be contacted\n\n"
        "Respond with only the label — no explanation, no punctuation."
    )

    response = _get_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text),
        ]
    )

    raw = response.content.strip().lower()

    # Normalise to one of the three valid labels
    if "high" in raw or "intent" in raw or "buy" in raw or "sign" in raw:
        intent = "high_intent"
    elif "product" in raw or "inquiry" in raw or "question" in raw:
        intent = "product_inquiry"
    else:
        intent = "greeting"

    return {"intent": intent}


# NODE 2: RAG Retriever

def rag_retriever_node(state: AgentState) -> dict:
    """
    Retrieve the most relevant knowledge-base chunks for the user's latest message
    and store them in state["context"].
    """
    user_text = _latest_human_text(state)
    context = retrieve(user_text)
    return {"context": context}


# NODE 3: Lead Collection Trigger

def lead_collection_trigger_node(state: AgentState) -> dict:
    """
    Flip the 'collecting_lead' flag to True.
    Also run RAG retrieval so the response generator has relevant context
    even when entering the lead-collection flow.
    """
    user_text = _latest_human_text(state)
    context = retrieve(user_text)
    return {"collecting_lead": True, "context": context}


# NODE 4: Response Generator

def response_generator_node(state: AgentState) -> dict:
    """
    Build the system prompt + full message history and call ChatGroq to
    produce the next assistant message.

    When collecting_lead is True:
      - Asks for ONE missing lead field at a time (name → email → platform).
      - Attempts to extract any fields the user may have already provided.
    """
    lead_info: Dict[str, str] = dict(state.get("lead_info", {}))
    collecting = state.get("collecting_lead", False)
    context = state.get("context", "")

    # --- If in lead collection mode, try extracting fields from latest message ---
    if collecting:
        user_text = _latest_human_text(state)
        extracted = _extract_fields_from_text(user_text)

        for field in ("name", "email", "platform"):
            if not lead_info.get(field) and extracted.get(field):
                # Extra validation for email
                if field == "email" and not validate_email(extracted["email"]):
                    continue
                lead_info[field] = extracted[field]

    # Build system prompt
    persona = (
        "You are Ava, a friendly and knowledgeable sales assistant for AutoStream — "
        "an AI-powered automated video editing SaaS. "
        "You are helpful, concise, and enthusiastic about AutoStream's products. "
        "Always be warm and professional."
    )

    rag_section = ""
    if context:
        rag_section = f"\n\n## Relevant Knowledge Base Context\n{context}"

    lead_section = ""
    if collecting:
        missing = [f for f in ("name", "email", "platform") if not lead_info.get(f)]
        if missing:
            next_field = missing[0]
            field_prompts = {
                "name": (
                    "You are in the process of capturing this user as a lead. "
                    "You still need their FULL NAME. "
                    "Ask naturally for their name — do NOT ask for email or platform yet."
                ),
                "email": (
                    "You already have the user's name. "
                    "Now naturally ask for their EMAIL ADDRESS — do NOT ask for anything else yet."
                ),
                "platform": (
                    "You already have the user's name and email. "
                    "Now ask which social platform or channel they found AutoStream on "
                    "(e.g. LinkedIn, Instagram, YouTube, Google, etc.). "
                    "Do NOT ask for anything else."
                ),
            }
            lead_section = (
                f"\n\n## Lead Collection Status\n"
                f"Collected so far: {lead_info}\n"
                f"Next required field: {next_field}\n"
                f"Instruction: {field_prompts[next_field]}"
            )
        else:
            lead_section = (
                "\n\n## Lead Collection Status\n"
                "All lead fields collected. Acknowledge warmly and tell the user "
                "the team will be in touch very soon."
            )

    system_content = persona + rag_section + lead_section

    # Compose message list for LLM

    messages_for_llm = [SystemMessage(content=system_content)] + list(
        state["messages"]
    )

    response = _get_llm().invoke(messages_for_llm)
    ai_message = AIMessage(content=response.content)

    return {
        "messages": state["messages"] + [ai_message],
        "lead_info": lead_info,
    }


# NODE 5: Lead Capture

def lead_capture_node(state: AgentState) -> dict:
    """
    Called only when all three lead fields (name, email, platform) are filled.
    Invokes mock_lead_capture() and appends a confirmation message to the
    conversation. Sets lead_captured = True to prevent duplicate capture.
    """
    if state.get("lead_captured"):
        return {}  # idempotent guard

    lead_info = state.get("lead_info", {})
    name = lead_info.get("name", "")
    email = lead_info.get("email", "")
    platform = lead_info.get("platform", "")

    confirmation = mock_lead_capture(name=name, email=email, platform=platform)

    # Replace the last AI message with the enriched confirmation
    existing_messages = list(state["messages"])
    existing_messages.append(AIMessage(content=confirmation))

    return {
        "messages": existing_messages,
        "lead_captured": True,
    }
