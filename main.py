"""
main.py — AutoStream Lead Agent CLI Entry Point
Runs an interactive chat loop in the terminal.
"""

from __future__ import annotations

import sys

# ── Early guard: check for GROQ API key before loading heavy modules ─────────
from config import GROQ_API_KEY

if not GROQ_API_KEY:
    print("\n❌  ERROR: GROQ_API_KEY is not set.")
    print("   1. Copy '.env.example' to '.env'")
    print("   2. Replace 'your_groq_key_here' with your actual Groq API key")
    print("   3. Get a free key at: https://console.groq.com\n")
    sys.exit(1)

from langchain_core.messages import HumanMessage

from agent.graph import graph
from agent.state import AgentState


# Welcome banner

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🎬  AutoStream  —  AI Sales Assistant  🎬          ║
║                                                              ║
║   Powered by LangGraph + Groq (llama-3.3-70b-versatile)      ║
║   Type  'quit'  or  'exit'  to end the session.            ║
╚══════════════════════════════════════════════════════════════╝
"""


# Initial state factory

def _initial_state() -> AgentState:
    return AgentState(
        messages=[],
        intent="",
        lead_info={"name": "", "email": "", "platform": ""},
        lead_captured=False,
        collecting_lead=False,
        context="",
    )


# Main chat loop

def main() -> None:
    print(BANNER)
    print("Ava(🤖): Hi there! 👋 I'm Ava, your AutoStream assistant. "
          "How can I help you today?\n")

    state: AgentState = _initial_state()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAva(🤖): Goodbye! 👋")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("\nAva(🤖): Thanks for chatting! Have a great day! 🎬")
            break

        # Append the new human message to the conversation
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run the graph
        try:
            result: AgentState = graph.invoke(state)
        except Exception as exc:
            print(f"\n[Error invoking graph: {exc}]\n")
            continue

        # Update state for the next turn
        state = result

        # Print the latest AI message
        ai_messages = [
            msg for msg in state["messages"]
            if hasattr(msg, "type") and msg.type == "ai"
        ]
        if ai_messages:
            print(f"\nAva(🤖): {ai_messages[-1].content}\n")


if __name__ == "__main__":
    main()
