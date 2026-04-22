# 🎬 AutoStream Lead Agent

An **autonomous Social-to-Lead agentic workflow** built with LangChain + LangGraph for the AutoStream SaaS platform — your AI-powered automated video editing solution.

---

## 📁 Project Structure

```
autostream-lead-agent/
├── main.py                    # CLI chat entry point
├── config.py                  # API key loading & model config
├── requirements.txt
├── .env.example
├── .gitignore
│
├── agent/
│   ├── __init__.py
│   ├── graph.py               # LangGraph StateGraph wiring
│   ├── nodes.py               # All 5 agent nodes
│   ├── state.py               # AgentState TypedDict
│   └── tools.py               # Lead capture + email validator
│
└── rag/
    ├── __init__.py
    ├── knowledge_base.md      # AutoStream product knowledge
    └── retriever.py           # Sentence-transformers RAG retriever
```

---

## 🚀 Local Setup

### Prerequisites
- Conda environment `serviceHive` already created
- Python 3.11

### Steps

```bash
# 1. Activate the conda environment
conda activate serviceHive

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Set up your API key
copy .env.example .env
# Then open .env and replace `your_groq_key_here` with your actual Groq API key

# 4. Run the agent
python main.py
```

> **Get a free Groq API key**: https://console.groq.com

---

## 🏗️ Architecture

### Why LangGraph?

LangGraph models the agent as a **directed state graph** — each conversation turn flows through clearly defined nodes with explicit conditional routing. This makes the logic transparent, testable, and easy to extend. Unlike a simple chain, LangGraph allows **cycles** (e.g., continuing to collect lead fields across multiple turns) and **conditional branching** based on state without tangling the code.

### State Management

A single `AgentState` TypedDict is passed into every node and updated immutably. Key fields:
- **`messages`**: Full conversation history as LangChain `BaseMessage` objects.
- **`intent`**: Classified intent (`greeting`, `product_inquiry`, `high_intent`).
- **`lead_info`**: Dict of `{name, email, platform}` — populated incrementally.
- **`collecting_lead`**: Boolean flag, only set `True` after `high_intent` is detected.
- **`context`**: The RAG-retrieved knowledge base chunks for the current turn.

### How RAG Works

At startup, `rag/retriever.py` reads `knowledge_base.md` and splits it into chunks by `##` section headers. Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` (~80 MB, runs locally, no API cost). On every `product_inquiry` turn, the user's query is embedded and **cosine similarity** is computed against all chunk embeddings. The top-2 most relevant chunks are injected into the system prompt, giving the LLM grounded, up-to-date information about AutoStream's products.

### Graph Flow

```
START
  │
  ▼
intent_classifier
  │
  ├── "greeting"            ──────────────────────▶ response_generator
  │
  ├── "product_inquiry"     ──▶ rag_retriever ────▶ response_generator
  │
  └── "high_intent"
        ├── not collecting  ──▶ lead_collection_trigger ──▶ response_generator
        └── collecting      ──────────────────────────────▶ response_generator
                                                              │
                                          all fields filled? ├── YES ──▶ lead_capture ──▶ END
                                                              └── NO  ───────────────────▶ END
```

---

## 📱 WhatsApp Webhook Integration

To deploy this agent as a WhatsApp chatbot, you can expose it via [Meta's WhatsApp Cloud API](https://developers.facebook.com/docs/whatsapp/cloud-api/) and a **FastAPI** webhook endpoint.

### Architecture

- Each incoming WhatsApp message arrives as a `POST /webhook` request.
- The phone number acts as the **session key** to maintain per-user `AgentState`.
- A simple in-memory dict (or Redis for production) stores states keyed by phone number.

### FastAPI Webhook Snippet

```python
# webhook.py  — drop-in alongside main.py
from fastapi import FastAPI, Request
from langchain_core.messages import HumanMessage
from agent.graph import graph
from agent.state import AgentState
import httpx, os

app = FastAPI()

# In-memory session store (use Redis in production)
sessions: dict[str, AgentState] = {}

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")


def _new_state() -> AgentState:
    return AgentState(
        messages=[], intent="", context="",
        lead_info={"name": "", "email": "", "platform": ""},
        lead_captured=False, collecting_lead=False,
    )


@app.get("/webhook")
async def verify(request: Request):
    """Meta webhook verification handshake."""
    params = dict(request.query_params)
    if params.get("hub.verify_token") == os.getenv("VERIFY_TOKEN"):
        return int(params["hub.challenge"])
    return {"status": "forbidden"}, 403


@app.post("/webhook")
async def receive_message(request: Request):
    """Handle incoming WhatsApp messages."""
    data = await request.json()
    try:
        entry   = data["entry"][0]["changes"][0]["value"]
        message = entry["messages"][0]
        phone   = message["from"]
        text    = message["text"]["body"]
    except (KeyError, IndexError):
        return {"status": "ignored"}

    # Load or create session state
    state = sessions.get(phone, _new_state())
    state["messages"] = state["messages"] + [HumanMessage(content=text)]

    # Run the LangGraph agent
    result = graph.invoke(state)
    sessions[phone] = result

    # Extract latest AI reply
    ai_msgs = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
    reply = ai_msgs[-1].content if ai_msgs else "Sorry, I couldn't process that."

    # Send reply via WhatsApp Cloud API
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "to": phone,
                "text": {"body": reply},
            },
        )

    return {"status": "ok"}
```

**Run it with:** `uvicorn webhook:app --host 0.0.0.0 --port 8000`

---

## 💬 Sample Conversation

```
╔══════════════════════════════════════════════════════════════╗
║          🎬  AutoStream  —  AI Sales Assistant  🎬           ║
╚══════════════════════════════════════════════════════════════╝

Ava: Hi there! 👋 I'm Ava, your AutoStream assistant. How can I help you today?

You: Hey there!

Ava: Hey! 👋 Welcome to AutoStream — where AI takes the hassle out of video
     editing. What can I help you with today?

You: What plans do you offer and how much do they cost?

Ava: Great question! AutoStream offers two plans:

     🎯 **Basic Plan — $29/month**
     • Up to 10 videos/month  •  720p resolution  •  Email support

     🚀 **Pro Plan — $79/month**
     • Unlimited videos  •  4K Ultra HD  •  AI Captions (30+ languages)
     • 24/7 priority support  •  Custom branding  •  Team of up to 10

     All new accounts get a **7-day free trial** of the Pro plan! 🎉

You: This looks great, I'd love to get started!

Ava: Amazing! I'd love to help you get started with AutoStream. 🎬
     Could I grab your full name first?

You: Sure, I'm Alex Johnson

Ava: Great to meet you, Alex! 👋 What's the best email address to reach you at?

You: alex.johnson@example.com

Ava: Perfect! Last question — which platform or channel did you discover
     AutoStream on? (e.g., LinkedIn, Instagram, YouTube, Google...)

You: I found you on LinkedIn

============================================================
✅  LEAD CAPTURED — AutoStream CRM
============================================================
  Name    : Alex Johnson
  Email   : alex.johnson@example.com
  Platform: Linkedin
============================================================

Ava: 🎉 Thanks, Alex! Your details have been saved. Our team will reach out 
     to alex.johnson@example.com shortly to walk you through AutoStream and 
     get you set up. Welcome aboard!
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph (StateGraph) |
| LLM Inference | Groq API — `llama-3.3-70b-versatile` via `langchain-groq` |
| RAG Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| RAG Retrieval | Cosine similarity (NumPy) |
| Config | `python-dotenv` |
| WhatsApp Webhook | FastAPI + Meta Cloud API |

---

## 📝 Notes

- The agent **never asks for lead info** until a `high_intent` turn is detected.
- Fields are collected **one at a time** (name → email → platform).
- If a user provides all three fields in one message, they are extracted automatically.
- All API keys are loaded from `.env` — never hardcoded.
