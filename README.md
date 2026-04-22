

https://github.com/user-attachments/assets/cc9cbe6f-8d69-425e-acee-9cab71abe3ff

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
├── assets/                    # Project media and demo videos
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


## 💬 Sample Conversation

```
(E:\alternativeofC\enviorenments\serviceHive) PS D:\ml_udemy\00 ML CODE\Projects\autostream-lead-agent> python main.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████| 103/103 [00:00<00:00, 5592.04it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

╔══════════════════════════════════════════════════════════════╗
║          🎬  AutoStream  —  AI Sales Assistant  🎬          ║
║                                                              ║
║   Powered by LangGraph + Groq (llama-3.3-70b-versatile)      ║
║   Type  'quit'  or  'exit'  to end the session.            ║
╚══════════════════════════════════════════════════════════════╝

Ava(🤖): Hi there! 👋 I'm Ava, your AutoStream assistant. How can I help you today?

You: hii there

Ava(🤖): Hello. It's lovely to meet you. I'm Ava, a sales assistant for AutoStream. How can I assist you today? Are you interested in learning more about our AI-powered automated video editing solutions?

You: what plans do you offer??

Ava(🤖): We have two fantastic plans to choose from at AutoStream. 

Our **Basic Plan** is $29/month, which includes up to 10 videos per month, resolution up to 720p, and standard email support during business hours. It's perfect for individual creators just starting out with automated video editing.

Our **Pro Plan** is $79/month, offering unlimited videos per month, resolution up to 4K Ultra HD, AI captions, and 24/7 priority support. This plan is ideal for professional creators, marketing teams, and high-volume businesses.

Which one of these plans sounds like it might be the best fit for you?

You: can i upgrade my plans from basic to pro

Ava(🤖): You can upgrade from the Basic plan to the Pro plan at any time. It's a quick and easy process that you can do right from your account dashboard. Just log in, go to your account settings, and click on the "Upgrade" button. Your new Pro plan features will be available immediately, and you'll be charged the Pro plan rate at your next billing cycle.

Would you like me to walk you through the steps or answer any questions you might have about the Pro plan features?

You: what is your refund ploicy by the way??

Ava(🤖): Our refund policy is quite straightforward. If you're not satisfied with your purchase, you can request a refund within the first 7 days of buying a plan. To do so, simply reach out to our support team at support@autostream.io, and we'll be happy to assist you.

Please note that we don't issue refunds after the initial 7-day period, so it's essential to review our plans and features carefully before making a purchase. If you have any questions or concerns, I'm here to help you make an informed decision.

You: okay that's great i want to sign up basic plan for my youtube chanel

Ava(🤖): The Basic plan is a great starting point for individual creators like yourself.

Before I can get you signed up, may I please know your full name? I just need to capture this information to complete your registration.

You: Abhishek Godara

Ava(🤖): Nice to have you on board, Abhishek. I already have your name and platform (Youtube) noted. To complete your registration, could you please share your email address with me? This will be used to send you important updates, invoices, and help you access your AutoStream account.

You: abhishekgodara032@gmail.com

============================================================
✅  LEAD CAPTURED — AutoStream CRM
============================================================
  Name    : Abhishek Godara
  Email   : abhishekgodara032@gmail.com
  Platform: Youtube
============================================================


Ava(🤖): 🎉 Thanks, Abhishek Godara! Your details have been saved. Our team will reach out to abhishekgodara032@gmail.com shortly to walk you through AutoStream and get you set up. Welcome aboard!

You: thatnk you

Ava(🤖): You're welcome, Abhishek. It was a pleasure assisting you. If you have any more questions or need help in the future, don't hesitate to reach out. Good luck with your Youtube channel, and I hope AutoStream helps you create some amazing content!

You: exit 

Ava(🤖): Thanks for chatting! Have a great day! 🎬
(E:\alternativeofC\enviorenments\serviceHive) PS D:\ml_udemy\00 ML CODE\Projects\autostream-lead-agent> 
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

---
