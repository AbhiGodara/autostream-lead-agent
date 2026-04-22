"""
config.py — AutoStream Lead Agent Configuration
Loads environment variables and defines shared constants.
"""

import os
import warnings
from dotenv import load_dotenv

# Load variables from .env file in the project root
load_dotenv()

# Groq API key — must be set in .env
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    warnings.warn(
        "GROQ_API_KEY is not set. "
        "Please copy .env.example to .env and add your Groq key before running the agent.",
        stacklevel=2,
    )

# LLM model identifier
MODEL_NAME: str = "llama-3.3-70b-versatile"
