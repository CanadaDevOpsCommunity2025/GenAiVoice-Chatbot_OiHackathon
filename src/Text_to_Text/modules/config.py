import os
from datetime import datetime

# Model Configuration
MODEL_PATH = "E:/Hackathon/mistral-7b-instruct-v0.1.Q4_K_S.gguf"
EMOTION_MODEL_NAME = "monologg/bert-base-cased-goemotions-original"

# API Configuration
TRANSCRIPT_API = "http://localhost:9575/transcript?mode=plain"
TRANSCRIPT_DIR = "/mnt/d/Data_Files/Transcripts"
API_KEY = "AIzaSyD9TCYOyPIe66EirzLsykxjm5CQFybVuHw"
SEARCH_ENGINE_ID = "76495cf7aff3549d3"

# File Configuration
MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(MEMORY_DIR, exist_ok=True)
MEMORY_FILE = os.path.join(MEMORY_DIR, "long_term_memory.pkl")

# Create transcript directory
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
AI_CONVERSATION_PATH = os.path.join(TRANSCRIPT_DIR, f"ai_responses_{timestamp_str}.txt")

# Model Parameters
LLM_PARAMS = {
    "n_gpu_layers": 32,
    "n_ctx": 4096,
    "n_batch": 48,
    "use_mmap": True,
    "use_mlock": True
}

# Response Generation Parameters
RESPONSE_PARAMS = {
    "max_tokens": 512,
    "temperature": 0.5,
    "top_p": 0.65,
    "repeat_penalty": 1.1
}

# Intent Classification Keywords
INTENT_MAP = {
    "summary": ["summarize", "give me a short version"],
    "recommendation": ["suggest", "recommend"],
    "emotion_boost": ["cheer me up", "support"],
    "clarification": ["explain", "what does this mean"],
    "task": ["remind me", "set an alarm", "track my tasks"],
    "live_data": ["what's the weather", "news updates"]
} 