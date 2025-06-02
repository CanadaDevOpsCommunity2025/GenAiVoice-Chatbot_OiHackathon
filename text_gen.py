import os, torch, time, requests, pickle
from datetime import datetime
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import deque
import asyncio
import concurrent.futures  # Parallel execution support
from fastapi import BackgroundTasks

from pydantic import BaseModel

# Define UserInput model
class UserInput(BaseModel):
    input: str

# === CONFIG ===
MODEL_PATH = "/mnt/d/WSL/Ubuntu/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_S.gguf"
TRANSCRIPT_API = "http://localhost:9575/transcript?mode=plain"
TRANSCRIPT_DIR = "/mnt/d/Data_Files/Transcripts"
MEMORY_FILE = "long_term_memory.pkl"  # Persistent memory storage
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ai_conversation_path = os.path.join(TRANSCRIPT_DIR, f"ai_responses_{timestamp_str}.txt")

# === LOAD MODELS ===
print("🔄 Loading LLM...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=32,
    n_ctx=4096,  # Optimized for memory handling
    n_batch=48,  # Balanced batch processing
    use_mmap=True,
    use_mlock=True
)
print("✅ LLM loaded.")

emotion_model_name = "monologg/bert-base-cased-goemotions-original"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
labels_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
emotion_labels = requests.get(labels_url).text.strip().split("\n")

# === STATE ===
chat_history = deque(maxlen=100)  # Extended memory storage
long_term_memory = {}  # Dictionary-based persistent memory
last_seen_line = ""
mode = "text"
app = FastAPI()

# === LOAD MEMORY ===
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "rb") as f:
            long_term_memory = pickle.load(f)
    except Exception as e:
        print(f"[⚠️ Memory Load Error] {e}")
        long_term_memory = {}

# === CONFIG ===
API_KEY = "AIzaSyD9TCYOyPIe66EirzLsykxjm5CQFybVuHw"  # Replace with your API key
SEARCH_ENGINE_ID = "76495cf7aff3549d3"  # Replace with your Search Engine ID

import os
import requests

# === CONFIG ===
API_KEY = os.getenv("AIzaSyD9TCYOyPIe66EirzLsykxjm5CQFybVuHw")  # Store API key securely
SEARCH_ENGINE_ID = os.getenv("76495cf7aff3549d3")  # Store CX value securely

def google_search(query, num_results=5):
    """Fetches search results using Google Custom Search API."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "num": num_results
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises error if request fails
        data = response.json()

        search_results = []
        if "items" in data:
            for item in data["items"]:
                search_results.append({
                    "title": item["title"],
                    "link": item["link"],
                    "snippet": item["snippet"]
                })
        return search_results

    except requests.exceptions.RequestException as e:
        return f"(Error fetching search results: {e})"

# === EMOTION DETECTION ===
def detect_emotions(text, threshold=0.4):
    """Detects emotions for response adaptation."""
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        probs = torch.sigmoid(logits)[0]

    detected = [(emotion_labels[i], probs[i].item()) for i in range(len(probs)) if probs[i] > threshold]
    detected.sort(key=lambda x: x[1], reverse=True)

    return [e[0] for e in detected] if detected else ["neutral"]

# === INTENT CLASSIFICATION ===
def classify_intent(user_input):
    """Classifies user intent for task execution."""
    intent_map = {
        "summary": ["summarize", "give me a short version"],
        "recommendation": ["suggest", "recommend"],
        "emotion_boost": ["cheer me up", "support"],
        "clarification": ["explain", "what does this mean"],
        "task": ["remind me", "set an alarm", "track my tasks"],
        "live_data": ["what's the weather", "news updates"]
    }

    for intent, keywords in intent_map.items():
        if any(keyword in user_input.lower() for keyword in keywords):
            return intent
    return "conversation"

# === MEMORY MANAGEMENT ===
def update_memory(user_input, ai_response):
    """Stores relevant interactions for future recall."""
    long_term_memory["last_user_request"] = user_input
    long_term_memory["last_ai_response"] = ai_response
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(long_term_memory, f)

# === API INTEGRATION ===
import requests

OPENWEATHER_API_KEY = "fcec17b33daa983f43f231dfdc9c7dab"

def get_weather(city="Ottawa"):
    """Fetches real-time weather data from OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return f"🌦️ {data['weather'][0]['description']}, Temp: {data['main']['temp']}°C"
    except Exception as e:
        return f"(Error fetching weather: {e})"
# === AI RESPONSE GENERATION ===

def generate_response(user_input):
    """Generates AI response with memory, emotional context-awareness, and automation."""
    emotions = detect_emotions(user_input)
    emotion_str = ", ".join(emotions)
    intent = classify_intent(user_input)

    # Avoid appending the same user input repeatedly
    if len(chat_history) == 0 or chat_history[-1] != f"User: {user_input}":
        chat_history.append(f"User: {user_input}")

    # Generate cleaned-up conversation history (remove consecutive duplicates)
    cleaned_history = []
    last_line = None
    for line in chat_history:
        if line != last_line:
            cleaned_history.append(line)
            last_line = line
    history_text = "\n".join(cleaned_history)

    # Handle task automation & live data retrieval
    if intent == "task":
        return f"(Task Scheduled) AI will remind you later: {user_input}"
    elif intent == "live_data" and "weather" in user_input.lower():
        return get_weather("Ottawa")

    # Compose the prompt
    prompt = f"""### Instruction:
You are a helpful AI assistant with emotional intelligence and long-term memory.
Adjust your tone based on the user's emotion(s): {emotion_str}.
Recognized user intent: {intent}.
Long-term memory: {long_term_memory}
{history_text}
AI Assistant:"""

    # Generate response using the LLM
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            llm,
            prompt,
            max_tokens=512,
            temperature=0.5,
            top_p=0.65,
            repeat_penalty=1.1
        )
        result = future.result()

    ai_response = result["choices"][0]["text"].strip()

    # Avoid appending the same AI response repeatedly
    if len(chat_history) == 0 or chat_history[-1] != f"AI: {ai_response}":
        chat_history.append(f"AI: {ai_response}")

    update_memory(user_input, ai_response)
    return f"(Detected Emotion: {emotion_str})\n{ai_response}"

TASKS = []

@app.post("/set_task")
async def set_task(data: UserInput, background_tasks: BackgroundTasks):
    """Schedules tasks asynchronously."""
    task_description = data.input
    background_tasks.add_task(schedule_task, task_description)
    return {"status": "Task scheduled!"}

def schedule_task(task_description):
    """Executes the scheduled task asynchronously."""
    time.sleep(5)  # Simulating delay before reminder
    print(f"🔔 Reminder: {task_description}")

# === REQUEST MODEL ===
class UserInput(BaseModel):
    input: str

# === API ENDPOINTS ===
@app.post("/toggle_mode")
async def toggle_mode():
    """Switch between text and voice modes."""
    global mode
    mode = "voice" if mode == "text" else "text"
    return {"mode": mode}

@app.post("/respond")
async def respond_to_input(data: UserInput):
    """Processes user input and returns AI-generated response asynchronously."""
    user_input = data.input
    ai_response = generate_response(user_input)
    return {"response": ai_response}

# === MAIN ENTRY POINT ===
if __name__ == "__main__":
    import uvicorn
    print("🚀 AI Assistant running with FastAPI...")
    uvicorn.run(app, host="0.0.0.0", port=8989, log_level="info")
