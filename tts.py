import requests
from TTS.api import TTS
import re
import sounddevice as sd
import numpy as np
import time

# === Config ===
SERVER_URL = "http://172.27.148.150:8989/"
DEFAULT_SAMPLE_RATE = 56050
TIMEOUT_DURATION = 5  # Reduced timeout
MAX_RETRIES = 3  # Retry mechanism for server response

# === Load TTS Model Once ===
def load_tts_model():
    try:
        print("🔄 Loading TTS model (Jenny)...")
        return TTS(model_name="tts_models/en/jenny/jenny", progress_bar=False, gpu=True)
    except Exception as e:
        print(f"[❌ TTS Model Load Failed] {e}")
        return None

tts = load_tts_model()

# === Send Input to Server with Retry Mechanism ===
def get_response_from_server(user_input):
    payload = {"input": user_input}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(SERVER_URL + "respond", json=payload, timeout=TIMEOUT_DURATION)
            if response.status_code == 200:
                full_reply = response.json().get("response", "[No response]")
                print("🤖 Bot Response:", full_reply)
                return clean_reply(full_reply)
            else:
                print(f"[❌ Server Error] Code {response.status_code}")
                return "[Failed to get response]"
        except requests.exceptions.RequestException as e:
            print(f"[🚨 Request Error] Attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            time.sleep(1)  # Shorter retry delay

    return "[Max retries exceeded]"

# === Clean Output for TTS ===
def clean_reply(text):
    """Removes redundant system messages and detected emotions."""
    text = re.sub(r"\(Detected Emotion:.*?\)", "", text)
    lines = text.strip().split("\n")
    
    # Remove duplicate lines to prevent repeated speech generation
    seen = set()
    cleaned_lines = [line for line in lines if line.strip() and line not in seen and not line.startswith("User 1:")]
    seen.update(cleaned_lines)
    
    return " ".join(cleaned_lines)

# === Speak Using TTS (Parallel Processing) ===
def speak(text, sample_rate=DEFAULT_SAMPLE_RATE):
    if not tts:
        print("[⚠️ TTS not available]")
        return
    if not text:
        print("[⚠️ Empty text received for TTS]")
        return

    try:
        print("🔊 Speaking immediately...")
        wav = np.array(tts.tts(text), dtype=np.float32)

        # Non-blocking playback for responsiveness
        sd.play(wav, samplerate=int(sample_rate), blocking=False)
    except Exception as e:
        print(f"[❌ TTS Error] {e}")

# === CLI Entry Point ===
if __name__ == "__main__":
    try:
        print("🎤 Type your message below. Type 'exit' to quit.\n")
        while True:
            user_input = input("🧠 You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("👋 Exiting.")
                break
            if user_input:
                reply = get_response_from_server(user_input)
                if reply:
                    speak(reply)  # Speak immediately
    except KeyboardInterrupt:
        print("\n👋 Exiting on Ctrl+C.")
    finally:
        print("🛑 Cleaning up resources...")
        sd.stop()  # Stop any ongoing audio playback
