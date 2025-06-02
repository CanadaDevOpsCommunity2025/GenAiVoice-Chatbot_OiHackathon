from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import requests
from TTS.api import TTS
import re
import numpy as np
import io
import time
import uvicorn

# === CONFIGURATION ===
SERVER_URL = "http://172.27.148.150:8989/"
DEFAULT_SAMPLE_RATE = 56050
TIMEOUT_DURATION = 5
MAX_RETRIES = 3

# === Initialize FastAPI App ===
app = FastAPI()

# === Load TTS Model ===
def load_tts_model():
    try:
        print("🔄 Loading TTS model (Jenny)...")
        
        # Dynamically detect GPU availability
        device = "cuda" if TTS.is_cuda_available() else "cpu"
        tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=False)
        tts.to(device)  # Set model to CPU or GPU dynamically
        
        print(f"✅ TTS model loaded successfully on {device}.")
        return tts
    except Exception as e:
        print(f"[❌ TTS Model Load Failed] {e}")
        return None

# Initialize TTS model
tts = load_tts_model()

# === Clean Text Input ===
def clean_reply(text):
    """Removes redundant system messages and detected emotions."""
    return re.sub(r"\(Detected Emotion:.*?\)", "", text).strip()

# === Send Input to External Server with Retry Handling ===
def get_response_from_server(user_input):
    payload = {"input": user_input}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(SERVER_URL + "respond", json=payload, timeout=TIMEOUT_DURATION)
            response.raise_for_status()  # Raise an error if the status isn't 200

            # Decode response properly
            data = response.json()
            if not isinstance(data, dict) or "response" not in data:
                return "[Invalid response from server]"

            return clean_reply(data.get("response", "[No response]"))

        except requests.exceptions.RequestException as e:
            print(f"[🚨 Request Error] Attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            time.sleep(1)  # Shorter retry delay

    return "[Max retries exceeded]"

# === TTS Processing Function ===
def process_tts(text):
    if not tts:
        raise HTTPException(status_code=500, detail="TTS model not available.")
    if not text:
        raise HTTPException(status_code=400, detail="Empty text received for TTS.")
    
    try:
        print(f"🔊 Generating speech for: {text}")
        wav = np.array(tts.tts(text), dtype=np.float32)

        # Convert audio into byte stream
        wav_io = io.BytesIO()
        wav_bytes = (wav * 32767).astype(np.int16).tobytes()
        wav_io.write(wav_bytes)
        wav_io.seek(0)

        return StreamingResponse(wav_io, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Error: {e}")

# === API Endpoint: Text-to-Speech ===
@app.post("/tts")
async def tts_endpoint(text: str = Query(..., description="Text to convert to speech")):
    return process_tts(text)

# === API Endpoint: Get Bot Response and Speak ===
@app.post("/chat")
async def chat(text: str = Query(..., description="User input for chatbot")):
    response_text = get_response_from_server(text)
    return process_tts(response_text)

# === Run the Server & Show Testing Options ===
if __name__ == "__main__":
    print("\n🚀 Starting TTS API server on port 6969...\n")
    
    print("📌 **Test Options:**")
    print("1️⃣ Test TTS with a simple command:")
    print("   curl -X POST 'http://localhost:6969/tts?text=Hello%20world!'")
    
    print("2️⃣ Test Chat + TTS:")
    print("   curl -X POST 'http://localhost:6969/chat?text=How%20are%20you?'")
    
    print("\n📢 Open your browser and go to: http://localhost:6969/docs")
    print("   (This will show the FastAPI interactive documentation.)\n")

    uvicorn.run(app, host="0.0.0.0", port=6969, log_level="info")
