from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import psutil
import numpy as np
from transcript import transcript_lines_plain, transcript_lines_speaker, transcript_lock, write_buffer
from speaker import load_known_speakers
from utils import latency_data

known_speakers = load_known_speakers()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/transcript")
async def get_transcript(mode: str = "plain"):
    with transcript_lock:
        if mode == "plain":
            return list(transcript_lines_plain)
        elif mode == "speaker":
            return list(transcript_lines_speaker)
        return []

@app.get("/status")
async def status():
    with transcript_lock:
        return {
            "known_speakers": list(known_speakers.keys()),
            "total_lines_speaker": len(transcript_lines_speaker),
            "total_lines_plain": len(transcript_lines_plain),
            "write_buffer_size": len(write_buffer),
            "active_threads": threading.active_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "avg_latency_sec": round(np.mean(latency_data), 3) if latency_data else None,
            "max_latency_sec": round(np.max(latency_data), 3) if latency_data else None
        }

@app.get("/")
async def index():
    return {"message": "FastAPI is running. Use /transcript and /status endpoints."} 

