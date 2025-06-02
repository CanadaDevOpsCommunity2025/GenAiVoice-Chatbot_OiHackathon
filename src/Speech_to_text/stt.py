"""
Main entry point for the Oi-Chatbot Speech-to-Text system.

- Starts background audio, transcript writer, and performance monitor threads
- Launches the FastAPI server for API access
- NOTE: This folder was renamed from 'Components' to 'Speech_to_text'.
"""

import numpy as np
import io
import threading
import uvicorn
from audio import select_input_device, audio_thread
from models import get_models
from speaker import identify_speaker, load_known_speakers
from transcript import log_transcript, periodic_writer
from utils import performance_monitor
from api import app
import queue
import time
from scipy.io.wavfile import write
from utils import latency_data

# === CONFIG ===
samplerate = 16000
blocksize = 16000
speaker_threshold = 0.55

import torch
from resemblyzer import VoiceEncoder
from faster_whisper import WhisperModel

def get_models():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VoiceEncoder().to(device_type)
    whisper_model = WhisperModel("medium", device=device_type, compute_type="float16" if device_type == "cuda" else "int8")
    return encoder, whisper_model, device_type 

# === MAIN ENTRY POINT ===
def process_chunk(audio_frames):
    import numpy as np
    import io
    from scipy.io.wavfile import write
    import time
    from utils import latency_data
    from models import get_models
    from speaker import identify_speaker, load_known_speakers
    from transcript import log_transcript

    start = time.time()
    print("[🧠 Processing audio chunk...]")

    full_chunk = np.concatenate(audio_frames)
    avg_volume = np.mean(np.abs(full_chunk))
    print(f"[📊 Chunk Stats] Mean volume: {avg_volume:.6f}")

    # Try lowering the threshold if needed
    if avg_volume < 0.001:
        print("[🔇 Silence skipped]")
        return

    # Convert audio chunk to WAV in memory
    wav_io = io.BytesIO()
    write(wav_io, 16000, (full_chunk * 32767).astype(np.int16))
    wav_io.seek(0)

    # Load models locally to avoid scope issues
    encoder, whisper_model, _ = get_models()
    known_speakers = load_known_speakers()

    try:
        segments, _ = whisper_model.transcribe(
            wav_io,
            vad_filter=True,
            vad_parameters={"threshold": 0.6, "min_silence_duration_ms": 300}
        )
        print("[🔍 Raw Whisper Output]:", segments)

        for segment in segments:
            text = segment.text.strip()
            if text:
                print(f"[📄 Transcribed] {text}")
                speaker = identify_speaker(audio_frames, encoder, known_speakers, 0.55, 16000)
                log_transcript(speaker, text)
    except Exception as e:
        print(f"[❌ Whisper Error] {e}")

    end = time.time()
    latency_data.append(end - start)


def start_background_tasks():
    q = queue.Queue()
    device = select_input_device()
    print(f"🎧 Using device: {device}")
    print("🚀 FastAPI server running at http://localhost:9575")
    threading.Thread(target=audio_thread, args=(q, process_chunk, samplerate, blocksize, device), daemon=True).start()
    threading.Thread(target=periodic_writer, daemon=True).start()
    threading.Thread(target=performance_monitor, daemon=True).start()

if __name__ == "__main__":
    start_background_tasks()
    uvicorn.run(app, host="0.0.0.0", port=9575, log_level="info")
