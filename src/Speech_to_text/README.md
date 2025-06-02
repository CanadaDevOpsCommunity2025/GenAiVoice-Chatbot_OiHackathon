# Oi-Chatbot Speech-to-Text System

> **Note:** This folder was renamed from 'Components' to 'Speech_to_text'.

A modular, real-time speech-to-text (STT) system with speaker diarization, powered by Whisper and resemblyzer, with a FastAPI backend for transcript access and monitoring.

---

## Features
- Real-time audio capture and transcription
- Speaker identification and clustering
- FastAPI server for transcript and status endpoints
- Modular, maintainable codebase
- Performance monitoring
- Automatic transcript saving

---

## Directory Structure
```
src/Speech_to_text/
├── stt.py           # Main entry point
├── audio.py         # Audio device selection and streaming
├── models.py        # Model loading (Whisper, VoiceEncoder)
├── speaker.py       # Speaker identification and clustering
├── transcript.py    # Transcript logging and file writing
├── utils.py         # Performance monitoring and utilities
├── api.py           # FastAPI app and endpoints
└── README.md        # Project documentation
```

---

## Setup Instructions

### 1. **Clone the Repository**
```
git clone <your-repo-url>
cd <your-repo>/src/Speech_to_text
```

### 2. **Install Dependencies**
Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:
```
pip install -r requirements.txt
```

**Example `requirements.txt`:**
```
fastapi
uvicorn
sounddevice
numpy
torch
resemblyzer
scikit-learn
faster-whisper
scipy
psutil
```

> **Note:** You may need to install additional system dependencies for `sounddevice` and `torch` depending on your OS and hardware.

---

## How to Run

From the `src/Speech_to_text` directory, run:
```
python stt.py
```

- The FastAPI server will start at [http://localhost:9575](http://localhost:9575)
- Audio will be captured from the first available input device.
- Transcripts will be saved in `E:/Oi-Chatbot/Transcripts/` by default.

---

## API Endpoints

- `GET /` — Health check, returns a welcome message.
- `GET /transcript?mode=plain|speaker` — Get the transcript (plain text or with speaker labels).
- `GET /status` — Get system status, known speakers, and performance metrics.

---

## Configuration
- **Transcript Directory:** Change `transcript_dir` in `transcript.py` if you want to save transcripts elsewhere.
- **Audio Device:** The first available input device is auto-selected. Modify `select_input_device()` in `audio.py` for custom logic.
- **Model Type:** Whisper model type is set to `medium` in `models.py`. You can change this as needed.

---

## Credits & Acknowledgments
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
- [FastAPI](https://fastapi.tiangolo.com/)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

---

## License
Specify your license here (e.g., MIT, Apache 2.0, etc.) 