from datetime import datetime
import threading
import os
import time
from collections import deque

transcript_dir = "E:/Oi-Chatbot/Transcripts"
os.makedirs(transcript_dir, exist_ok=True)
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
speaker_transcript_path = os.path.join(transcript_dir, f"with_speakers_{timestamp_str}.txt")
plain_transcript_path = os.path.join(transcript_dir, f"plain_{timestamp_str}.txt")

MAX_TRANSCRIPT_LINES = 10000
transcript_lines_speaker = deque(maxlen=MAX_TRANSCRIPT_LINES)
transcript_lines_plain = deque(maxlen=MAX_TRANSCRIPT_LINES)
transcript_lock = threading.Lock()
write_buffer = []

def log_transcript(speaker, text):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line_with_speaker = f"[{timestamp}] {speaker}: {text}"
    line_plain = f"{text}"
    print(line_with_speaker)
    with transcript_lock:
        transcript_lines_speaker.append(line_with_speaker)
        transcript_lines_plain.append(line_plain)
        write_buffer.append((line_with_speaker, line_plain))

def periodic_writer():
    while True:
        try:
            time.sleep(5)
            with transcript_lock:
                if write_buffer:
                    with open(speaker_transcript_path, "a", encoding="utf-8") as f1, \
                         open(plain_transcript_path, "a", encoding="utf-8") as f2:
                        print("loading")
                        for speaker_line, plain_line in write_buffer:
                            f1.write(speaker_line + "\n")
                            f2.write(plain_line + "\n")
                    write_buffer.clear()
        except Exception as e:
            print(f"[❌ Writer Thread Error] {e}") 