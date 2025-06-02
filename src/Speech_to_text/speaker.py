import numpy as np
from resemblyzer import preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from collections import deque
import pickle
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

speaker_embedding_history = deque(maxlen=10)

samplerate = 16000
blocksize = 16000
speaker_threshold = 0.55
MAX_TRANSCRIPT_LINES = 10000

known_speakers = {}
try:
    if os.path.exists("known_speakers.pkl"):
        with open("known_speakers.pkl", "rb") as f:
            known_speakers = pickle.load(f)
except Exception as e:
    print(f"[⚠️ Failed to load known_speakers.pkl] {e}. Regenerating...")
    known_speakers = {}

recent_predictions = []
q = queue.Queue()
transcript_lines_speaker = deque(maxlen=MAX_TRANSCRIPT_LINES)
transcript_lines_plain = deque(maxlen=MAX_TRANSCRIPT_LINES)
transcript_lock = threading.Lock()
write_buffer = []
executor = ThreadPoolExecutor(max_workers=2)
latency_data = deque(maxlen=100)

def cluster_speakers():
    if len(speaker_embedding_history) < 5:
        return None
    embeddings = np.array(speaker_embedding_history)
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.4, linkage="ward")
    labels = clustering_model.fit_predict(embeddings)
    return labels[-1]


def identify_speaker(audio_frames, encoder, known_speakers, speaker_threshold, samplerate):
    wav = preprocess_wav(np.concatenate(audio_frames), source_sr=samplerate)
    if np.mean(np.abs(wav)) < 0.01:
        return "Unknown"
    embedding = encoder.embed_utterance(wav)
    speaker_embedding_history.append(embedding)
    speaker_label = cluster_speakers()
    if speaker_label is not None:
        identity = f"Speaker_{speaker_label + 1}"
        known_speakers[identity] = embedding
    else:
        identity = "Unknown"
    return identity


def load_known_speakers(path="known_speakers.pkl"):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[⚠️ Failed to load {path}] {e}. Regenerating...")
    return {}

def save_known_speakers(known_speakers, path="known_speakers.pkl"):
    try:
        with open(path, "wb") as f:
            pickle.dump(known_speakers, f)
    except Exception as e:
        print(f"[⚠️ Failed to save {path}] {e}") 