import sounddevice as sd
import numpy as np
import queue
import threading
import time

def select_input_device():
    device = None
    try:
        devices = sd.query_devices()
        for idx, d in enumerate(devices):
            if d['max_input_channels'] >= 1:
                print(f"[🎤 Found input device] ID {idx}: {d['name']}")
                device = idx
                break
        if device is None:
            raise RuntimeError("No suitable input device with 1+ channels.")
    except Exception as e:
        print(f"[❌ Audio Device Error] {e}")
        device = None
    return device

def audio_callback(indata, frames, time, status, q):
    if status:
        print("[⚠️ Audio Status]", status)
    q.put(bytes(indata))

def audio_thread(q, process_chunk, samplerate, blocksize, device):
    buffer = []
    while True:
        try:
            with sd.RawInputStream(samplerate=samplerate, blocksize=blocksize,
                                   device=device, dtype='int16', channels=1,
                                   callback=lambda indata, frames, time, status: audio_callback(indata, frames, time, status, q)):
                while True:
                    data = q.get()
                    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    buffer.append(audio_np)
                    if len(buffer) >= 3:
                        process_chunk(buffer[-3:])
                        buffer = buffer[-1:]
        except Exception as e:
            print(f"[❌ Audio Thread Error] {e} on device={device}")
            time.sleep(3) 