import psutil
import time
from collections import deque

latency_data = deque(maxlen=100)

def performance_monitor():
    while True:
        cpu = psutil.cpu_percent(interval=5)
        mem = psutil.virtual_memory().percent
        print(f"[📊 Performance] CPU: {cpu:.1f}% | Memory: {mem:.1f}%") 