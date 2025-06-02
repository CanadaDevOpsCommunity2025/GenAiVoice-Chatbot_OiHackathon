import pickle
import os
from collections import deque
from .config import MEMORY_FILE

class MemoryManager:
    def __init__(self, max_history=100):
        self.chat_history = deque(maxlen=max_history)
        self.long_term_memory = self._load_memory()

    def _load_memory(self):
        """Load long-term memory from file."""
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, "rb") as f:
                    return pickle.load(f)
            else:
                print(f"[ℹ️ No existing memory file found at {MEMORY_FILE}. Creating new memory.")
                return {}
        except Exception as e:
            print(f"[⚠️ Memory Load Error] {e}")
            return {}

    def update_memory(self, user_input, ai_response):
        """Update both chat history and long-term memory."""
        self.chat_history.append(f"User: {user_input}")
        self.chat_history.append(f"AI: {ai_response}")
        
        self.long_term_memory["last_user_request"] = user_input
        self.long_term_memory["last_ai_response"] = ai_response
        
        self._save_memory()

    def _save_memory(self):
        """Save long-term memory to file."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
            
            with open(MEMORY_FILE, "wb") as f:
                pickle.dump(self.long_term_memory, f)
            print(f"[✅ Memory saved successfully to {MEMORY_FILE}]")
        except Exception as e:
            print(f"[⚠️ Memory Save Error] {e}")

    def get_chat_history(self):
        """Get formatted chat history."""
        return "\n".join(self.chat_history)

    def get_long_term_memory(self):
        """Get long-term memory."""
        return self.long_term_memory 