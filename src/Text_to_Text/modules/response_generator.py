import concurrent.futures
from .models import ModelManager
from .emotion_detector import EmotionDetector
from .intent_classifier import IntentClassifier
from .memory_manager import MemoryManager
from .api_integration import APIIntegration

class ResponseGenerator:
    def __init__(self):
        self.model_manager = ModelManager()
        self.emotion_detector = EmotionDetector()
        self.intent_classifier = IntentClassifier()
        self.memory_manager = MemoryManager()
        self.api_integration = APIIntegration()

    def generate_response(self, user_input):
        """Generates AI response with context awareness."""
        emotions = self.emotion_detector.detect_emotions(user_input)
        emotion_str = ", ".join(emotions)
        intent = self.intent_classifier.classify_intent(user_input)

        # Handle special intents
        if intent == "task":
            return f"(Task Scheduled) AI will remind you later: {user_input}"
        elif intent == "live_data":
            return self.api_integration.fetch_live_data(user_input)

        # Generate context-aware response
        history_text = self.memory_manager.get_chat_history()
        long_term_memory = self.memory_manager.get_long_term_memory()

        prompt = f"""### Instruction:
You are a helpful AI assistant with emotional intelligence and long-term memory. Adjust your tone based on the user's emotion(s): {emotion_str}.
Recognized user intent: {intent}.
Long-term memory: {long_term_memory}
{history_text}
AI Assistant:"""

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.model_manager.generate_llm_response, prompt)
            result = future.result()

        ai_response = result["choices"][0]["text"].strip()
        self.memory_manager.update_memory(user_input, ai_response)
        
        return f"(Detected Emotion: {emotion_str})\n{ai_response}" 