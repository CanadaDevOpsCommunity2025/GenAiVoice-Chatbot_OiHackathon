from .config import INTENT_MAP

class IntentClassifier:
    def __init__(self):
        self.intent_map = INTENT_MAP

    def classify_intent(self, user_input):
        """Classifies user intent based on input text."""
        user_input = user_input.lower()
        
        for intent, keywords in self.intent_map.items():
            if any(keyword in user_input for keyword in keywords):
                return intent
                
        return "conversation" 