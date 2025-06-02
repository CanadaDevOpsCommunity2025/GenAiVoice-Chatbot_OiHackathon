import torch
from .models import ModelManager

class EmotionDetector:
    def __init__(self):
        self.model_manager = ModelManager()
        self.tokenizer, self.model, self.labels = self.model_manager.get_emotion_model()

    def detect_emotions(self, text, threshold=0.4):
        """Detects emotions in the given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits)[0]

        detected = [(self.labels[i], probs[i].item()) 
                   for i in range(len(probs)) 
                   if probs[i] > threshold]
        detected.sort(key=lambda x: x[1], reverse=True)

        return [e[0] for e in detected] if detected else ["neutral"] 