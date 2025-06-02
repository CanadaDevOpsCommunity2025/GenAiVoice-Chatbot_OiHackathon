import torch
import requests
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import MODEL_PATH, EMOTION_MODEL_NAME, LLM_PARAMS

class ModelManager:
    def __init__(self):
        self.llm = None
        self.emotion_tokenizer = None
        self.emotion_model = None
        self.emotion_labels = None
        self._load_models()

    def _load_models(self):
        """Load all required models."""
        print("🔄 Loading LLM...")
        self.llm = Llama(
            model_path=MODEL_PATH,
            **LLM_PARAMS
        )
        print("✅ LLM loaded.")

        print("🔄 Loading Emotion Model...")
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
        labels_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
        self.emotion_labels = requests.get(labels_url).text.strip().split("\n")
        print("✅ Emotion Model loaded.")

    def generate_llm_response(self, prompt):
        """Generate response using LLM."""
        return self.llm(
            prompt,
            max_tokens=512,
            temperature=0.5,
            top_p=0.65,
            repeat_penalty=1.1
        )

    def get_emotion_model(self):
        """Get emotion model components."""
        return self.emotion_tokenizer, self.emotion_model, self.emotion_labels 