# config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    BERT_MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    DATA_PATH = Path("C:\Users\91989\OneDrive\Desktop\Sentimental_Analysis\flask_backend")
    MODEL_PATH = Path("models")
    CACHE_DIR = Path("cache")
    AUDIO_PATH = Path("audio")
    
    # Labels for classification
    CONCERN_LABELS = [
        "anxiety", "depression", "trauma", 
        "addiction", "eating_disorder", "other"
    ]
    
    POLARITY_LABELS = ["negative", "neutral", "positive"]
    INTENSITY_LEVELS = ["low", "medium", "high"]