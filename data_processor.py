# data_processor.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import librosa
import numpy as np
from typing import Union, List, Dict
import json
import config

class MentalHealthDataset(Dataset):
    def __init__(self, texts: List[str], labels: Dict[str, List[int]], tokenizer: BertTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'concern_labels': torch.tensor(self.labels['concern'][idx]),
            'polarity_labels': torch.tensor(self.labels['polarity'][idx]),
            'intensity_labels': torch.tensor(self.labels['intensity'][idx])
        }

class InputProcessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    
    def process_audio(self, audio_path: Union[str, config.Path]) -> str:
        """Convert audio to text using speech recognition"""
        import speech_recognition as sr
        
        # Load and convert audio to wav
        audio, sr_rate = librosa.load(audio_path)
        wav_path = str(config.CACHE_DIR / "temp.wav")
        librosa.output.write_wav(wav_path, audio, sr_rate)
        
        # Convert to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            
        return text
    
    def prepare_input(self, input_data: Union[str, config.Path], input_type: str = "text") -> dict:
        if input_type == "audio":
            text = self.process_audio(input_data)
        else:
            text = input_data
            
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        return encodings
