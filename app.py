# config.py
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class Config:
    BERT_MODEL_NAME: str = "bert-base-uncased"
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 16
    EPOCHS: int = 5
    LEARNING_RATE: float = 2e-5
    
    # Directory paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"
    
    # Labels
    CONCERN_LABELS: List[str] = [
        "anxiety", "depression", "trauma", 
        "addiction", "eating_disorder", "other"
    ]
    POLARITY_LABELS: List[str] = ["negative", "neutral", "positive"]
    INTENSITY_LEVELS: List[str] = ["low", "medium", "high"]
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

# dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import Dict, List
import numpy as np

class MentalHealthDataset(Dataset):
    def __init__(self, texts: List[str], labels: Dict[str, np.ndarray], tokenizer: BertTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'concern_labels': torch.tensor(self.labels['concern'][idx], dtype=torch.float),
            'polarity_labels': torch.tensor(self.labels['polarity'][idx], dtype=torch.long),
            'intensity_labels': torch.tensor(self.labels['intensity'][idx], dtype=torch.long)
        }

# model.py
import torch
import torch.nn as nn
from transformers import BertModel
from typing import Dict

class MentalHealthBERT(nn.Module):
    def __init__(self, num_concern_labels: int, num_polarity_labels: int, num_intensity_labels: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
        hidden_size = self.bert.config.hidden_size
        
        # Task-specific layers
        self.concern_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_concern_labels),
            nn.Sigmoid()
        )
        
        self.polarity_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_polarity_labels)
        )
        
        self.intensity_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_intensity_labels)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        return {
            'concern': self.concern_classifier(pooled_output),
            'polarity': self.polarity_classifier(pooled_output),
            'intensity': self.intensity_classifier(pooled_output)
        }

# trainer.py
from torch.optim import AdamW
from torch.nn import BCELoss, CrossEntropyLoss
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthTrainer:
    def __init__(self, model: MentalHealthBERT, device: torch.device):
        self.model = model
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        
        # Loss functions for different tasks
        self.concern_criterion = BCELoss()
        self.polarity_criterion = CrossEntropyLoss()
        self.intensity_criterion = CrossEntropyLoss()

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            concern_labels = batch['concern_labels'].to(self.device)
            polarity_labels = batch['polarity_labels'].to(self.device)
            intensity_labels = batch['intensity_labels'].to(self.device)

            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            
            # Calculate losses
            concern_loss = self.concern_criterion(outputs['concern'], concern_labels)
            polarity_loss = self.polarity_criterion(outputs['polarity'], polarity_labels)
            intensity_loss = self.intensity_criterion(outputs['intensity'], intensity_labels)
            
            # Combined loss
            loss = concern_loss + polarity_loss + intensity_loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(train_loader)

# analyzer.py
import json
from typing import Dict, Union

class MentalHealthAnalyzer:
    def __init__(self, model_path: Union[str, Path]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        
        self.model = MentalHealthBERT(
            num_concern_labels=len(Config.CONCERN_LABELS),
            num_polarity_labels=len(Config.POLARITY_LABELS),
            num_intensity_labels=len(Config.INTENSITY_LEVELS)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def analyze_text(self, text: str) -> Dict:
        # Tokenize input text
        encoding = self.tokenizer(
            text,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move tensors to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        # Process and format results
        results = {
            'concerns': {
                label: float(score)
                for label, score in zip(Config.CONCERN_LABELS, outputs['concern'][0])
            },
            'polarity': {
                label: float(score)
                for label, score in zip(Config.POLARITY_LABELS, 
                                      torch.softmax(outputs['polarity'][0], dim=0))
            },
            'intensity': {
                label: float(score)
                for label, score in zip(Config.INTENSITY_LEVELS, 
                                      torch.softmax(outputs['intensity'][0], dim=0))
            }
        }

        return results

# main.py
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_data(csv_path: Union[str, Path]) -> tuple:
    """Prepare data for training"""
    data = pd.read_csv(csv_path)
    
    # Convert labels to appropriate format
    concern_labels = np.array(data[Config.CONCERN_LABELS].values)
    polarity_labels = np.array(data['polarity'].map(
        {label: idx for idx, label in enumerate(Config.POLARITY_LABELS)}
    ))
    intensity_labels = np.array(data['intensity'].map(
        {label: idx for idx, label in enumerate(Config.INTENSITY_LEVELS)}
    ))
    
    return train_test_split(
        data['text'].values,
        {
            'concern': concern_labels,
            'polarity': polarity_labels,
            'intensity': intensity_labels
        },
        test_size=0.2,
        random_state=42
    )

def main():
    logger.info("Starting mental health text analysis pipeline")
    
    # Prepare data
    train_texts, val_texts, train_labels, val_labels = prepare_data(
        Config.DATA_DIR / "mental_health_dataset_Sheet1.csv"
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    
    # Create datasets
    train_dataset = MentalHealthDataset(train_texts, train_labels, tokenizer)
    val_dataset = MentalHealthDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    
    # Initialize model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = MentalHealthBERT(
        num_concern_labels=len(Config.CONCERN_LABELS),
        num_polarity_labels=len(Config.POLARITY_LABELS),
        num_intensity_labels=len(Config.INTENSITY_LEVELS)
    ).to(device)
    
    trainer = MentalHealthTrainer(model, device)
    
    # Training loop
    logger.info("Starting training")
    for epoch in range(Config.EPOCHS):
        loss = trainer.train_epoch(train_loader)
        logger.info(f"Epoch {epoch + 1}/{Config.EPOCHS}, Loss: {loss:.4f}")
    
    # Save model
    model_path = Config.MODEL_DIR / "mental_health_bert.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Example usage
    analyzer = MentalHealthAnalyzer(model_path)
    test_text = "I've been feeling very anxious and stressed lately, having trouble sleeping."
    result = analyzer.analyze_text(test_text)
    print("\nAnalysis Result:", json.dumps(result, indent=2))

if __name__ == "__main__":
    main()