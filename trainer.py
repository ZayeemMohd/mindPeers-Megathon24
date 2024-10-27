# trainer.py
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import config

class MentalHealthTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            concern_labels = batch['concern_labels'].to(self.device)
            polarity_labels = batch['polarity_labels'].to(self.device)
            intensity_labels = batch['intensity_labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            
            # Calculate losses for each task
            concern_loss = self.criterion(outputs['concern'], concern_labels)
            polarity_loss = self.criterion(outputs['polarity'], polarity_labels)
            intensity_loss = self.criterion(outputs['intensity'], intensity_labels)
            
            # Combined loss
            loss = concern_loss + polarity_loss + intensity_loss
            total_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
        return total_loss / len(train_loader)