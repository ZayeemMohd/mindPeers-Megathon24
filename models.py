import torch.nn as nn
from transformers import BertModel
import config

class MentalHealthBERT(nn.Module):
    def __init__(self, num_concern_labels, num_polarity_labels, num_intensity_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL_NAME)
        
        hidden_size = self.bert.config.hidden_size
        
        # Multiple heads for different tasks
        self.concern_classifier = nn.Linear(hidden_size, num_concern_labels)
        self.polarity_classifier = nn.Linear(hidden_size, num_polarity_labels)
        self.intensity_classifier = nn.Linear(hidden_size, num_intensity_labels)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        return {
            'concern': self.concern_classifier(pooled_output),
            'polarity': self.polarity_classifier(pooled_output),
            'intensity': self.intensity_classifier(pooled_output)
        }