import torch
import config ,models,trainer,data_processor

class MentalHealthAnalyzer:
    def __init__(self, model_path: config.Path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = trainer.MentalHealthBERT(
            num_concern_labels=len(config.CONCERN_LABELS),
            num_polarity_labels=len(config.POLARITY_LABELS),
            num_intensity_labels=len(config.INTENSITY_LEVELS)
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        self.input_processor = models.InputProcessor()
        
    def analyze(self, input_data: models.Union[str, config.Path], input_type: str = "text") -> models.Dict:
        """Analyze input data and return comprehensive analysis"""
        # Process input
        encodings = self.input_processor.prepare_input(input_data, input_type)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(
                encodings['input_ids'].to(self.device),
                encodings['attention_mask'].to(self.device)
            )
            
        # Convert outputs to predictions
        predictions = {
            'concern': torch.sigmoid(outputs['concern']).cpu().numpy(),
            'polarity': torch.softmax(outputs['polarity'], dim=1).cpu().numpy(),
            'intensity': torch.softmax(outputs['intensity'], dim=1).cpu().numpy()
        }
        
        # Format results
        results = {
            'concerns': {
                label: float(score)
                for label, score in zip(config.CONCERN_LABELS, predictions['concern'][0])
            },
            'polarity': {
                label: float(score)
                for label, score in zip(config.POLARITY_LABELS, predictions['polarity'][0])
            },
            'intensity': {
                label: float(score)
                for label, score in zip(config.INTENSITY_LEVELS, predictions['intensity'][0])
            }
        }
        
        return results