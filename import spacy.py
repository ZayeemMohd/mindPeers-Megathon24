import spacy
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

import re

class MentalHealthAnalyzer:
    def __init__(self):
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.scaler = MinMaxScaler()
        
        # Define mental health keywords and categories
        self.mental_health_categories = {
            'anxiety': ['anxiety', 'worry', 'panic', 'stress', 'nervous'],
            'depression': ['depression', 'sad', 'hopeless', 'unmotivated', 'exhausted'],
            'trauma': ['trauma', 'ptsd', 'flashback', 'nightmare', 'trigger'],
            'addiction': ['addiction', 'substance', 'craving', 'relapse', 'withdrawal'],
            'eating_disorders': ['eating', 'anorexia', 'bulimia', 'binge', 'weight']
        }
        
    def preprocess_text(self, text):
        """Clean and preprocess input text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        doc = self.nlp(text)
        return ' '.join([token.text for token in doc if not token.is_stop])

    def analyze_polarity(self, text):
        """Detect sentiment polarity using transformers"""
        result = self.sentiment_analyzer(text)[0]
        return {
            'label': result['label'],
            'score': result['score']
        }
    
    def extract_keywords(self, text):
        """Extract relevant mental health keywords using NER and custom rules"""
        doc = self.nlp(text)
        keywords = {
            'symptoms': [],
            'emotions': [],
            'triggers': [],
            'coping_mechanisms': []
        }
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['SYMPTOM', 'EMOTION']:
                keywords['symptoms'].append(ent.text)
                
        # Custom keyword extraction based on mental health categories
        for category, terms in self.mental_health_categories.items():
            for term in terms:
                if term in text.lower():
                    keywords['symptoms'].append(term)
                    
        return keywords
    
    def classify_concerns(self, text):
        """Classify the main mental health concerns"""
        concerns = {}
        processed_text = self.preprocess_text(text)
        
        for category, keywords in self.mental_health_categories.items():
            score = sum(1 for keyword in keywords if keyword in processed_text)
            if score > 0:
                concerns[category] = score / len(keywords)
                
        return concerns
    
    def calculate_intensity(self, text):
        """Score the intensity of mental health concerns"""
        # Combine multiple factors for intensity scoring
        factors = {
            'sentiment_strength': abs(self.sentiment_analyzer(text)[0]['score']),
            'keyword_density': len(self.extract_keywords(text)['symptoms']) / len(text.split()),
            'concern_severity': max(self.classify_concerns(text).values(), default=0)
        }
        
        # Calculate weighted average
        weights = {'sentiment_strength': 0.3, 'keyword_density': 0.3, 'concern_severity': 0.4}
        intensity_score = sum(score * weights[factor] for factor, score in factors.items())
        
        return {
            'score': intensity_score,
            'level': self._map_intensity_to_level(intensity_score),
            'factors': factors
        }
    
    def _map_intensity_to_level(self, score):
        """Map numerical intensity score to categorical level"""
        if score < 0.3:
            return 'Low'
        elif score < 0.6:
            return 'Moderate'
        else:
            return 'High'
    
    def analyze_sentiment_shift(self, texts, timestamps):
        """Analyze sentiment changes over time"""
        if len(texts) != len(timestamps):
            raise ValueError("Number of texts and timestamps must match")
            
        timeline_data = []
        for text, timestamp in zip(texts, timestamps):
            sentiment = self.analyze_polarity(text)
            intensity = self.calculate_intensity(text)
            concerns = self.classify_concerns(text)
            
            timeline_data.append({
                'timestamp': timestamp,
                'sentiment': sentiment['label'],
                'sentiment_score': sentiment['score'],
                'intensity': intensity['score'],
                'concerns': concerns
            })
            
        return pd.DataFrame(timeline_data)
    
    def generate_report(self, text, historical_texts=None, historical_timestamps=None):
        """Generate comprehensive analysis report"""
        report = {
            'current_analysis': {
                'polarity': self.analyze_polarity(text),
                'keywords': self.extract_keywords(text),
                'concerns': self.classify_concerns(text),
                'intensity': self.calculate_intensity(text)
            }
        }
        
        if historical_texts and historical_timestamps:
            report['timeline_analysis'] = self.analyze_sentiment_shift(
                historical_texts + [text],
                historical_timestamps + [datetime.now()]
            )
            
        return report

    def process_batch(self, texts):
        """Process multiple texts in batch"""
        results = []
        for text in texts:
            results.append(self.generate_report(text))
        return results

# Example usage
def main():
    analyzer = MentalHealthAnalyzer()
    
    # Single text analysis
    # sample_text = """I've been feeling really anxious lately and having trouble sleeping. 
    #                 Work stress is getting to me and I'm worried about my future."""
    sample_text ="""How do people go through their day-to-day lives without chronically 
                   worrying/obsessing about death."""
    
    result = analyzer.generate_report(sample_text)
    print("Analysis Report:")
    print(f"Polarity: {result['current_analysis']['polarity']}")
    print(f"Keywords: {result['current_analysis']['keywords']}")
    print(f"Concerns: {result['current_analysis']['concerns']}")
    print(f"Intensity: {result['current_analysis']['intensity']}")
    
    # Timeline analysis
    historical_texts = [
        "I'm feeling okay today, just a bit tired.",
        "The anxiety is getting worse, having panic attacks.",
        "Started therapy today, feeling hopeful."
    ]
    historical_timestamps = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 15),
        datetime(2024, 2, 1)
    ]
    
    timeline_result = analyzer.generate_report(
        sample_text,
        historical_texts,
        historical_timestamps
    )
    
    if 'timeline_analysis' in timeline_result:
        print("\nTimeline Analysis:")
        print(timeline_result['timeline_analysis'])

if __name__ == "__main__":
    main()