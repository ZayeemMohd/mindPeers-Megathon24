import config,models,trainer,data_processor

# main.py
def main():
    # Load and preprocess data
    data = pd.read_csv(config.DATA_PATH / "mental_health_data.csv")
    
    # Initialize tokenizer and create datasets
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'].values,
        {
            'concern': data['concern_labels'].values,
            'polarity': data['polarity_labels'].values,
            'intensity': data['intensity_labels'].values
        },
        test_size=0.2
    )
    
    # Create datasets
    train_dataset = MentalHealthDataset(train_texts, train_labels, tokenizer)
    val_dataset = MentalHealthDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE
    )
    
    # Initialize model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MentalHealthBERT(
        num_concern_labels=len(config.CONCERN_LABELS),
        num_polarity_labels=len(config.POLARITY_LABELS),
        num_intensity_labels=len(config.INTENSITY_LEVELS)
    ).to(device)
    
    trainer = MentalHealthTrainer(model, device)
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}, Loss: {loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), Config.MODEL_PATH / "mental_health_bert.pt")
    
    # Example usage of analyzer
    analyzer = MentalHealthAnalyzer(Config.MODEL_PATH / "mental_health_bert.pt")
    
    # Analyze text input
    text_result = analyzer.analyze("I've been feeling very anxious lately")
    print("Text Analysis Result:", json.dumps(text_result, indent=2))
    
    # Analyze audio input
    audio_result = analyzer.analyze("path/to/audio.wav", input_type="audio")
    print("Audio Analysis Result:", json.dumps(audio_result, indent=2))

if __name__ == "__main__":
    main()