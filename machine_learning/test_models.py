import os
import joblib
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from utils.preprocessing import preprocess_text
from utils.text_to_bert_embeddings import text_to_bert_embeddings

# Filepaths
TRAINED_MODELS_DIR = "./trained_models/"

# Define label mapping
label_mapping = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral (Non-Offensive)"
}

# Confidence threshold
THRESHOLD = 0.7  # Set a threshold for confidence

# Initialize BERT model and tokenizer
print("Initializing BERT tokenizer and model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)
bert_model.eval()

# Hardcoded model selection
selected_model_name = "Logistic Regression"
selected_model_file = os.path.join(TRAINED_MODELS_DIR, "logistic_regression.pkl")

# Load the selected model
print(f"\nLoading hardcoded model: {selected_model_name} from {selected_model_file}...")
model = joblib.load(selected_model_file)

# Test the selected model with manual input
while True:
    input_text = input("\nEnter text to classify (or type 'exit' to quit): ").strip()
    if input_text.lower() == "exit":
        print(f"Exiting testing for {selected_model_name}...")
        break

    # Convert input text to BERT embeddings
    input_embedding = text_to_bert_embeddings([input_text], tokenizer, bert_model, device)

    # Reshape input for prediction
    input_embedding = input_embedding.reshape(1, -1)

    # Predict the label and calculate probabilities
    predicted_label = model.predict(input_embedding)[0]
    probabilities = model.predict_proba(input_embedding)[0]
    confidence = probabilities[predicted_label]

    if confidence < THRESHOLD:
        print("\nLow confidence in prediction. Label may be ambiguous.")
    else:
        label_description = label_mapping.get(predicted_label, "Unknown")
        print(f"\nPredicted label: {predicted_label} ({label_description})")
        print(f"Confidence: {confidence * 100:.2f}%")

    # Display class probabilities
    print("\nClass Probabilities:")
    for label, prob in enumerate(probabilities):
        print(f"  {label_mapping[label]}: {prob:.2f}")
