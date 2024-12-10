import os
import joblib
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from utils.text_to_bert_embeddings import text_to_bert_embeddings

# Filepaths
TRAINED_MODELS_DIR = "./trained_models/"

# Define model files
MODEL_FILES = {
    "Logistic Regression": os.path.join(TRAINED_MODELS_DIR, "logistic_regression.pkl"),
    "Random Forest": os.path.join(TRAINED_MODELS_DIR, "random_forest.pkl"),
    "Support Vector Machine": os.path.join(TRAINED_MODELS_DIR, "svm.pkl"),
    "Naive Bayes": {
        "model": os.path.join(TRAINED_MODELS_DIR, "naive_bayes.pkl"),
        "scaler": os.path.join(TRAINED_MODELS_DIR, "naive_bayes_scaler.pkl"),
    },
}

# Define label mapping
label_mapping = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral (Non-Offensive)",
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

# Load models
print("\nLoading models...")
models = {}
for model_name, model_path in MODEL_FILES.items():
    if model_name == "Naive Bayes":
        # Load Naive Bayes model and scaler
        models[model_name] = {
            "model": joblib.load(model_path["model"]),
            "scaler": joblib.load(model_path["scaler"]),
        }
    else:
        # Load other models
        models[model_name] = joblib.load(model_path)

# Test all models with manual input
while True:
    input_text = input("\nEnter text to classify (or type 'exit' to quit): ").strip()
    if input_text.lower() == "exit":
        print("Exiting testing...")
        break

    # Generate BERT embeddings for the input text
    print("Generating BERT embedding for the input text...")
    input_embedding = text_to_bert_embeddings([input_text], tokenizer, bert_model, device, batch_size=1)
    input_embedding = input_embedding.reshape(1, -1)

    print("\nResults for all models:")
    for model_name, model_data in models.items():
        try:
            if model_name == "Naive Bayes":
                # Scale the input for Naive Bayes
                scaler = model_data["scaler"]
                model = model_data["model"]
                scaled_embedding = scaler.transform(input_embedding)
                predicted_label = model.predict(scaled_embedding)[0]
                probabilities = model.predict_proba(scaled_embedding)[0]
            else:
                model = model_data
                predicted_label = model.predict(input_embedding)[0]
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_embedding)[0]
                else:
                    probabilities = np.zeros(len(label_mapping))
                    probabilities[predicted_label] = 1.0

            confidence = probabilities[predicted_label]

            label_description = label_mapping.get(predicted_label, "Unknown")
            result = (
                f"Predicted Label: {predicted_label} ({label_description})\n"
                f"Confidence: {confidence * 100:.2f}%"
            )

            # Add notice for low confidence
            if confidence < THRESHOLD:
                result += f"\nNotice: Low confidence ({confidence * 100:.2f}%) - Label may be ambiguous."

            # Display class probabilities
            result += "\nClass Probabilities:\n" + "\n".join(
                [f"  {label_mapping[label]}: {prob:.2f}" for label, prob in enumerate(probabilities)]
            )

        except Exception as e:
            result = f"Error during prediction: {e}"

        print(f"\n{model_name}:\n{result}")
