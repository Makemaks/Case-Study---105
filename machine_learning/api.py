import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel
from utils.text_to_bert_embeddings import text_to_bert_embeddings

# Initialize FastAPI app
app = FastAPI()

# Filepaths
TRAINED_MODELS_DIR = "./trained_models/"
MODEL_FILES = {
    "Logistic Regression": os.path.join(TRAINED_MODELS_DIR, "logistic_regression.pkl"),
    "Random Forest": os.path.join(TRAINED_MODELS_DIR, "random_forest.pkl"),
    "Support Vector Machine": os.path.join(TRAINED_MODELS_DIR, "svm.pkl"),
    "Naive Bayes": os.path.join(TRAINED_MODELS_DIR, "naive_bayes.pkl"),
    "Naive Bayes Scaler": os.path.join(TRAINED_MODELS_DIR, "naive_bayes_scaler.pkl"),
}

label_mapping = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral (Non-Offensive)"
}

# Confidence threshold
THRESHOLD = 0.7  # Minimum confidence to flag low-confidence predictions

# Initialize BERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)
bert_model.eval()

# Load models
models = {name: joblib.load(path) for name, path in MODEL_FILES.items() if name != "Naive Bayes Scaler"}
naive_bayes_scaler = joblib.load(MODEL_FILES["Naive Bayes Scaler"])

# Request schema
class TextRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(text_request: TextRequest):
    input_text = text_request.text

    # Generate BERT embeddings
    try:
        input_embedding = text_to_bert_embeddings([input_text], tokenizer, bert_model, device, batch_size=1).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

    results = {}
    for model_name, model in models.items():
        try:
            if model_name == "Naive Bayes":
                # Naive Bayes requires scaling
                input_scaled = naive_bayes_scaler.transform(input_embedding)
                predicted_label = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
            else:
                predicted_label = model.predict(input_embedding)[0]
                probabilities = model.predict_proba(input_embedding)[0] if hasattr(model, "predict_proba") else None

            confidence = probabilities[predicted_label] if probabilities is not None else None
            confidence_notice = (
                f"Low confidence ({confidence * 100:.2f}%) - Label may be ambiguous."
                if confidence and confidence < THRESHOLD
                else None
            )

            results[model_name] = {
                "predicted_label": label_mapping.get(predicted_label, "Unknown"),
                "confidence": confidence * 100 if confidence is not None else None,
                "confidence_notice": confidence_notice,
                "class_probabilities": {
                    label_mapping[label]: prob for label, prob in enumerate(probabilities)
                } if probabilities is not None else "Not available",
            }
        except Exception as e:
            results[model_name] = {"error": f"Prediction failed: {str(e)}"}

    return {"input_text": input_text, "predictions": results}
