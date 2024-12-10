import os
import numpy as np
import joblib
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils.data_loader import load_data
from utils.evaluation import evaluate_model
from models.logistic_regression import train_logistic_regression
from models.random_forest import train_random_forest
from models.svm import train_svm
from models.naive_bayes import train_naive_bayes

# Filepaths
RAW_DATA_FILEPATH = "./data/train.csv"
TRAINED_MODELS_DIR = "./trained_models/"
TRAINED_PIPELINE_DIR = "./trained_pipeline/"
EMBEDDING_FILE = "./bert_embeddings.npy"

# Ensure directories exist
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(TRAINED_PIPELINE_DIR, exist_ok=True)

# Define model paths
MODEL_PATHS = {
    "Logistic Regression": os.path.join(TRAINED_MODELS_DIR, "logistic_regression.pkl"),
    "Random Forest": os.path.join(TRAINED_MODELS_DIR, "random_forest.pkl"),
    "Naive Bayes": os.path.join(TRAINED_MODELS_DIR, "naive_bayes.pkl"),
    "Support Vector Machine": os.path.join(TRAINED_MODELS_DIR, "svm.pkl"),
}

# Step 1: Load Dataset
print("Loading dataset...")
tweets, labels = load_data(RAW_DATA_FILEPATH)

# Step 2: Initialize BERT
print("Initializing BERT tokenizer and model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)
bert_model.eval()

# Step 3: Generate BERT embeddings
if os.path.exists(EMBEDDING_FILE):
    print("Loading precomputed embeddings...")
    X = np.load(EMBEDDING_FILE)
else:
    from utils.text_to_bert_embeddings import text_to_bert_embeddings
    print("Generating BERT embeddings...")
    X = text_to_bert_embeddings(tweets, tokenizer, bert_model, device, batch_size=32)
    np.save(EMBEDDING_FILE, X)

# Step 4: Split Data
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Step 5: Handle Class Imbalance with SMOTE
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Step 6: Train or Load Models
models = {}
for model_name, model_path in MODEL_PATHS.items():
    if os.path.exists(model_path):
        print(f"Loading saved model: {model_name}...")
        models[model_name] = joblib.load(model_path)
    else:
        print(f"Training {model_name}...")
        if model_name == "Logistic Regression":
            models[model_name] = train_logistic_regression(X_train, y_train, X_val, y_val, model_path)
        elif model_name == "Random Forest":
            models[model_name] = train_random_forest(X_train, y_train)
        elif model_name == "Support Vector Machine":
            models[model_name] = train_svm(X_train, y_train)
        elif model_name == "Naive Bayes":
            models[model_name] = train_naive_bayes(X_train, y_train)
        # Save the trained model
        joblib.dump(models[model_name], model_path)

# Step 7: Evaluate Models
print("\nEvaluating models...")
model_results = {}
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    model_results[model_name] = evaluate_model(y_test, y_pred)

# Step 8: Print and Compare Results
print("\nModel Comparison:")
for model_name, metrics in model_results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
