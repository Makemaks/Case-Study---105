import os
import numpy as np
import joblib
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils.data_loader import load_data
from utils.preprocessing import preprocess_and_generate_embeddings
from utils.evaluation import evaluate_model
from models.logistic_regression import train_logistic_regression
from models.random_forest import train_random_forest
from models.svm import train_svm
from models.naive_bayes import train_naive_bayes

# Filepaths
RAW_DATA_FILEPATH = "./data/train.csv"
TRAINED_MODELS_DIR = "./trained_models/"
EMBEDDING_FILE = "./bert_embeddings.npy"

# Ensure directories exist
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

# Define model paths
MODEL_PATHS = {
    "Logistic Regression": os.path.join(TRAINED_MODELS_DIR, "logistic_regression.pkl"),
    "Random Forest": os.path.join(TRAINED_MODELS_DIR, "random_forest.pkl"),
    "Naive Bayes": os.path.join(TRAINED_MODELS_DIR, "naive_bayes.pkl"),
    "Support Vector Machine": os.path.join(TRAINED_MODELS_DIR, "svm.pkl"),
}

def main():
    # Step 1: Load Dataset
    print("Loading dataset...")
    tweets, labels = load_data(RAW_DATA_FILEPATH)

    # Step 2: Generate BERT Embeddings
    print("Processing data and generating embeddings...")
    X, tokenizer, bert_model = preprocess_and_generate_embeddings(
        tweets, embedding_file=EMBEDDING_FILE, batch_size=32
    )

    # Step 3: Split Data
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Step 4: Handle Class Imbalance with SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Step 5: Train or Load Models
    models = {}
    scalers = {}  # To store scalers for models that need them
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            print(f"Loading saved model: {model_name}...")
            models[model_name] = joblib.load(model_path)
            if model_name == "Naive Bayes":
                scaler_path = model_path.replace(".pkl", "_scaler.pkl")
                scalers[model_name] = joblib.load(scaler_path)
        else:
            print(f"Training {model_name}...")
            if model_name == "Logistic Regression":
                models[model_name] = train_logistic_regression(X_train, y_train, X_val, y_val, model_path)
            elif model_name == "Random Forest":
                models[model_name] = train_random_forest(X_train, y_train, model_path)
            elif model_name == "Support Vector Machine":
                models[model_name] = train_svm(X_train, y_train, model_path)
            elif model_name == "Naive Bayes":
                models[model_name] = train_naive_bayes(X_train, y_train, model_path)
                scaler_path = model_path.replace(".pkl", "_scaler.pkl")
                scalers[model_name] = joblib.load(scaler_path)

    # Step 6: Evaluate Models
    print("\nEvaluating models...")
    model_results = {}
    for model_name, model in models.items():
        X_test_processed = X_test
        if model_name == "Naive Bayes" and model_name in scalers:
            X_test_processed = scalers[model_name].transform(X_test)
        y_pred = model.predict(X_test_processed)
        model_results[model_name] = evaluate_model(y_test, y_pred)


    # Step 7: Print and Compare Results
    print("\nModel Comparison:")
    best_model = None
    best_f1_score = 0

    for model_name, metrics in model_results.items():
        print(f"Model: {model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print("-" * 40)
        
        # Identify the best model based on F1 Score
        if metrics['f1_score'] > best_f1_score:
            best_f1_score = metrics['f1_score']
            best_model = model_name

    # Print overall best model
    print(f"\nOverall Best Model: {best_model}")
    print(f"Best F1 Score: {best_f1_score:.4f}")


if __name__ == "__main__":
    main()
