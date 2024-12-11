import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils.text_to_bert_embeddings import text_to_bert_embeddings
from utils.data_loader import load_data
from utils.preprocessing import preprocess_and_generate_embeddings

# Filepaths
DATA_FILE = "./data/train.csv"
TRAINED_MODELS_DIR = "./trained_models/"
EMBEDDING_FILE = "./bert_embeddings.npy"

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

tweets, labels = load_data(DATA_FILE)
data = pd.DataFrame({"tweet": tweets, "label": labels}) 

# Define label mapping
LABEL_MAPPING = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral (Non-Offensive)"
}

# Evaluation function
def evaluate_model(y_true, y_pred):
    """Evaluate the model and return performance metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }

# Initialize BERT model and tokenizer
@st.cache_resource
def initialize_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.to(device)
    bert_model.eval()
    return tokenizer, bert_model, device

tokenizer, bert_model, device = initialize_bert()

# Utility functions
@st.cache_resource
def load_data(file_path):
    """Loads dataset from a file."""
    data = pd.read_csv(file_path)
    return data['tweet'], data['class']

@st.cache_resource
def load_model(model_name):
    """Loads the selected model."""
    if model_name == "Naive Bayes":
        return {
            "model": joblib.load(MODEL_FILES["Naive Bayes"]["model"]),
            "scaler": joblib.load(MODEL_FILES["Naive Bayes"]["scaler"]),
        }
    return joblib.load(MODEL_FILES[model_name])

def predict_with_all_models(input_text, models, tokenizer, bert_model, device):
    """Generates predictions for input text using all models."""
    input_embedding = text_to_bert_embeddings([input_text], tokenizer, bert_model, device, batch_size=1)
    input_embedding = input_embedding.reshape(1, -1)
    results = {}

    for model_name, model_data in models.items():
        try:
            if model_name == "Naive Bayes":
                scaler = model_data["scaler"]
                model = model_data["model"]
                scaled_embedding = scaler.transform(input_embedding)
                predicted_label = model.predict(scaled_embedding)[0]
                probabilities = model.predict_proba(scaled_embedding)[0]
            else:
                model = model_data
                predicted_label = model.predict(input_embedding)[0]
                probabilities = model.predict_proba(input_embedding)[0] if hasattr(model, "predict_proba") else None

            label_description = LABEL_MAPPING[predicted_label]
            confidence = probabilities[predicted_label] if probabilities is not None else "N/A"
            results[model_name] = {
                "label": label_description,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}

    return results

# Load all models
@st.cache_resource
def load_all_models():
    """Loads all models."""
    return {name: load_model(name) for name in MODEL_FILES}

models = load_all_models()

# Load and preprocess test data
tweets, labels = load_data(DATA_FILE)
X, _, _ = preprocess_and_generate_embeddings(tweets, embedding_file=EMBEDDING_FILE, batch_size=32)

# Split data
_, X_test, _, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Generate predictions
predictions = {}
scalers = {name: model_data["scaler"] for name, model_data in models.items() if isinstance(model_data, dict) and "scaler" in model_data}

for model_name, model_data in models.items():
    try:
        X_test_processed = X_test
        if model_name == "Naive Bayes" and model_name in scalers:
            X_test_processed = scalers[model_name].transform(X_test)
        model = model_data["model"] if isinstance(model_data, dict) else model_data
        predictions[model_name] = model.predict(X_test_processed)
    except Exception as e:
        predictions[model_name] = {"error": str(e)}

# Evaluate models
model_metrics = {}
for model_name, y_pred in predictions.items():
    if "error" not in y_pred:
        metrics = evaluate_model(y_test, y_pred)
        model_metrics[model_name] = metrics

# Streamlit App
st.title("Exploratory Data Analysis & Model Evaluation Dashboard")

# Tabs for navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset Overview", 
    "Visualizations", 
    "Interactive Analysis", 
    "Model Comparison", 
    "Text Classification"
])

# Tab 1: Dataset Overview
with tab1:
    st.header("Dataset Overview")
    tweets, labels = load_data(DATA_FILE)

    st.subheader("First Few Rows")
    st.write(pd.DataFrame({"tweet": tweets, "label": labels}).head())

    st.subheader("Summary Statistics")
    st.write(labels.describe())

    st.subheader("Class Distribution")
    st.bar_chart(labels.value_counts())
    missing_values = tweets.isnull().sum()
    st.write(missing_values)
    if missing_values.any():
        st.warning("Dataset contains missing values. Consider cleaning the data.")

# Tab 2: Visualizations
with tab2:
    st.header("Visualizations")
    numeric_columns = data.select_dtypes(include=[np.number])
    if not numeric_columns.empty:
        correlation = numeric_columns.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for visualization.")

# Tab 3: Interactive Analysis
with tab3:
    st.header("Interactive Analysis")
    if data.shape[1] >= 2:
        scatter_x = st.selectbox("Select X-Axis", data.columns)
        scatter_y = st.selectbox("Select Y-Axis", data.columns)
        scatter_color = st.selectbox("Select Color Feature", data.columns)
        fig = px.scatter(data, x=scatter_x, y=scatter_y, color=scatter_color, title="Scatter Plot")
        st.plotly_chart(fig)
    else:
        st.warning("Not enough columns for scatter plot.")

# Tab 4: Model Comparison
with tab4:
    st.header("Model Comparison")

    # Display metrics in a table
    st.subheader("Performance Metrics Table")
    metrics_df = pd.DataFrame(model_metrics).T.reset_index()
    metrics_df.columns = ["Model", "Accuracy", "F1 Score", "Precision", "Recall"]
    st.dataframe(metrics_df)

    # Visualize metrics with bar charts
    st.subheader("Performance Metrics Comparison")

    # Accuracy
    st.write("### Accuracy Comparison")
    fig = px.bar(metrics_df, x="Model", y="Accuracy", color="Model", title="Accuracy Comparison")
    st.plotly_chart(fig)

    # F1 Score
    st.write("### F1 Score Comparison")
    fig = px.bar(metrics_df, x="Model", y="F1 Score", color="Model", title="F1 Score Comparison")
    st.plotly_chart(fig)

    # Precision
    st.write("### Precision Comparison")
    fig = px.bar(metrics_df, x="Model", y="Precision", color="Model", title="Precision Comparison")
    st.plotly_chart(fig)

    # Recall
    st.write("### Recall Comparison")
    fig = px.bar(metrics_df, x="Model", y="Recall", color="Model", title="Recall Comparison")
    st.plotly_chart(fig)

# Tab 5: Text Classification

with tab5:
    st.header("Text Classification")
    input_text = st.text_input("Input Text", "")

    if input_text:
        predictions = predict_with_all_models(input_text, models, tokenizer, bert_model, device)
        for model_name, result in predictions.items():
            if "error" in result:
                st.error(f"{model_name}: {result['error']}")
            else:
                st.write(f"**{model_name}**")
                st.write(f"- Predicted Label: {result['label']}")
                st.write(f"- Confidence: {result['confidence'] * 100:.2f}%")
