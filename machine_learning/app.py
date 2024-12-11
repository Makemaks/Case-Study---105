import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
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

LABEL_MAPPING = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral (Non-Offensive)"
}

# Evaluation function
def evaluate_model(y_true, y_pred):
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

# Cache BERT model and tokenizer
@st.cache_resource
def initialize_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.to(device)
    bert_model.eval()
    return tokenizer, bert_model, device

# Cache dataset loading
@st.cache_resource
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data['tweet'], data['class']

# Cache model loading
@st.cache_resource
def load_model(model_name):
    if model_name == "Naive Bayes":
        return {
            "model": joblib.load(MODEL_FILES["Naive Bayes"]["model"]),
            "scaler": joblib.load(MODEL_FILES["Naive Bayes"]["scaler"]),
        }
    return joblib.load(MODEL_FILES[model_name])

@st.cache_resource
def load_all_models():
    return {name: load_model(name) for name in MODEL_FILES}

# Preload all resources with progress bar
progress = st.progress(0)
progress.text("Loading dataset...")
tweets, labels = load_data(DATA_FILE)
data = pd.DataFrame({"tweet": tweets, "label": labels})
progress.progress(20)

progress.text("Initializing BERT...")
tokenizer, bert_model, device = initialize_bert()
progress.progress(40)

progress.text("Generating embeddings...")
X, _, _ = preprocess_and_generate_embeddings(tweets, embedding_file=EMBEDDING_FILE, batch_size=32)
_, X_test, _, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
progress.progress(60)

progress.text("Loading models...")
models = load_all_models()
progress.progress(80)

progress.text("Evaluating models...")
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

model_metrics = {}
for model_name, y_pred in predictions.items():
    if "error" not in y_pred:
        metrics = evaluate_model(y_test, y_pred)
        model_metrics[model_name] = metrics
progress.progress(100)
progress.text("All resources loaded.")

# Streamlit App
st.title("Exploratory Data Analysis & Model Evaluation Dashboard")

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

    # Display label definitions
    st.subheader("Label Definitions")
    label_definitions = pd.DataFrame(
        list(LABEL_MAPPING.items()), 
        columns=["Label ID", "Definition"]
    )
    st.table(label_definitions)

    # Display first few rows
    st.subheader("First Few Rows")
    st.write(data.head())

    # Display class distribution
    st.subheader("Class Distribution")
    class_distribution = data["label"].value_counts().reset_index()
    class_distribution.columns = ["Class", "Count"]
    st.write(class_distribution)

    # Basic Statistics
    st.subheader("Basic Statistics")
    if not data.empty:
        # Compute statistics
        basic_stats = {
            "Statistic": ["Count", "Mean", "Standard Deviation", "Min", "25th Percentile", "50th Percentile (Median)", "75th Percentile", "Max"],
            "Value": [
                int(data["label"].count()),
                round(data["label"].mean(), 4),
                round(data["label"].std(), 4),
                data["label"].min(),
                data["label"].quantile(0.25),
                data["label"].median(),
                data["label"].quantile(0.75),
                data["label"].max()
            ]
        }
        stats_df = pd.DataFrame(basic_stats)
        st.write(stats_df)

    # Display missing values
    st.subheader("Missing Values")
    missing_values = data.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Count"]
    missing_values["Missing Percentage"] = (missing_values["Missing Count"] / len(data)) * 100
    st.write(missing_values)

    # Summary note
    if missing_values["Missing Count"].sum() > 0:
        st.warning("Some columns contain missing values. Consider cleaning the data.")
    else:
        st.success("No missing values detected.")

# Tab 2: Visualizations
with tab2:
    st.header("Visualizations")

    # Pie chart for class distribution
    st.subheader("Class Distribution")
    class_counts = data["label"].value_counts()
    class_labels = [f"{LABEL_MAPPING[label]} ({label})" for label in class_counts.index]
    fig_pie = px.pie(
        values=class_counts.values,
        names=class_labels,
        title="Class Distribution",
        hole=0.4,
    )
    st.plotly_chart(fig_pie)

    # Normal curve for label distribution
    st.subheader("Density and Histogram of Labels")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram of labels
    sns.histplot(data["label"], kde=True, stat="density", bins=30, ax=ax, color="skyblue")

    # Plot normal curve (KDE)
    sns.kdeplot(data["label"], ax=ax, color="red", lw=2, label="KDE (Normal Curve)")

    # Add labels and title
    ax.set_title("Density and Histogram of Labels", fontsize=14)
    ax.set_xlabel("Label", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()

    # Display the plot
    st.pyplot(fig)

    st.subheader("Word Clouds for Each Class")
    for label in class_counts.index:
        st.write(f"Word Cloud for Label {label} ({LABEL_MAPPING[label]})")
        text_data = " ".join(data[data["label"] == label]["tweet"])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)


# Tab 3: Interactive Analysis
with tab3:
    st.header("Interactive Analysis")
    scatter_x = st.selectbox("Select X-Axis", data.columns)
    scatter_y = st.selectbox("Select Y-Axis", data.columns)
    fig = px.scatter(data, x=scatter_x, y=scatter_y, title="Scatter Plot")
    st.plotly_chart(fig)

# Tab 4: Model Comparison
with tab4:
    st.header("Model Comparison")
    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame(model_metrics).T.reset_index()
    metrics_df.columns = ["Model", "Accuracy", "F1 Score", "Precision", "Recall"]
    st.dataframe(metrics_df)

    # Visualize metrics with bar charts
    st.subheader("Performance Metrics Comparison")
    for metric in ["Accuracy", "F1 Score", "Precision", "Recall"]:
        fig = px.bar(metrics_df, x="Model", y=metric, color="Model", title=f"{metric} Comparison")
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
