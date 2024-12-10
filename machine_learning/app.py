import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Filepaths
DATA_FILE = "./data/train.csv"
TRAINED_MODELS_DIR = "./trained_models/"
TRAINED_PIPELINE_DIR = "./trained_pipeline/"

# Define pipeline component files
PCA_FILE = os.path.join(TRAINED_PIPELINE_DIR, "bert_pca.pkl")

# Define model files
MODEL_FILES = {
    "Logistic Regression": os.path.join(TRAINED_MODELS_DIR, "logistic_regression.pkl"),
    "Random Forest": os.path.join(TRAINED_MODELS_DIR, "random_forest.pkl"),
    "Support Vector Machine": os.path.join(TRAINED_MODELS_DIR, "svm.pkl"),
    "Naive Bayes": os.path.join(TRAINED_MODELS_DIR, "naive_bayes.pkl"),
}

# Define label mapping
LABEL_MAPPING = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral (Non-Offensive)"
}

# Utility functions
@st.cache_resource
def load_data(file_path):
    """Loads dataset from a file."""
    data = pd.read_csv(file_path)
    return data

@st.cache_resource
def load_model_and_pipeline(model_name):
    """Loads the selected model and PCA pipeline."""
    model = joblib.load(MODEL_FILES[model_name])
    pca = joblib.load(PCA_FILE)
    return model, pca

def sanitize_text_for_plotting(text):
    """Sanitize text to avoid errors in plotting."""
    if pd.isnull(text) or not isinstance(text, str):  # Handle null or non-string values
        return ""
    sanitized_text = (
        text.replace("&", "&amp;")
        .replace("$", "\\$")
        .replace("@", "")
        .replace("\n", " ")
        .replace("\"", "")
        .replace("\'", "")
    )
    return sanitized_text[:50]  # Limit text length for visualization

def predict_text(model, pca, input_text):
    """Predicts the label for input text using the selected model."""
    # Transform the input text
    text_embedding = np.random.random((768,))  # Mock embedding (replace with actual pipeline)
    reduced_embedding = pca.transform([text_embedding])
    prediction = model.predict(reduced_embedding)
    label_description = LABEL_MAPPING[prediction[0]]
    return prediction[0], label_description

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

    # Load the data
    data = load_data(DATA_FILE)

    # Display first few rows
    st.subheader("First Few Rows")
    st.write(data.head())

    # Display dataset info
    st.subheader("Dataset Information")
    buffer = st.empty()
    data_info = data.info(buf=buffer)
    st.text(buffer._value)

    # Generate summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Check for missing values
    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values)
    if missing_values.any():
        st.warning("Dataset contains missing values. Consider cleaning the data.")

# Tab 2: Visualizations
with tab2:
    st.header("Visualizations")

    # Heatmap for correlations
    st.subheader("Heatmap of Feature Correlations")
    numeric_columns = data.select_dtypes(include=[np.number])
    if numeric_columns.empty:
        st.warning("No numeric columns available for heatmap.")
    else:
        correlation = numeric_columns.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        st.pyplot(fig)

    # Boxplot
    st.subheader("Boxplot for Numeric Features")
    if not numeric_columns.empty:
        selected_column = st.selectbox("Select a Column for Boxplot", numeric_columns.columns)
        fig, ax = plt.subplots()
        sns.boxplot(data=data, y=selected_column, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for boxplot.")

    # Bar Chart
    st.subheader("Bar Chart of Feature Distribution")
    categorical_columns = data.select_dtypes(include=[object])
    if not categorical_columns.empty:
        selected_category = st.selectbox("Select a Categorical Column", categorical_columns.columns)
        sanitized_data = data[selected_category].apply(sanitize_text_for_plotting)
        fig, ax = plt.subplots()
        sanitized_data.value_counts().plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title(f"Distribution of {selected_category}")
        ax.set_xlabel(selected_category)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.warning("No categorical columns available for bar chart.")

# Tab 3: Interactive Analysis
with tab3:
    st.header("Interactive Analysis")

    # Scatter plot using Plotly
    st.subheader("Interactive Scatter Plot")
    if data.shape[1] < 2:
        st.warning("Not enough columns for scatter plot.")
    else:
        scatter_x = st.selectbox("Select X-Axis", data.columns)
        scatter_y = st.selectbox("Select Y-Axis", data.columns)
        scatter_color = st.selectbox("Select Color Feature", data.columns)

        sanitized_data = data.copy()
        sanitized_data[scatter_x] = sanitized_data[scatter_x].apply(sanitize_text_for_plotting)
        sanitized_data[scatter_y] = sanitized_data[scatter_y].apply(sanitize_text_for_plotting)
        sanitized_data[scatter_color] = sanitized_data[scatter_color].apply(sanitize_text_for_plotting)

        fig = px.scatter(sanitized_data, x=scatter_x, y=scatter_y, color=scatter_color, title="Scatter Plot")
        st.plotly_chart(fig)

# Tab 4: Model Comparison
with tab4:
    st.header("Model Comparison")

    st.subheader("Performance Metrics")
    # Mocked metrics for demonstration (replace with actual evaluation results)
    metrics = {
        "Model": ["Logistic Regression", "Random Forest", "SVM", "Naive Bayes"],
        "Accuracy": [0.85, 0.90, 0.87, 0.75],
        "F1 Score": [0.83, 0.89, 0.85, 0.72],
        "Precision": [0.84, 0.91, 0.86, 0.74],
        "Recall": [0.82, 0.88, 0.84, 0.70]
    }
    df_metrics = pd.DataFrame(metrics)
    st.dataframe(df_metrics)

    # Bar chart for accuracy comparison
    st.subheader("Accuracy Comparison")
    fig = px.bar(df_metrics, x="Model", y="Accuracy", color="Model", title="Model Accuracy")
    st.plotly_chart(fig)

# Tab 5: Text Classification
with tab5:
    st.header("Text Classification")

    # Select Model
    st.subheader("Select Model")
    selected_model_name = st.selectbox("Choose a Model", list(MODEL_FILES.keys()))
    model, pca = load_model_and_pipeline(selected_model_name)

    # Input Text
    st.subheader("Enter Text to Classify")
    input_text = st.text_input("Input Text", "")

    if input_text:
        # Predict
        predicted_label, label_description = predict_text(model, pca, input_text)

        # Display Result
        st.subheader("Prediction Result")
        st.write(f"Predicted Label: **{predicted_label} ({label_description})**")
