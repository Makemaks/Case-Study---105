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

@st.cache_resource
def load_data(file_path):
    """Loads dataset from a file."""
    data = pd.read_csv(file_path)
    return data

def load_model_and_pipeline(model_name):
    """Loads the selected model and PCA pipeline."""
    model = joblib.load(MODEL_FILES[model_name])
    pca = joblib.load(PCA_FILE)
    return model, pca

def predict_text(model, pca, input_text):
    """Predicts the label for input text using the selected model."""
    # Transform the input text
    text_embedding = np.random.random((768,))  # Mock embedding (replace with actual pipeline)
    reduced_embedding = pca.transform([text_embedding])
    prediction = model.predict(reduced_embedding)
    label_description = LABEL_MAPPING[prediction[0]]
    return prediction[0], label_description

# Streamlit App
st.title("Exploratory Data Analysis & Prediction Dashboard")
# st.sidebar.title("Navigation")

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Visualizations", "Interactive Analysis", "Text Classification"])

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
    st.write(data.isnull().sum())

# Tab 2: Visualizations
with tab2:
    st.header("Visualizations")

    # Heatmap for correlations
    st.subheader("Heatmap of Feature Correlations")
    numeric_columns = data.select_dtypes(include=[np.number])
    correlation = numeric_columns.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Boxplot
    st.subheader("Boxplot for Numeric Features")
    selected_column = st.selectbox("Select a Column for Boxplot", numeric_columns.columns)
    fig, ax = plt.subplots()
    sns.boxplot(data=data, y=selected_column, ax=ax)
    st.pyplot(fig)

    # Bar Chart
    st.subheader("Bar Chart of Feature Distribution")
    categorical_columns = data.select_dtypes(include=[object])
    selected_category = st.selectbox("Select a Categorical Column", categorical_columns.columns)
    fig, ax = plt.subplots()
    data[selected_category].value_counts().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title(f"Distribution of {selected_category}")
    ax.set_xlabel(selected_category)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Tab 3: Interactive Analysis
with tab3:
    st.header("Interactive Analysis")

    # Scatter plot using Plotly
    st.subheader("Interactive Scatter Plot")
    scatter_x = st.selectbox("Select X-Axis", data.columns)
    scatter_y = st.selectbox("Select Y-Axis", data.columns)
    scatter_color = st.selectbox("Select Color Feature", data.columns)
    fig = px.scatter(data, x=scatter_x, y=scatter_y, color=scatter_color, title="Scatter Plot")
    st.plotly_chart(fig)

    # Pairplot
    st.subheader("Pairplot of Features")
    pairplot_features = st.multiselect("Select Features for Pairplot", numeric_columns.columns)
    if len(pairplot_features) > 1:
        fig = sns.pairplot(data[pairplot_features])
        st.pyplot(fig)

    # Interactive Filter
    st.subheader("Filter Dataset")
    filter_column = st.selectbox("Select a Column to Filter", data.columns)
    unique_values = data[filter_column].unique()
    selected_values = st.multiselect(f"Select values of {filter_column} to display", unique_values)
    filtered_data = data[data[filter_column].isin(selected_values)]
    st.write(filtered_data)

# Tab 4: Text Classification
with tab4:
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
