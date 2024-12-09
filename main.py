import os
from utils.preprocessing import preprocess_text
from utils.data_loader import load_data, split_data, save_processed_data, load_processed_data
from utils.evaluation import evaluate_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from models.logistic_regression import train_logistic_regression
from models.random_forest import train_random_forest
from models.svm import train_svm
from models.naive_bayes import train_naive_bayes
from sklearn.utils import check_array
import joblib

# Filepaths
RAW_DATA_FILEPATH = "./data/train.csv"
PROCESSED_DIR = "./data/processed/"
MODEL_DIR = "./models/"

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if processed data exists
if all(os.path.exists(os.path.join(PROCESSED_DIR, f"{file}.csv")) for file in ["X_train", "X_test", "y_train", "y_test"]):
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data(PROCESSED_DIR)
else:
    print("Processing raw data...")
    # Step 1: Load raw data
    tweets, labels = load_data(RAW_DATA_FILEPATH)

    # Step 2: Preprocess the data
    tweets = tweets.apply(preprocess_text)

    # Step 3: Split the data
    X_train, X_test, y_train, y_test = split_data(tweets, labels)

    # Save processed data
    print("Saving processed data...")
    save_processed_data(PROCESSED_DIR, X_train, X_test, y_train, y_test)

# Ensure X_train and X_test are lists of strings
X_train = X_train.squeeze().tolist()
X_test = X_test.squeeze().tolist()

# Debugging X_train and y_train
print(f"X_train (type: {type(X_train)}, length: {len(X_train)}): First 5 samples: {X_train[:5]}")
print(f"y_train (type: {type(y_train)}, length: {len(y_train)})")
print(f"X_test (type: {type(X_test)}, length: {len(X_test)}): First 5 samples: {X_test[:5]}")

# Step 4: Vectorize the text data
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Scale the data for better convergence
scaler = StandardScaler(with_mean=False)
X_train_tfidf = scaler.fit_transform(X_train_tfidf)
X_test_tfidf = scaler.transform(X_test_tfidf)

# Debug shapes after vectorization and scaling
print(f"X_train_tfidf shape: {X_train_tfidf.shape}")
print(f"y_train shape: {y_train.shape}")

# Step 4.1: Apply PCA for dimensionality reduction
pca = PCA(n_components=500, random_state=42)
X_train_tfidf = pca.fit_transform(X_train_tfidf.toarray())
X_test_tfidf = pca.transform(X_test_tfidf.toarray())

# Debug shapes after PCA
print(f"After PCA - X_train_tfidf shape: {X_train_tfidf.shape}")
print(f"After PCA - X_test_tfidf shape: {X_test_tfidf.shape}")

# Ensure SMOTE compatibility
X_train_tfidf = check_array(X_train_tfidf)
y_train = y_train.squeeze()

# Step 5: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)

# Debug shapes after SMOTE
print(f"After SMOTE - X_train_tfidf shape: {X_train_tfidf.shape}")
print(f"After SMOTE - y_train shape: {y_train.shape}")

# Step 6: Train or Load Models
models = {}
MODEL_PATHS = {
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
    "Support Vector Machine": os.path.join(MODEL_DIR, "svm.pkl"),
    "Naive Bayes": os.path.join(MODEL_DIR, "naive_bayes.pkl"),
}

for model_name, model_path in MODEL_PATHS.items():
    if os.path.exists(model_path):
        print(f"Loading saved model: {model_name}...")
        models[model_name] = joblib.load(model_path)
    else:
        print(f"Training {model_name}...")
        if model_name == "Logistic Regression":
            models[model_name] = train_logistic_regression(X_train_tfidf, y_train)
        elif model_name == "Random Forest":
            models[model_name] = train_random_forest(X_train_tfidf, y_train)
        elif model_name == "Support Vector Machine":
            models[model_name] = train_svm(X_train_tfidf, y_train)
        elif model_name == "Naive Bayes":
            models[model_name] = train_naive_bayes(X_train_tfidf, y_train)
        # Save the trained model
        joblib.dump(models[model_name], model_path)

# Step 7: Evaluate models
model_results = {}
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test_tfidf)
    model_results[model_name] = evaluate_model(y_test, y_pred)

# Step 8: Print and compare results
print("\nModel Comparison:")
for model_name, metrics in model_results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
