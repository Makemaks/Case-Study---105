from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
import joblib

def train_naive_bayes(X_train, y_train, save_path):
    """Train Naive Bayes with non-negative transformed data and save the model."""
    # Transform BERT embeddings to a non-negative range
    scaler = MinMaxScaler()  # Scale data to range [0, 1]
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_scaled, y_train)

    # Save the scaler and model
    joblib.dump(scaler, save_path.replace(".pkl", "_scaler.pkl"))
    joblib.dump(model, save_path)

    return model
