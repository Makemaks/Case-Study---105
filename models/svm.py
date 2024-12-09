from sklearn.svm import SVC
import joblib

def train_svm(X_train, y_train):
    """Train SVM with verbose output and save the model."""
    model = SVC(
        kernel='linear',     # Linear kernel for text classification
        probability=True,    # Enable probability estimation
        random_state=42,     # Ensure reproducibility
        verbose=True         # Add verbose output
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "./trained_models/svm.pkl")  # Save the model
    return model
