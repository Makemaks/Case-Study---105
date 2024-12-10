from sklearn.svm import SVC
import joblib

def train_svm(X_train, y_train, save_path):
    """Train SVM model and save it."""
    model = SVC(
        kernel='linear',
        C=0.1,
        probability=True,  # For probability outputs
        random_state=42,
        max_iter=310,
        verbose=True
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)  # Save model
    return model
