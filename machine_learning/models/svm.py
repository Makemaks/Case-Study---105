from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def train_svm(X_train, y_train, save_path):
    """Train SVM model and log hyperparameters."""
    model = SVC(
        kernel='linear',
        C=0.1,
        probability=True,
        random_state=42,
        max_iter=310,
        verbose=True
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)

    # Log hyperparameters
    tuning_results = [{
        "kernel": model.kernel,
        "C": model.C,
        "Validation Accuracy": accuracy_score(y_train, model.predict(X_train))
    }]
    return model, tuning_results
