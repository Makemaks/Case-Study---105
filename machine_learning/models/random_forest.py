from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_random_forest(X_train, y_train, save_path):
    """Train Random Forest model and log hyperparameters."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)

    # Log hyperparameters
    tuning_results = [{
        "n_estimators": model.n_estimators,
        "Validation Accuracy": accuracy_score(y_train, model.predict(X_train))
    }]
    return model, tuning_results
