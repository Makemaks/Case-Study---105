from sklearn.ensemble import RandomForestClassifier
import joblib

def train_random_forest(X_train, y_train, save_path):
    """Train Random Forest model and save it."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)  # Save model
    return model
