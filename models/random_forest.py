from sklearn.ensemble import RandomForestClassifier
import joblib

def train_random_forest(X_train, y_train):
    """Train Random Forest with verbose output and save the model."""
    model = RandomForestClassifier(
        n_estimators=100,    # Number of trees
        random_state=42,     # Ensure reproducibility
        max_depth=None,      # Grow trees until all leaves are pure
        n_jobs=-1,           # Use all CPU cores
        verbose=1            # Add verbose output
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "./trained_models/random_forest.pkl")  # Save the model
    return model
