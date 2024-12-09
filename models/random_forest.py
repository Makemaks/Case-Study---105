from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=100,    # Number of trees
        random_state=42,     # Ensure reproducibility
        max_depth=None,      # Grow trees until all leaves are pure
        n_jobs=-1            # Use all CPU cores
    )
    model.fit(X_train, y_train)
    return model
