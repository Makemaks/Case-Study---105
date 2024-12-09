from sklearn.svm import SVC

def train_svm(X_train, y_train):
    """
    Trains a Support Vector Machine (SVM) Classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained SVM model.
    """
    model = SVC(
        kernel='linear',     # Linear kernel for text classification
        probability=True,    # Enable probability estimation
        random_state=42      # Ensure reproducibility
    )
    model.fit(X_train, y_train)
    return model
