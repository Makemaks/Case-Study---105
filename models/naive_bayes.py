from sklearn.naive_bayes import MultinomialNB

def train_naive_bayes(X_train, y_train):
    """
    Trains a Naive Bayes Classifier.

    Args:
        X_train: Training features (e.g., TF-IDF vectors).
        y_train: Training labels.

    Returns:
        Trained Naive Bayes model.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model
