from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(y_true, y_pred):
    """Evaluate the model and return performance metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')  # Weighted for imbalanced classes
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }
