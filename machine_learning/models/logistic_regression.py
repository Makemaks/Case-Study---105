from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_logistic_regression(X_train, y_train, X_val, y_val, save_path):
    """Train Logistic Regression with early stopping and log hyperparameters."""
    model = LogisticRegression(
        max_iter=310,
        solver='saga',
        tol=1e-4,
        random_state=42,
        penalty='l2',
        verbose=1
    )

    best_val_accuracy = 0
    no_improvement_epochs = 0
    patience = 10  # Early stopping patience

    tuning_results = []

    for epoch in range(1, 6):
        model.fit(X_train, y_train)
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        tuning_results.append({
            "Epoch": epoch,
            "C": model.C,
            "Validation Accuracy": val_accuracy
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_epochs = 0
            joblib.dump(model, save_path)  # Save the best model
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            break

    return model, tuning_results

