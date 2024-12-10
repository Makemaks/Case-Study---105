from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_logistic_regression(X_train, y_train, X_val, y_val, save_path):
    """Train Logistic Regression with early stopping."""
    model = LogisticRegression(
        max_iter=1000,         # Limit iterations
        solver='saga',         # Stable solver for large datasets
        tol=1e-4,              # Early stopping tolerance
        random_state=42,
        penalty='l2',          # Ridge regularization
        C=1.0,                 # Regularization strength
        verbose=1              # Verbose output for debugging
    )

    best_val_accuracy = 0
    no_improvement_epochs = 0
    patience = 10  # Early stopping patience

    for epoch in range(1, 1001):  # Hard limit of 1000 epochs
        model.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = accuracy_score(y_val, model.predict(X_val))

        print(f"Epoch {epoch}: Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_epochs = 0
            joblib.dump(model, save_path)  # Save the best model
            print("New best model saved.")
        else:
            no_improvement_epochs += 1
            print("No improvement. Continuing...")

        # Early stopping criterion
        if no_improvement_epochs >= patience:
            print("Early stopping triggered.")
            break

    return model
