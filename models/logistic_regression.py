from sklearn.linear_model import LogisticRegression
import joblib

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with verbose output and save the model."""
    model = LogisticRegression(
        max_iter=2000,          # Increased iterations
        solver='saga',          # Stable solver
        tol=1e-4,               # Early stopping tolerance
        random_state=42,
        penalty='l2',           # Ridge regularization
        C=1.0,                  # Regularization strength
        verbose=1               # Add verbose output
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "./trained_models/logistic_regression.pkl")  # Save the model
    return model
