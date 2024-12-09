from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with adjusted settings."""
    model = LogisticRegression(
        max_iter=2000,          # Increased iterations
        solver='liblinear',     # Stable solver
        tol=1e-4,               # Early stopping tolerance
        random_state=42,
        penalty='l2',           # Ridge regularization
        C=1.0                   # Regularization strength
    )
    model.fit(X_train, y_train)
    return model
