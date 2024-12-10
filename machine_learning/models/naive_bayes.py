from sklearn.naive_bayes import MultinomialNB
import joblib

def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes and save the model."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    joblib.dump(model, "./trained_models/naive_bayes.pkl")  # Save the model
    return model
