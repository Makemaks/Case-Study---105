from sklearn.svm import SVC
import joblib

def train_svm(X_train, y_train):
    """Train SVM with verbose output and save the model."""
    model = SVC(
        kernel='linear',
        C=0.1,
        probability=True,
        random_state=42,
        max_iter=5000,
        verbose=True
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "./trained_models/svm.pkl")  # Save the model
    return model 
