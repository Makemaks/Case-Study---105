import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(filepath):
    """Load raw data."""
    data = pd.read_csv(filepath)
    return data['tweet'], data['class']

def split_data(tweets, labels, test_size=0.2, random_state=42, sample_size=0.5):
    """Split and optionally downsample data."""
    # Downsample to reduce dataset size
    tweets, labels = tweets.sample(frac=sample_size, random_state=random_state), labels.sample(frac=sample_size, random_state=random_state)
    return train_test_split(tweets, labels, test_size=test_size, random_state=random_state)

def save_processed_data(processed_dir, X_train, X_test, y_train, y_test):
    """Save processed data to files."""
    os.makedirs(processed_dir, exist_ok=True)  # Create directory if not exists
    pd.Series(X_train).to_csv(os.path.join(processed_dir, "X_train.csv"), index=False, header=False)
    pd.Series(X_test).to_csv(os.path.join(processed_dir, "X_test.csv"), index=False, header=False)
    pd.Series(y_train).to_csv(os.path.join(processed_dir, "y_train.csv"), index=False, header=False)
    pd.Series(y_test).to_csv(os.path.join(processed_dir, "y_test.csv"), index=False, header=False)

def load_processed_data(processed_dir):
    """Load processed data from files."""
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"), header=None).squeeze()  # Ensure it's a Series
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"), header=None).squeeze()    # Ensure it's a Series
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"), header=None).squeeze()  # Ensure it's a Series
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"), header=None).squeeze()    # Ensure it's a Series
    return X_train, X_test, y_train, y_test
