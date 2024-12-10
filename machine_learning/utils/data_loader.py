import pandas as pd

def load_data(filepath):
    """Load raw data."""
    data = pd.read_csv(filepath)
    return data['tweet'], data['class']
