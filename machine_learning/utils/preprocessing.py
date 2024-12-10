import re
import nltk
from nltk.corpus import stopwords

# Ensure necessary resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stop_words.add("rt")

def preprocess_text(text):
    """Cleans and preprocesses text data."""
    text = re.sub(r"&[^\s;]+;", "", text)  # Remove HTML entities
    text = re.sub(r"@[^ ]+", "user", text)  # Replace user mentions
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove unnecessary symbols
    tokens = nltk.word_tokenize(text)  # Tokenize with NLTK
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    return " ".join(tokens)
