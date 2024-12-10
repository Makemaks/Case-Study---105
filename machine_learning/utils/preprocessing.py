import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from utils.text_to_bert_embeddings import text_to_bert_embeddings

def preprocess_and_generate_embeddings(texts, embedding_file, batch_size=32, max_length=128):
    """
    Preprocess texts and generate BERT embeddings.
    If embeddings are cached, load them instead.
    
    Args:
        texts (list): List of text data.
        embedding_file (str): Path to save/load embeddings.
        batch_size (int): Batch size for embedding generation.
        max_length (int): Max token length.

    Returns:
        np.ndarray: Generated embeddings.
        tokenizer: BERT tokenizer.
        bert_model: BERT model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.to(device)
    
    if os.path.exists(embedding_file):
        print("Loading precomputed embeddings...")
        return np.load(embedding_file), tokenizer, bert_model
    else:
        print("Generating BERT embeddings...")
        embeddings = text_to_bert_embeddings(texts, tokenizer, bert_model, device, batch_size, max_length)
        np.save(embedding_file, embeddings)
        return embeddings, tokenizer, bert_model
