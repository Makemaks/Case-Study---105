import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TextDataset(Dataset):
    """Dataset for handling text inputs."""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return {key: val.squeeze(0) for key, val in inputs.items()}


def text_to_bert_embeddings(texts, tokenizer, bert_model, device, batch_size=32, max_length=128):
    """
    Generate BERT embeddings for a list of texts.
    
    Args:
        texts (list): List of input texts.
        tokenizer: BERT tokenizer.
        bert_model: Pretrained BERT model.
        device: Computation device ('cuda' or 'cpu').
        batch_size (int): Batch size for processing.
        max_length (int): Maximum token length for inputs.

    Returns:
        np.ndarray: Array of CLS embeddings for the input texts.
    """
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []
    bert_model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embeddings
            embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(embeddings)
