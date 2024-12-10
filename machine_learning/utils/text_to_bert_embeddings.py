import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class TextDataset(Dataset):
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
        return inputs


def text_to_bert_embeddings(texts, tokenizer, bert_model, device, batch_size=32, max_length=128):
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []
    for batch in tqdm(dataloader, desc="Generating embeddings"):
        input_ids = batch["input_ids"].squeeze(1).to(device)
        attention_mask = batch["attention_mask"].squeeze(1).to(device)

        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings
            embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(embeddings)
