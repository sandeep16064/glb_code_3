# utils/data_loader.py
import torch
from torch.utils.data import Dataset
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        summary = self.data.iloc[idx, 1]
        tokenized_text = self.tokenizer(text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors="pt")
        tokenized_summary = self.tokenizer(summary, max_length=self.max_len, truncation=True, padding='max_length', return_tensors="pt")
        return tokenized_text['input_ids'].squeeze(), tokenized_summary['input_ids'].squeeze()

def load_text_data(csv_file):
    data = pd.read_csv(csv_file)
    texts = data['text'].tolist()
    return texts

def preprocess_text(nlp, text):
    return nlp(text)
