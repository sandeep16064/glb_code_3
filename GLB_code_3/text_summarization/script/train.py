# scripts/train.py
import os
import sys
sys.path.append(os.path.abspath('/content/glb_code/text_summarization'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from models.transformer import TransformerModel
from utils.data_loader import TextDataset
from utils.metrics import calculate_metrics

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            src, tgt = [b.to(device) for b in batch]
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
        
        val_loss = validate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Val Loss: {val_loss}')

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            src, tgt = [b.to(device) for b in batch]
            output = model(src, tgt)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab_size = 10000
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    batch_size = 32
    epochs = 10
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    train_data = TextDataset("data/dataset.csv", tokenizer)
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device)
