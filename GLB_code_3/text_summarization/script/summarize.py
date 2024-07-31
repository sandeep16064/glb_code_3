# scripts/summarize.py
import torch
from transformers import BartTokenizer
from models.transformer import TransformerModel

def summarize_text(model, text, tokenizer, device, max_len=512):
    model.eval()
    tokens = tokenizer(text, return_tensors='pt')
    src = tokens['input_ids'].to(device)
    tgt = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device)
    
    for _ in range(max_len):
        output = model(src, tgt)
        next_token = output.argmax(dim=-1)[:, -1]
        tgt = torch.cat((tgt, next_token.unsqueeze(0)), dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    summary = tokenizer.decode(tgt.squeeze().tolist(), skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab_size = 10000
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)
    
    text = "Load your text data here"
    summary = summarize_text(model, text, tokenizer, device)
    print(summary)
