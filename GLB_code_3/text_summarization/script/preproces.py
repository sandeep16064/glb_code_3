# scripts/preprocess.py
import os
import pandas as pd
import spacy
import sys
sys.path.append(os.path.abspath('/content/glb_code/text_summarization'))

from utils.data_loader import load_text_data, preprocess_text
from utils.graph_utils import build_knowledge_graph

def preprocess_data(csv_file):
    nlp = spacy.load("en_core_web_sm")
    data = pd.read_csv(csv_file)
    texts = data['text'].tolist()
    summaries = data['summary'].tolist()
    
    tokenized_texts = [preprocess_text(nlp, text) for text in texts]
    tokenized_summaries = [preprocess_text(nlp, summary) for summary in summaries]
    
    knowledge_graph = build_knowledge_graph(tokenized_texts)
    return tokenized_texts, tokenized_summaries, knowledge_graph

if __name__ == "__main__":
    csv_file = "data/dataset.csv"
    tokenized_texts, tokenized_summaries, knowledge_graph = preprocess_data(csv_file)
