# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttention(nn.Module):
    def __init__(self, d_model):
        super(GraphAttention, self).__init__()
        self.d_model = d_model
    
    def forward(self, query, key, value, graph):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Modify attention scores based on the knowledge graph
        for i, node in enumerate(graph.nodes):
            for neighbor in graph[node]:
                attention_scores[i, neighbor] += 1  # Simple example of boosting scores
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output
