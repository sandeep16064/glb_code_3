# models/knowledge_graph.py
import spacy
import networkx as nx

def build_knowledge_graph(docs):
    graph = nx.DiGraph()
    for doc in docs:
        for ent in doc.ents:
            graph.add_node(ent.text, label=ent.label_)
            for token in doc:
                if token.head == ent:
                    graph.add_edge(ent.text, token.head.text, relation=token.dep_)
    return graph
