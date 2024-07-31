# utils/graph_utils.py

import networkx as nx
import spacy

def build_knowledge_graph(docs):
    """
    Build a knowledge graph from a list of tokenized texts.
    
    Parameters:
    - docs: List of spacy tokenized documents.
    
    Returns:
    - graph: A networkx directed graph with entities as nodes and relationships as edges.
    """
    graph = nx.DiGraph()
    for doc in docs:
        for ent in doc.ents:
            graph.add_node(ent.text, label=ent.label_)
            for token in doc:
                if token.head == ent:
                    graph.add_edge(ent.text, token.head.text, relation=token.dep_)
    return graph

def add_entity(graph, entity, label):
    """
    Add an entity to the knowledge graph.
    
    Parameters:
    - graph: The knowledge graph.
    - entity: The entity to add.
    - label: The label of the entity.
    """
    if entity not in graph:
        graph.add_node(entity, label=label)

def add_relationship(graph, entity1, entity2, relation):
    """
    Add a relationship between two entities in the knowledge graph.
    
    Parameters:
    - graph: The knowledge graph.
    - entity1: The first entity.
    - entity2: The second entity.
    - relation: The relationship between the entities.
    """
    graph.add_edge(entity1, entity2, relation=relation)
