# script made with help of llm

import sys
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import multiprocessing
import os

def graphs_load_karo(filepath):
    """standard format se graphs banata hai"""
    saare_graphs = []
    abhi_wala_graph = None
    
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts: 
                continue
                
            if parts[0] in ['#', 't']:
                if abhi_wala_graph:
                    saare_graphs.append(abhi_wala_graph)
                abhi_wala_graph = nx.Graph()
                
            elif parts[0] == 'v':
                abhi_wala_graph.add_node(int(parts[1]), label=parts[2])
                
            elif parts[0] == 'e':
                abhi_wala_graph.add_edge(int(parts[1]), int(parts[2]), label=parts[3])
    
    if abhi_wala_graph:
        saare_graphs.append(abhi_wala_graph)
        
    return saare_graphs


def ek_graph_check_karo(args):
    """ek graph ko saare features ke saath match karta hai"""
    graph, feature_list = args
    
    node_match = isomorphism.categorical_node_match('label', None)
    edge_match = isomorphism.categorical_edge_match('label', None)
    
    row = np.zeros(len(feature_list), dtype=int)
    
    for idx, feature in enumerate(feature_list):
        matcher = isomorphism.GraphMatcher(graph, feature, 
                                          node_match=node_match, 
                                          edge_match=edge_match)
                                          
        if matcher.subgraph_is_isomorphic():
            row[idx] = 1
            
    return row


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 feature_mapper.py <graphs_file> <features_file> <output.npy>")
        return

    target_graphs = graphs_load_karo(sys.argv[1])
    feature_subgraphs = graphs_load_karo(sys.argv[2])
    
    cores = os.cpu_count()
    print(f"{cores} cores detect hue... parallel processing shuru karte hain")

    tasks = [(g, feature_subgraphs) for g in target_graphs]

    with multiprocessing.Pool(processes=cores) as pool:
        results = pool.map(ek_graph_check_karo, tasks)

    feature_matrix = np.array(results)
    
    np.save(sys.argv[3], feature_matrix)
    print(f"kaam khatam! {feature_matrix.shape} shape ki matrix save hui → {sys.argv[3]}")


if __name__ == "__main__":
    main()