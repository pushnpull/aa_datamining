import sys
import networkx as nx
from networkx.algorithms import isomorphism

def load_graphs(filepath):
    """Parses the assignment graph format into a list of NetworkX graphs."""
    graphs = []
    current_g = None
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == '#' or parts[0] == 't': 
                if current_g is not None: graphs.append(current_g)
                current_g = nx.Graph()
            elif parts[0] == 'v':
                current_g.add_node(int(parts[1]), label=parts[2])
            elif parts[0] == 'e':
                current_g.add_edge(int(parts[1]), int(parts[2]), label=parts[3])
    if current_g: graphs.append(current_g)
    return graphs

def is_subgraph(query, graph):
    """Checks if query is a subgraph of graph (node/edge labels must match)."""
    nm = isomorphism.categorical_node_match('label', None)
    em = isomorphism.categorical_edge_match('label', None)
    gm = isomorphism.GraphMatcher(graph, query, node_match=nm, edge_match=em)
    return gm.subgraph_is_isomorphic()

def parse_candidates(filepath):
    """Parses candidates.dat into a dictionary: {query_id: [list of candidate_ids]}."""
    results = {}
    current_q = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('q #'):
                current_q = int(line.split('#')[1].strip())
            elif line.startswith('c #'):
                parts = line.split('#')[1].strip().split()
                results[current_q] = [int(p) for p in parts]
    return results

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 check_score.py <db_file> <query_file> <candidates_file>")
        return

    print("Loading database and query graphs...")
    db = load_graphs(sys.argv[1])
    queries = load_graphs(sys.argv[2])
    candidates = parse_candidates(sys.argv[3])

    total_sq = 0
    num_queries = len(queries)

    print(f"{'Query':<8} | {'|Rq|':<6} | {'|Cq|':<6} | {'sq (Score)':<10}")
    print("-" * 40)

    for q_id in range(num_queries):
        if q_id not in candidates:
            print(f"Warning: Query {q_id} missing in candidates file.")
            continue
        
        q_graph = queries[q_id]
        cand_list = candidates[q_id]
        
        # Ground Truth: Find all graphs that ACTUALLY contain the query
        # Since Cq must be a superset of Rq, we only need to check candidates.
        # If your indexing is correct, Rq is a subset of Cq.
        actual_matches = 0
        for db_id in cand_list:
            if is_subgraph(q_graph, db[db_id]):
                actual_matches += 1
        
        cq_size = len(cand_list)
        sq = actual_matches / cq_size if cq_size > 0 else 0
        total_sq += sq
        
        print(f"{q_id:<8} | {actual_matches:<6} | {cq_size:<6} | {sq:.4f}")

    avg_score = total_sq / num_queries if num_queries > 0 else 0
    print("-" * 40)
    print(f"Average s_q Score: {avg_score:.4f}")
    print("\nNote: Higher s_q means better competitive performance.")

if __name__ == "__main__":
    main()