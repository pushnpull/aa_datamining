# script made with help of llm
import sys

def graph_ka_signature(lines):
    # Graph ka ek unique "fingerprint" banate hain duplicate pakadne ke liye
    nodes = []
    edges = []
    
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        
        if parts[0] == 'v':
            # v <id> <label>
            nodes.append(parts[2])   # sirf label rakh rahe
            
        elif parts[0] == 'e':
            # e <u> <v> <label>
            u = int(parts[1])
            v = int(parts[2])
            label = parts[3]
            
            # undirected ke liye chhota → bada kar do
            if u > v:
                u, v = v, u
                
            edges.append((u, v, label))

    # sorting karke canonical bana dete hain
    nodes_ka_tuple = tuple(nodes)
    edges_sorted = tuple(sorted(edges))
    
    return (len(nodes), nodes_ka_tuple, edges_sorted)


def preprocessing_karo(input_file, output_file):
    print(f"{input_file} ko process kar rahe hain bhai...")

    with open(input_file, 'r') as f:
        saari_lines = f.readlines()

    unique_fingerprint = set()
    achhe_graphs = []
    current_graph_lines = []
    kitne_graph_mile = 0
    kitne_duplicate_the = 0
    
    for line in saari_lines:
        clean_line = line.strip()
        if not clean_line: continue
        
        if clean_line.startswith('#') or clean_line.startswith('t #'):
            # purana graph khatam → check karo
            if current_graph_lines:
                sign = graph_ka_signature(current_graph_lines)
                
                if sign not in unique_fingerprint:
                    unique_fingerprint.add(sign)
                    achhe_graphs.append(current_graph_lines)
                else:
                    kitne_duplicate_the += 1
            
            # naya graph shuru
            current_graph_lines = []
            kitne_graph_mile += 1
        else:
            current_graph_lines.append(line)
    
    # aakhri wala graph bhi check karna zaroori hai
    if current_graph_lines:
        sign = graph_ka_signature(current_graph_lines)
        if sign not in unique_fingerprint:
            achhe_graphs.append(current_graph_lines)
        else:
            kitne_duplicate_the += 1

    print(f"Total graphs mile: {kitne_graph_mile}")
    print(f"Duplicate hata diye: {kitne_duplicate_the}")
    print(f"Unique graphs bache: {len(achhe_graphs)}")
    
    # thodi si safety
    if len(achhe_graphs) <= 1 and len(saari_lines) > 100:
        print(" # separator nahi hai → Gaston crash karega")
        return

    # output file mein likh do
    with open(output_file, 'w') as f:
        for graph_lines in achhe_graphs:
            f.write("#\n")
            for line in graph_lines:
                f.write(line)
    
    print(f"ho gaya → {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python3 preprocess.py <input.txt> <clean_output.txt>")
    else:
        preprocessing_karo(sys.argv[1], sys.argv[2])
