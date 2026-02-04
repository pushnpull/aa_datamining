# script was made with the help of llm

import sys

if len(sys.argv) not in (2, 3):
    print("Usage: python3 convert_gspan.py <input_yeast_file> [output_file]")
    sys.exit(1)

input_wali_file = sys.argv[1]

# Default output naam set kar diya
if len(sys.argv) == 3:
    output_wali_file = sys.argv[2]
else:
    output_wali_file = "yeast_fsg.txt"

kitne_graph_bane = 0

with open(input_wali_file, 'r') as fin, open(output_wali_file, 'w') as fout:
    for line in fin:
        line = line.strip()
        
        if line.startswith('#'):
            # Naya graph shuru ho raha hai
            graph_ka_id = line[1:]          # # ke baad wala part
            fout.write(f"t # {graph_ka_id}\n")
            kitne_graph_bane += 1
            
            # Agli line me nodes ki count aani chahiye
            nodes_ki_line = fin.readline().strip()
            if not nodes_ki_line.isdigit():
                print(f"Error: nodes ki quantitiy chahiye thi, mila '{nodes_ki_line}'")
                sys.exit(1)
            
            kitne_nodes = int(nodes_ki_line)
            
            # Ab saare nodes print karte hai (0 se shuru)
            for i in range(kitne_nodes):
                label = fin.readline().strip()
                fout.write(f"v {i} {label}\n")
            
            # Ab edges ki quantitiy
            edges_ki_line = fin.readline().strip()
            if not edges_ki_line.isdigit():
                print(f"Error: edges ki quantitiy chahiye thi, mila '{edges_ki_line}'")
                sys.exit(1)
            
            kitne_edges = int(edges_ki_line)
            
            # Edges ko set me rakhte hai taaki duplicate na aaye
            edges_set = set()
            
            for _ in range(kitne_edges):
                edge_line = fin.readline().strip()
                if not edge_line:
                    continue
                    
                parts = edge_line.split()
                if len(parts) != 3:
                    print(f"Galat edge line mili: '{edge_line}' (3 cheeze chahiye thi)")
                    sys.exit(1)
                
                src = int(parts[0])
                dst = int(parts[1])
                label = parts[2]
                
                # Chhota number pehle rakhenge (undirected ke liye)
                if src > dst:
                    src, dst = dst, src
                
                edges_set.add((src, dst, label))
            
            # Ab sorted order me likh dete hai
            for src, dst, label in sorted(edges_set):
                fout.write(f"u {src} {dst} {label}\n")

print(f" Output file ban gaya: {output_wali_file}")
print(f"Total {kitne_graph_bane} graphs converted")