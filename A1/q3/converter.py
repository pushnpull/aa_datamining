# script made with help of llm
import sys

def gaston_ke_liye(input_file, output_file):
    with open(input_file, 'r') as f:
        saari_lines = f.readlines()
    
    with open(output_file, 'w') as f:
        graph_number = 0
        node_id_naya = {}   # purana id → naya 0 se start wala
        
        for line in saari_lines:
            parts = line.strip().split()
            if not parts: 
                continue
            
            if parts[0] == '#':
                # naya graph shuru
                f.write(f"t # {graph_number}\n")
                graph_number += 1
                node_id_naya = {}   # har naye graph ke liye reset
            elif parts[0] == 'v':
                # node wala line
                purana_id = parts[1]
                label = parts[2]
                # naya continuous id de rahe
                node_id_naya[purana_id] = str(len(node_id_naya))
                f.write(f"v {node_id_naya[purana_id]} {label}\n")
            elif parts[0] == 'e':
                # edge wala line
                u = parts[1]
                v = parts[2]
                label = parts[3]
                f.write(f"e {node_id_naya[u]} {node_id_naya[v]} {label}\n")


if __name__ == "__main__":
    # python script.py input.txt output.gaston
    gaston_ke_liye(sys.argv[1], sys.argv[2])