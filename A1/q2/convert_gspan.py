# script was made with the help of llm
import sys

if len(sys.argv) not in (2, 3):
    print("Usage: python3 convert_gspan.py <input_yeast_file> [output_file]")
    sys.exit(1)

input_file_ka_rasta = sys.argv[1]

# Default output naam set kar diya
if len(sys.argv) == 3:
    output_file_ka_rasta = sys.argv[2]
else:
    output_file_ka_rasta = "yeast_gspan.txt"

kitne_graph_bane = 0
kitni_galtiyaan_hui = 0

# saare graphs ke liye ek hi label mapping chahiye
vertex_label_ka_naksha = {}
agli_label_id = 0   # gSpan ko 0 se shuru pasand hai

with open(input_file_ka_rasta, 'r') as fin, open(output_file_ka_rasta, 'w') as fout:
    while True:
        line = fin.readline()
        if not line:
            break
        line = line.strip()

        if line.startswith('#'):
            graph_ka_id = line[1:].strip()
            fout.write(f"t # {graph_ka_id}\n")
            kitne_graph_bane += 1

            vertices_ki_sankhya_str = fin.readline().strip()
            try:
                kitne_vertices = int(vertices_ki_sankhya_str)
            except:
                print(f"Graph {graph_ka_id} mein gadbad → vertices count galat: '{vertices_ki_sankhya_str}'")
                kitni_galtiyaan_hui += 1
                continue

            # har vertex ka label daal rahe
            for vid in range(kitne_vertices):
                label_text = fin.readline().strip()
                if not label_text:
                    print(f"Warning: graph {graph_ka_id} mein vertex {vid} ka label khali hai")
                    continue

                if label_text not in vertex_label_ka_naksha:
                    vertex_label_ka_naksha[label_text] = agli_label_id
                    agli_label_id += 1

                label_number = vertex_label_ka_naksha[label_text]
                fout.write(f"v {vid} {label_number}\n")

            edges_ki_sankhya_str = fin.readline().strip()
            try:
                kitne_edges = int(edges_ki_sankhya_str)
            except:
                print(f"Graph {graph_ka_id} mein edges count galat: '{edges_ki_sankhya_str}'")
                kitni_galtiyaan_hui += 1
                continue

            edges_list = []
            for _ in range(kitne_edges):
                edge_line = fin.readline().strip()
                parts = edge_line.split()
                if len(parts) != 3:
                    print(f"Galat edge line → graph {graph_ka_id}: '{edge_line}'")
                    continue
                try:
                    src = int(parts[0])
                    dst = int(parts[1])
                    edge_label = int(parts[2])   # edge label bhi number chahiye
                    if src > dst:
                        src, dst = dst, src     # chhota pehle
                    edges_list.append((src, dst, edge_label))
                except:
                    print(f"Edge samajh nahi aaya → graph {graph_ka_id}: '{edge_line}'")

            # sort karke likh dete hain (gSpan style)
            for src, dst, elabel in sorted(edges_list):
                fout.write(f"e {src} {dst} {elabel}\n")

print(f"Total graphs      : {kitne_graph_bane}")
print(f"skip hue : {kitni_galtiyaan_hui}")
for label, number in sorted(vertex_label_ka_naksha.items(), key=lambda x: x[1]):
    print(f"  {label} → {number}")
print(f"Output file ban gaya  : {output_file_ka_rasta}")