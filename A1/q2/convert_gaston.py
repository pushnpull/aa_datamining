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
    output_wali_file = "yeast_gaston.txt"



# Vertex ke labels ke liye map (string → unique number)
label_ka_map = {}
agli_id = 1          # 1 se shuru karenge

kitne_graph = 0

with open(input_wali_file, 'r') as fin, open(output_wali_file, 'w') as fout:
    for line in fin:
        line = line.strip()
        if line.startswith('#'):
            # Naya graph shuru ho gaya
            graph_id = line[1:]
            fout.write(f"t # {graph_id}\n")
            kitne_graph += 1
            
            # Kitne vertices hain
            vertices_ki_count = int(fin.readline().strip())
            
            # Har vertex ka label padho aur number assign karo
            for vid in range(vertices_ki_count):
                label_str = fin.readline().strip()
                if label_str not in label_ka_map:
                    label_ka_map[label_str] = agli_id
                    agli_id += 1
                label_number = label_ka_map[label_str]
                fout.write(f"v {vid} {label_number}\n")
            
            # Kitni edges hain
            edges_ki_count = int(fin.readline().strip())
            
            # Edges padho, sort karo (src < dst), duplicate hatao
            edges_set = set()
            for _ in range(edges_ki_count):
                parts = fin.readline().strip().split()
                src = int(parts[0])
                dst = int(parts[1])
                label = parts[2]   # label ko waise hi rakhenge
                if src > dst:
                    src, dst = dst, src
                edges_set.add((src, dst, label))
            
            # Sorted edges likh do
            for src, dst, label in sorted(edges_set):
                fout.write(f"e {src} {dst} {label}\n")

print(f" {kitne_graph} graphs process kiye.")
for label_text, number in sorted(label_ka_map.items(), key=lambda x: x[1]):
    print(f"  {label_text} → {number}")