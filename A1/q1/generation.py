# script was made with the help of llm

import random
import sys

if len(sys.argv) != 4:
    print("Usage: python3 generation.py <totl_items> <num_transctions> <outpt_file>")
    print("Example: python3 generation.py 10000 15000 new.dat")
    print("Galat input, aise chalao ↑")
    sys.exit(1)

total_cheezein     = int(sys.argv[1])          # kitne alag-alag items honge
kitne_billing      = int(sys.argv[2])          # total transactions / bills
output_file_ka_naam = sys.argv[3]              # jahan save karna hai


alpha = 1.35                  # thoda sa skew, bahut zyada nahi
random_seed = 42424223          # same result chahiye har baar toh yahi rakhna
random.seed(random_seed)

saari_cheezein = list(range(1, total_cheezein + 1))

# Zipf wala weight (mild skew)
weights = [1.0 / (rank ** alpha) for rank in range(1, total_cheezein + 1)]
total_weight = sum(weights)
probability_list = [w / total_weight for w in weights]

# ek bill mein kitne items hone chahiye — intentionally lambi bills
def bill_ki_length():
    r = random.random()
    if r < 0.35:
        return random.randint(60, 110)
    elif r < 0.75:
        return random.randint(111, 180)
    elif r < 0.95:
        return random.randint(181, 280)
    else:
        return random.randint(281, 450)   # bahut kam log itna bada basket lete hain



with open(output_file_ka_naam, 'w', encoding='utf-8') as file:
    for i in range(kitne_billing):
        length = bill_ki_length()

        # replacement ke saath sample (Zipf probability se)
        jo_cheezein_mili = random.choices(saari_cheezein, weights=probability_list, k=length)

        # ek hi bill mein duplicate nahi chahiye
        unique_cheezein = sorted(set(jo_cheezein_mili))

        if unique_cheezein:
            file.write(" ".join(map(str, unique_cheezein)) + "\n")

        if (i + 1) % 1000 == 0:
            print(f"  {i+1:6d} / {kitne_billing}", end='\r')

print(f"  File→ {output_file_ka_naam}")
print(f"  Total bills    → {kitne_billing}")
print(f"  Total items    → {total_cheezein}")