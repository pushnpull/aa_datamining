# script was made with the help of llm

import sys
import os
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("Usage: python3 plot_runtimes.py <results_directory> <title_suffix>")
    print("Example: python3 plot_runtimes.py ./experiment1 'Long Baskets Dataset'")
    print("Bhai sahi tarike se arguments de na!")
    sys.exit(1)

folder_ka_path     = sys.argv[1]         # jahan .time files pade hain
title_ke_peeche    = sys.argv[2]         # graph ke title mein kya extra likhna hai

supports = [5, 10, 25, 50, 90]           # minimum support % jo try kiye the

apriori_ke_time    = []                  # Apriori ke saare runtime
fpgrowth_ke_time   = []                  # FP-Growth ke saare runtime

for min_sup in supports:
    ap_file = os.path.join(folder_ka_path, f"ap{min_sup}.time")
    fp_file = os.path.join(folder_ka_path, f"fp{min_sup}.time")

    try:
        with open(ap_file) as f:
            time_value = float(f.read().strip())
            apriori_ke_time.append(time_value)
    except:
        print(f"Warning: {ap_file} nahi mila ya kharab hai → 0 daal raha hoon")
        apriori_ke_time.append(0.0)

    try:
        with open(fp_file) as f:
            time_value = float(f.read().strip())
            fpgrowth_ke_time.append(time_value)
    except:
        print(f"Warning: {fp_file} nahi mila → 0 daal raha hoon")
        fpgrowth_ke_time.append(0.0)


plt.figure(figsize=(10, 6))              # thoda bada banate hain, achha dikhega

plt.plot(supports, apriori_ke_time, marker='o', label='Apriori', linewidth=2.5, color='#e74c3c')
plt.plot(supports, fpgrowth_ke_time, marker='s', label='FP-Growth', linewidth=2.5, color='#27ae60')

plt.xlabel("Minimum Support (%)")
plt.ylabel("Runtime (sec) (log scale)")

main_title = f"Runtime Comparison – {title_ke_peeche}"
plt.title(main_title, fontsize=14, fontweight='bold')

plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12)

# Zyadatar aise experiments mein time bahut spread hota hai → log scale zaroori!
plt.yscale('log')

plt.tight_layout()

output_plot = os.path.join(folder_ka_path, "plot.png")
plt.savefig(output_plot, dpi=300, bbox_inches='tight')   # 6000 dpi thoda zyada ho jata hai heavy
plt.close()

print(f"Graph save ho gaya yahan: {output_plot}")
print(f"Supports jo plot hue: {supports}")
print(f"Title: {main_title}")