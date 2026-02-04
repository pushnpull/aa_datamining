# script was made with the help of llm
import sys
import os
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print(" python3 plot_runtimes.py <results_folder> <title_ka_pichhe_ka_text>")
    sys.exit(1)

results_ka_folder = sys.argv[1]
title_ka_extra_text = sys.argv[2]

# ye supports jo compare karne hain
support_values = [5, 10, 25, 50, 95]
algorithms = ['fsg', 'gspan', 'gaston']

# har algo ke time ko store karne ke liye
runtime_data = {algo: [] for algo in algorithms}

# har support ke liye time file se padh rahe
for support in support_values:
    for algo in algorithms:
        time_file = os.path.join(results_ka_folder, f"{algo}{support}.time")
        try:
            with open(time_file) as f:
                time_value = float(f.read().strip())
                runtime_data[algo].append(time_value)
        except:
            print(f"Time file nahi mili: {time_file} → 0 daal rahe")
            runtime_data[algo].append(0.0)

# ab plot banate hain
plt.figure(figsize=(10, 6))

for algo in algorithms:
    plt.plot(support_values, runtime_data[algo], marker='o', label=algo.upper())

plt.xlabel("Minimum support (%)")
plt.ylabel("Runtime (seconds)")
plt.title(f"Runtime comparison - {title_ka_extra_text}")
plt.yscale('log')          # log scale best hai runtime ke liye
plt.grid(True)
plt.legend()

# plot ko save kar dete hain
output_plot_file = os.path.join(results_ka_folder, "plot.png")
plt.savefig(output_plot_file, dpi=300)
plt.close()

