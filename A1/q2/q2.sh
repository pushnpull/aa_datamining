# script was made with the help of llm
#!/usr/bin/env bash
set -euo pipefail

# Usage check
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <gspan_exe> <fsg_exe> <gaston_exe> <dataset_file> <output_folder>"
    exit 1
fi

gspan_wala=$(realpath "$1")
fsg_wala=$(realpath "$2")
gaston_wala=$(realpath "$3")
dataset=$(realpath "$4")
output_dir=$(realpath "$5")

mkdir -p "$output_dir"

# converted files ke naam
fsg_input="$output_dir/yeast_fsg.txt"
gspan_input="$output_dir/yeast_gspan.txt"
gaston_input="$output_dir/yeast_gaston.txt"

# conversion chalao (python scripts current folder mein hone chahiye)
python3 convert_fsg.py "$dataset" "$fsg_input" || exit 1
python3 convert_gspan.py "$dataset" "$gspan_input" || exit 1
python3 convert_gaston.py "$dataset" "$gaston_input" || exit 1

total_graphs=$(grep -c '^#' "$dataset")

supports=(5 10 25 50 95)


for pe in "${supports[@]}"; do
    echo "  $pe% support ..."
    abs_support=$(( (pe * total_graphs) / 100 ))

    # --- 1. FSG ---
    # Binary often ignores output path and writes to [input_name].fp
    timeout 3600s /usr/bin/time -f "%e" --quiet -o "$output_dir/fsg${pe}.time" \
        "$fsg_wala" -s "$pe" "$fsg_input" > /dev/null 2>&1 || true
    
    if [ $? -eq 124 ]; then
        echo "3666" > "$output_dir/fsg${pe}.time"
    fi


    if [ -f "$output_dir/yeast_fsg.fp" ]; then mv "$output_dir/yeast_fsg.fp" "$output_dir/fsg${pe}";
    fi

    # --- 2. gSpan ---
    frac_support=$(echo "scale=2; $pe / 100" | bc)
    timeout 3600s /usr/bin/time -f "%e" --quiet -o "$output_dir/gspan${pe}.time" \
        "$gspan_wala" -f "$gspan_input" -s "$frac_support" -o "$output_dir/gspan${pe}" > /dev/null 2>&1 || true

    if [ $? -eq 124 ]; then
        echo "3666" > "$output_dir/gspan${pe}.time"
    fi

    if [ -f "$output_dir/yeast_gspan.txt.fp" ]; then
        mv "$output_dir/yeast_gspan.txt.fp" "$output_dir/gspan${pe}"
    fi

    # --- 3. Gaston ---
    timeout 3600s /usr/bin/time -f "%e" --quiet -o "$output_dir/gaston${pe}.time" \
        "$gaston_wala" "$abs_support" "$gaston_input" "$output_dir/gaston${pe}" > /dev/null 2>&1 || true

    if [ $? -eq 124 ]; then
        echo "3666" > "$output_dir/gaston${pe}.time"
    fi
done



python3 plots.py "$output_dir" "Yeast FSM Comparison"


# bash q2.sh /mnt/c/Users/abhay/Desktop/2.holi_semester/col761/A1/q2/gSpan-64 /mnt/c/Users/abhay/Desktop/2.holi_semester/col761/A1/q2/fsg/Linux/fsg /mnt/c/Users/abhay/Desktop/2.holi_semester/col761/A1/q2/gaston-1.1/gaston ./167.txt_graph 1