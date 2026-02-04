# script was made with the help of llm

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "sahi se chalao script!"
    echo "Usage: $0 <apriori_exe> <fpgrowth_exe> <dataset_path> <output_dir>"
    echo ""
    echo "Example (original dataset ke liye):"
    echo "  $0 ./apriori ./fpgrowth ../webdocs.dat ./results_webdocs"
    echo ""
    echo "Example (synthetic / generated dataset ke liye):"
    echo "  $0 ./apriori ./fpgrowth generated_transactions.dat ./results_synthetic"
    exit 1
fi

APRIORI_WALA="$1"           # apriori executable ka path
FPGROWTH_WALA="$2"          # fp-growth executable ka path
DATASET_FILE="$3"           # dataset jisme transactions hain
OUTPUT_FOLDER="$4"          # saare results yahan save honge

mkdir -p "$OUTPUT_FOLDER"

supports=(5 10 25 50 90)    # in % minimum support pe test karenge

echo "Chal raha hai experiment → $(basename "$DATASET_FILE") pe"
echo "Supports try karenge: ${supports[*]}%"

for sup in "${supports[@]}"; do
    echo "→ Support ${sup}% pe chal raha hai..."

    # Apriori chalate hain
    timeout 3600s /usr/bin/time -f "%e" --quiet -o "$OUTPUT_FOLDER/ap${sup}.time" \
        "$APRIORI_WALA" -s${sup} "$DATASET_FILE" "$OUTPUT_FOLDER/ap${sup}" > /dev/null 2>&1 || true

    if [ $? -eq 124 ]; then
        echo "  Apriori ${sup}% pe 1 ghante mein nahi khatam hua → timeout!"
        echo "3666" > "$OUTPUT_FOLDER/ap${sup}.time"
    fi

    # FP-Growth chalate hain
    timeout 3600s /usr/bin/time -f "%e" --quiet -o "$OUTPUT_FOLDER/fp${sup}.time" \
        "$FPGROWTH_WALA" -s${sup} "$DATASET_FILE" "$OUTPUT_FOLDER/fp${sup}" > /dev/null 2>&1 || true

    if [ $? -eq 124 ]; then
        echo "  FP-Growth ${sup}% pe bhi timeout ho gaya → slow hai bhai!"
        echo "3666" > "$OUTPUT_FOLDER/fp${sup}.time"
    fi

done



python3 plots.py "$OUTPUT_FOLDER" "$(basename "$DATASET_FILE" .dat)"

echo "Graph ban gaya → $OUTPUT_FOLDER/plot.png"   # (agar plot script mein naam yahi rakha hai)
