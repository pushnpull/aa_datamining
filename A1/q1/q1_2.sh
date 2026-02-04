# script was made with the help of llm

#!/usr/bin/env bash
set -euo pipefail

# Check kiya inputs sahi diye ya nahi
if [ "$#" -ne 2 ]; then
    echo "sahi tarike se chalao na!"
    echo "Usage: $0 <total_unique_items> <num_transactions>"
    echo "  $0 1000 15000"
    exit 1
fi

TOTAL_CHEEZEIN="$1"      # kitne alag-alag items hone chahiye
KITNE_BILL="$2"          # kitne customers/bills generate karne hain
OUTPUT_FILE="synthetic_billing_data.dat"

echo "fake shopping dataset"
echo "→ ${KITNE_BILL} bills banenge"
echo "→ ${TOTAL_CHEEZEIN} alag-alag items honge"

# Python wala script call kar rahe hain
python3 generation.py "$TOTAL_CHEEZEIN" "$KITNE_BILL" "$OUTPUT_FILE"

# Check karo file bani ya nahi
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "File nahi bani: $OUTPUT_FILE"
    exit 1
fi

