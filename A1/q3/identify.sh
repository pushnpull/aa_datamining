# script was made with the help of llm
#!/bin/bash
# identify.sh <path_graph_dataset> <path_discriminative_subgraphs>
source venv_q3/bin/activate
INPUT_DB=$1
OUTPUT_SUBGRAPHS=$2
GASTON_EXEC=$(realpath gaston-1.1/gaston) # path based on env.sh


echo "tep 1: Removin duplicats..."
python3 preprocess.py "$INPUT_DB" "cleaned_database.txt"

DB_SIZE=$(grep -c "^#" "cleaned_database.txt")
echo "Total graphs in database: $DB_SIZE"   

MIN_SUP_COUNT=$(echo "$DB_SIZE * 0.02" | bc | awk '{print int($1+0.5)}')
echo "Setting minimum support count to: $MIN_SUP_COUNT"

# 1. DB to Gaston Format
python3 converter.py "cleaned_database.txt" "gaston_input.db"

$GASTON_EXEC -m 7 -t $MIN_SUP_COUNT gaston_input.db gaston_output.txt

python3 select_features.py "gaston_output.txt" "$OUTPUT_SUBGRAPHS" 50 "$DB_SIZE"
