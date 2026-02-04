#!/bin/bash
# generate_candidates.sh <db_features> <query_features> <output_file>
source venv_q3/bin/activate
DB_FEATS=$1
QUERY_FEATS=$2
OUTPUT=$3

python3 filter_candidates.py "$DB_FEATS" "$QUERY_FEATS" "$OUTPUT"