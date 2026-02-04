#!/bin/bash
source venv_q3/bin/activate
# convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>

IN=$1
SUBGRAPH=$2
FEATURES=$3

python3 feature_mapper.py "$IN" "$SUBGRAPH" "$FEATURES"