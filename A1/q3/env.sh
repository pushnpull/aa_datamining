# script was made with the help of llm

#!/bin/bash

if command -v deactivate &> /dev/null; then deactivate; fi

python3 -m venv venv_q3
source venv_q3/bin/activate

pip install --upgrade pip
pip install numpy networkx matplotlib

cd gaston-1.1
make clean
make
cd ..

export GASTON_PATH=$(realpath gaston-1.1/gaston)