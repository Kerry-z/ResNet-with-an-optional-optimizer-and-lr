#!/bin/bash

PYTHON_BIN="/e/CodeTool/Anaconda/Anaconda/python.exe"
echo "Using Python: $PYTHON_BIN"

mkdir -p logs

DEPTH=3
N_ITER=64000

declare -A lrs
lrs=( ["sgd"]="0.01 0.1 0.5" ["momentum"]="0.01 0.1 0.5" ["adam"]="0.001 0.0005 0.0001" )

for opt in sgd momentum adam; do
  for lr in ${lrs[$opt]}; do
    echo "Running $opt with lr=$lr"
    $PYTHON_BIN train.py --n $DEPTH --n_iter $N_ITER --optimizer $opt --lr $lr \
      --checkpoint_dir ./checkpoint_${opt}_${lr} > logs/${opt}_lr${lr}.log
  done
done

