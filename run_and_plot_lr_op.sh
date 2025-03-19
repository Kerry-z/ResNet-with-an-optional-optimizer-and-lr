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

python3 << EOF
import re
import matplotlib.pyplot as plt

optimizers = {'sgd': [0.01, 0.1, 0.5], 'momentum': [0.01, 0.1, 0.5], 'adam': [0.001, 0.0005, 0.0001]}
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for opt, lr_list in optimizers.items():
    for lr in lr_list:
        losses, iterations = [], []
        with open(f'logs/{opt}_lr{lr}.log') as f:
            for line in f:
                if 'Train Loss:' in line:
                    iter_match = re.search(r'Iter: (\d+)', line)
                    loss_match = re.search(r'Train Loss: ([0-9.]+)', line)
                    if iter_match and loss_match:
                        iterations.append(int(iter_match.group(1)))
                        losses.append(float(loss_match.group(1)))
        if iterations:
            plt.plot(iterations, losses, label=f"{opt} lr={lr}")

plt.xlabel('Iterations')
plt.ylabel('Train Loss')
plt.title('Train Loss vs Iterations')
plt.legend()

plt.subplot(1, 2, 2)
for opt, lr_list in optimizers.items():
    for lr in lr_list:
        accs, iterations = [], []
        with open(f'logs/{opt}_lr{lr}.log') as f:
            for line in f:
                if 'Test Acc:' in line:
                    iter_match = re.search(r'Iter: (\d+)', line)
                    acc_match = re.search(r'Test Acc: ([0-9.]+)', line)
                    if iter_match and acc_match:
                        iterations.append(int(iter_match.group(1)))
                        accs.append(float(acc_match.group(1)))
        if iterations:
            plt.plot(iterations, accs, label=f"{opt} lr={lr}")

plt.xlabel('Iterations')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs Iterations')
plt.legend()

plt.tight_layout()
plt.savefig('train_loss_test_acc_plot.png')
plt.show()
EOF

echo "Plot saved as train_loss_test_acc_plot.png"

