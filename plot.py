import os
import re
import matplotlib.pyplot as plt
import chardet
from collections import defaultdict

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

log_dir = os.path.join(current_dir, 'logs') # Change this if needed
logs = [f for f in os.listdir(log_dir) if f.endswith('.log')]



data = defaultdict(lambda: defaultdict(list))  # data[optimizer][lr] = [(iter, loss), ...]


for log_file in logs:
    with open(os.path.join(log_dir, log_file), 'rb') as raw_f:
        result = chardet.detect(raw_f.read())
        encoding = result['encoding']
        print(f"Detected encoding for {log_file}: {encoding}")
    match = re.match(r'(adam|momentum|sgd)_lr([\d.]+)\.log', log_file)
    if not match:
        continue
    optimizer, lr = match.group(1), match.group(2)
    iterations, losses = [], []
    with open(os.path.join(log_dir, log_file), 'r', encoding=encoding, errors='ignore') as f:
        for line in f:
            found = re.match(r"iter:\s*(\d+), loss:\s*([\d.]+)", line)
            if found:
                iterations.append(int(found.group(1)))
                losses.append(float(found.group(2)))
    if iterations:
        data[optimizer][lr] = (iterations, losses)

# optimizer with different lr
for optimizer in data:
    plt.figure(figsize=(10, 6))
    for lr in sorted(data[optimizer].keys(), key=float):
        plt.plot(data[optimizer][lr][0], data[optimizer][lr][1], label=f'lr={lr}')
    plt.title(f'{optimizer.upper()} - Different Learning Rates')
    plt.xlabel('Iteration')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{optimizer}_lr_comparison.png')
    plt.close()

# learning rate with different optimizer
all_lrs = set()
for opt in data:
    all_lrs.update(data[opt].keys())

for lr in sorted(all_lrs, key=float):
    plt.figure(figsize=(10, 6))
    for opt in data:
        if lr in data[opt]:
            plt.plot(data[opt][lr][0], data[opt][lr][1], label=f'{opt}')
    plt.title(f'Learning Rate = {lr} - Optimizer Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'optimizer_comparison_lr{lr}.png')
    plt.close()
