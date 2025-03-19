import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import glob

logs = glob.glob('checkpoint_*/loss_log.txt')
data = []

for log_file in logs:
    optimizer_lr = log_file.split(os.sep)[0].replace('checkpoint_', '')
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(
                r'Iter: (\d+).*?Train Loss: ([0-9.]+).*?Test Loss: ([0-9.]+).*?Train Acc: ([0-9.]+).*?Test Acc: ([0-9.]+)', 
                line
            )
            if match:
                iteration = int(match.group(1))
                train_loss = float(match.group(2))
                test_loss = float(match.group(3))
                train_acc = float(match.group(4))
                test_acc = float(match.group(5))
                data.append({
                    'optimizer_lr': optimizer_lr,
                    'iter': iteration,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_error': 100 - train_acc,
                    'test_error': 100 - test_acc
                })

df = pd.DataFrame(data)
df.to_csv('train_test_report.csv', index=False)
print("✅ Saved detailed report to train_test_report.csv")

# Plot Train/Test Loss
plt.figure(figsize=(14, 6))
for name, group in df.groupby('optimizer_lr'):
    plt.plot(group['iter'], group['train_loss'], label=f'{name} Train Loss', linestyle='-')
    plt.plot(group['iter'], group['test_loss'], label=f'{name} Test Loss', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Test Loss vs Iteration')
plt.savefig('loss_plot.png')
plt.show()

# Plot Train/Test Error
plt.figure(figsize=(14, 6))
for name, group in df.groupby('optimizer_lr'):
    plt.plot(group['iter'], group['train_error'], label=f'{name} Train Error', linestyle='-')
    plt.plot(group['iter'], group['test_error'], label=f'{name} Test Error', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Error (%)')
plt.legend()
plt.title('Train and Test Error vs Iteration')
plt.savefig('error_plot.png')
plt.show()

