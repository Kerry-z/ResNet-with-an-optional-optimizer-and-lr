import os
import re
import matplotlib.pyplot as plt
import chardet

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

log_dir = os.path.join(current_dir, 'logs') # Change this if needed
logs = [f for f in os.listdir(log_dir) if f.endswith('.log')]

results = []

for log_file in logs:
    with open(os.path.join(log_dir, log_file), 'rb') as raw_f:
        result = chardet.detect(raw_f.read())
        encoding = result['encoding']
        print(f"Detected encoding for {log_file}: {encoding}")
    match = re.match(r'(adam|momentum|sgd)_lr([\d.]+)\.log', log_file)
    with open(os.path.join(log_dir, log_file), 'r', encoding=encoding, errors='ignore') as f:
        content = f.read()
        match = re.findall(r"accuracy:\s*([\d.]+)\s*%", content)
        if match:
            final_acc = float(match[-1])  # final accuracy
            final_error = 100 - final_acc
            results.append({
                'method': log_file.replace('.log', ''),
                'accuracy': final_acc,
                'error': final_error
            })


results.sort(key=lambda x: x['method'])

methods = [r['method'] for r in results]
accuracies = [r['accuracy'] for r in results]

plt.figure(figsize=(12, 6))
bars = plt.bar(methods, accuracies, color='skyblue')

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=45, ha='right')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy Comparison Across Optimizers and Learning Rates')
plt.tight_layout()
plt.savefig('test_accuracy_comparison.png')
plt.show()


