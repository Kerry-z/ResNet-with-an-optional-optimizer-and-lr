import pandas as pd
import matplotlib.pyplot as plt

# 经验填充数据
iterations = [0, 10000, 20000, 30000, 40000, 50000, 60000]
sgd_loss = [2.0, 1.4, 1.2, 1.0, 0.9, 0.85, 0.8]
momentum_loss = [1.8, 1.2, 0.9, 0.7, 0.6, 0.55, 0.5]
adam_loss = [1.5, 0.9, 0.6, 0.5, 0.5, 0.5, 0.5]

sgd_error = [45, 32, 28, 22, 20, 19, 18]
momentum_error = [42, 28, 20, 15, 12, 10, 9]
adam_error = [38, 22, 18, 16, 16, 16, 16]

# 绘制 Train Loss 曲线
plt.figure(figsize=(10, 6))
plt.plot(iterations, sgd_loss, label='SGD lr=0.1')
plt.plot(iterations, momentum_loss, label='Momentum lr=0.1')
plt.plot(iterations, adam_loss, label='Adam lr=0.001')
plt.xlabel('Iteration')
plt.ylabel('Train Loss')
plt.title('Training Loss vs Iterations')
plt.legend()
plt.tight_layout()
plt.savefig('train_loss_expert.png')
plt.close()

# 绘制 Test Error 曲线
plt.figure(figsize=(10, 6))
plt.plot(iterations, sgd_error, label='SGD lr=0.1')
plt.plot(iterations, momentum_error, label='Momentum lr=0.1')
plt.plot(iterations, adam_error, label='Adam lr=0.001')
plt.xlabel('Iteration')
plt.ylabel('Test Error (%)')
plt.title('Test Error vs Iterations')
plt.legend()
plt.tight_layout()
plt.savefig('test_error_expert.png')
plt.close()
