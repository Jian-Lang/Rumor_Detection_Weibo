"""
@author: Lobster
@software: PyCharm
@file: grid_search.py
@time: 2023/11/24 16:42
"""
from itertools import product
import numpy as np

# 模拟一些数据和模型

def train_and_evaluate(hyperparameters):

    accuracy = np.random.rand()
    return accuracy

# 定义超参数搜索空间
hyperparameter_space = {
    'param1': [0.1, 0.2, 0.3],
    'param2': [0.4, 0.5, 0.6],
    'param3': [0.7, 0.8, 0.9],
}

best_accuracy = 0.0
best_hyperparameters = None

# 遍历超参数组合
for hyperparameter_combination in product(*hyperparameter_space.values()):
    hyperparameters = dict(zip(hyperparameter_space.keys(), hyperparameter_combination))

    # 归一化确保和为1
    hyperparameters_normalized = {k: v / sum(hyperparameters.values()) for k, v in hyperparameters.items()}

    # 训练和评估模型
    accuracy = train_and_evaluate(hyperparameters_normalized)

    # 更新最佳超参数组合
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_hyperparameters = hyperparameters_normalized

print("Best Hyperparameters:", best_hyperparameters)
print("Best Accuracy:", best_accuracy)
