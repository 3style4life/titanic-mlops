import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class IoTSymmetricDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sliding_windows(df, sequence_length):
    """
    直接读取 DataFrame 并切分为时序滑动窗口（已修复 Pandas 索引兼容问题）
    """
    # 核心修复：如果是 DataFrame，立刻提取其底层的 NumPy 数组
    if isinstance(df, pd.DataFrame):
        data = df.values
    else:
        data = df

    X, y = [], []

    # 此时 data 已经是纯 NumPy 数组，[:, :-1] 切片绝对不会报错
    raw_features = data[:, :-1]  # 前面的所有列是时间步电压特征
    raw_labels = data[:, -1]     # 最后一列是类别标签

    for i in range(len(data) - sequence_length):
        window = raw_features[i : i + sequence_length]
        label = raw_labels[i + sequence_length - 1]

        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)