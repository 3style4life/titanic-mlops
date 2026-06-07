import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TitanicDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def preprocess_data(data_path, is_train=True):
    df = pd.read_csv(data_path)

    # 填充缺失值
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    # 构造交叉特征
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)

    # 独热编码
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    feature_cols = ['Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Sex_male', 'Embarked_Q', 'Embarked_S']

    for col in feature_cols:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    if is_train:
        return df[feature_cols], df['Survived']
    return df[feature_cols], None