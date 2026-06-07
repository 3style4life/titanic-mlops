import torch
import torch.nn as nn

class IoTLSTMClassifier(nn.Module):
    """
    经典工业级：LSTM 时间序列分类网络
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(IoTLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. LSTM 层：提取时间序列的前后依赖关系、趋势和周期性
        # batch_first=True 确保输入格式为 (batch, seq_len, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)

        # 2. 全连接层：将 LSTM 提取的时序特征映射到最终的类别上
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 初始化隐藏状态 h0 和细胞状态 c0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 前向传播经过 LSTM，out 的形状: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))

        # 我们只需要时间序列【最后一个时间步】的输出用来做分类决策
        out = out[:, -1, :]

        # 经过全连接层
        out = self.fc(out)
        return out