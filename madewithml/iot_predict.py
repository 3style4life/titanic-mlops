import torch
import numpy as np
from madewithml.iot_models import IoTLSTMClassifier

def run_iot_predict(new_signal_window, model_path, input_dim, num_classes):
    """
    接收一个单一的滑动窗口数据 (1, sequence_length, input_dim)，返回预测的设备/心脏状态
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 实例化一个一模一样的网络空壳
    model = IoTLSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=num_classes)

    # 2. 强行灌入保存好的灵魂（权重）
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. 转换格式并推理
    tensor_input = torch.tensor(new_signal_window, dtype=torch.float32).to(device)

    # 如果输入少了一个 batch 维度，强行帮它补上，变为 (1, seq_len, features)
    if len(tensor_input.shape) == 2:
        tensor_input = tensor_input.unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor_input)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)

    return predicted.item(), probabilities[0].tolist()