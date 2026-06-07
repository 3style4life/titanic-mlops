import torch
import pandas as pd
from torch.utils.data import DataLoader
from madewithml.titanic_data import TitanicDataset, preprocess_data
from madewithml.titanic_models import TitanicNeuralNet

def run_predict_pipeline(test_data_path, model_weight_path, scaler, batch_size=32):
    """
    标准的工业级预测流水线
    """
    print("🔮 预测流水线启动...")

    # 1. 触发数据工程模块（is_train=False，此时不切分标签，只拿特征）
    X_test, _ = preprocess_data(test_data_path, is_train=False)

    # 2. 必须使用和训练集【完全相同】的 scaler 进行特征缩放（防止数据偏斜）
    X_test_scaled = scaler.transform(X_test)

    # 3. 封装为 PyTorch DataLoader
    test_loader = DataLoader(TitanicDataset(X_test_scaled), batch_size=batch_size, shuffle=False)

    # 4. 重新组装神经网络，并强行灌入训练好的灵魂（加载权重）
    model = TitanicNeuralNet(input_dim=X_test.shape[1])
    # map_location 确保了即使你在 GPU 上训练的模型，也能在只有 CPU 的机器上跑预测
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    model.eval() # 开启预测模式（会自动关闭 Dropout）

    # 5. 执行批量预测
    all_predictions = []
    with torch.no_grad():
        for X_batch in test_loader:
            outputs = model(X_batch)
            # 神经网络输出的是 0~1 的生存概率，大于 0.5 记为存活(1)，否则记为死亡(0)
            preds = (outputs > 0.5).int().flatten().tolist()
            all_predictions.extend(preds)

    print(f"✅ 预测完成！成功处理了 {len(all_predictions)} 条乘客数据。")
    return all_predictions