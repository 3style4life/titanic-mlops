import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from madewithml.iot_data import IoTSymmetricDataset
from madewithml.iot_models import IoTLSTMClassifier

def train_iot_pipeline(X_train, y_train, X_val, y_val, input_dim, num_classes, epochs=10, batch_size=64):
    train_loader = DataLoader(IoTSymmetricDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(IoTSymmetricDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = IoTLSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"📟 运行设备: {device} | 正在启动物联网时序 LSTM 训练流水线...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {val_acc*100:.2f}%")

    # ====== 核心新增：训练结束，直接返回模型本身 ======
    return model