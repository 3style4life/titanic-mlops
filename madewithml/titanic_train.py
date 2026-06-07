import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# 注意这里的导入路径改成了新的文件名
from madewithml.titanic_data import TitanicDataset
from madewithml.titanic_models import TitanicNeuralNet

def train_titanic_pipeline(X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    train_loader = DataLoader(TitanicDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TitanicDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = TitanicNeuralNet(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    print("🏋️ 生产级 Titanic 模块化流水线正式启动训练...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
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
                outputs = model(X_batch)
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_acc = correct / total
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}%")

    return model