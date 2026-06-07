import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from madewithml.titanic_models import TitanicNeuralNet

# 1. 初始化 FastAPI 实例
app = FastAPI(title="Titanic 生死预测服务 API", version="1.0.0")

# 2. 定义接收请求的数据格式 (前端传过来的乘客信息)
class PassengerInput(BaseModel):
    Pclass: int
    Age: float
    Fare: float
    FamilySize: int
    IsAlone: int
    Sex_male: int
    Embarked_Q: int
    Embarked_S: int

# 3. 在 API 启动时，提前把模型和灵魂（权重）加载到内存中
MODEL_PATH = "titanic_best_model.pth"
input_dim = 8 # 我们的特征总数
model = TitanicNeuralNet(input_dim=input_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

@app.get("/")
def home():
    return {"message": "🚢 欢迎来到泰坦尼克号生死在线预测系统 API！请请求 /predict 接口。"}

@app.post("/predict")
def predict_survival(passenger: PassengerInput):
    """
    实时接收一位乘客的数据，返回生死预测
    """
    # 将输入的 JSON 数据转为 PyTorch 能够识别的 Tensor 矩阵
    features = [
        passenger.Pclass, passenger.Age, passenger.Fare,
        passenger.FamilySize, passenger.IsAlone, passenger.Sex_male,
        passenger.Embarked_Q, passenger.Embarked_S
    ]

    # 转换维度符合 batch 要求 -> (1, 8)
    features_tensor = torch.tensor([features], dtype=torch.float32)

    # 模型推理
    with torch.no_grad():
        probability = model(features_tensor).item()
        prediction = 1 if probability > 0.5 else 0

    status = "存活 (Survived)" if prediction == 1 else "遇难 (Deceased)"

    return {
        "prediction": prediction,
        "probability": f"{probability * 100:.2f}%",
        "status": status
    }