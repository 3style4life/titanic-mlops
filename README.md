# 🚢 基于 PyTorch 与 MLOps 规范的 Titanic 生存预测系统

本项目基于 Kaggle 经典的 Titanic（泰坦尼克号）数据集，使用 PyTorch 搭建了多层感知机（MLP）进行生存预测。项目在传统的算法实现基础上进行了**工程化重构**，引入了 **MLflow** 实现实验超参数与指标的动态追踪，并基于 **FastAPI** 实现了生产级的模型在线推理服务部署。

## ✨ 项目亮点与架构

- **标准工业级流水线**：将数据预处理、模型定义、训练流程、离线预测完全解耦，支持端到端的 Pipeline 运行。
- **实验追踪 (MLflow)**：自动记录训练过程中的核心超参数（学习率、Batch Size、训练轮次）与验证集评估指标（Accuracy），告别手动表格记录。
- **服务化部署 (FastAPI)**：将训练好的神经网络模型封装为 RESTful API 推理接口，支持传入 JSON 格式的乘客特征并实时返回生存概率预测。

---

## 📂 项目目录结构

```text
titanic-mlops/
├── madewithml/               # 核心算法与数据工程模块
│   ├── __init__.py           # 使其成为一个可导入的 Python 包
│   ├── titanic_data.py       # 数据预处理（缺失值填充、交叉特征构造、Dataset定义）
│   ├── titanic_models.py     # PyTorch 3层全连接神经网络（MLP）架构
│   ├── titanic_train.py      # 模型训练流水线（集成动态学习率调整机制）
│   ├── titanic_predict.py    # 标准工业级离线预测流水线
│   └── titanic_mlflow.py     # MLflow 实验看板外挂模块
├── datasets/                 # 数据集目录
│   ├── titanic_train.csv     # 训练集数据
│   └── titanic_test.csv      # 测试集数据
├── app_titanic.py            # FastAPI 在线服务化部署脚本
├── run_titanic.py            # 项目总控入口（一键触发训练、评估与实验记录）
├── .gitignore                # Git 忽略配置文件（排除本地权重和日志污染）
└── requirements.txt          # 项目依赖依赖包列表