import mlflow
import torch

def log_titanic_experiment(model_name, hyperparameters, metrics, model_weight_path=None):
    """
    使用 MLflow 记录训练实验与固化模型
    """
    # 1. 开启一个 MLflow 实验记录
    mlflow.set_experiment("Titanic_Deep_Learning")

    with mlflow.start_run():
        print(f"📊 [MLflow] 正在将实验数据同步至本地看板...")

        # 2. 记录超参数 (Hyperparameters)
        mlflow.log_params(hyperparameters)

        # 3. 记录最终指标 (Metrics)
        mlflow.log_metrics(metrics)

        # 4. 追踪模型文件 (Artifacts)
        if model_weight_path:
            mlflow.log_artifact(model_weight_path, artifact_path="saved_models")
            print(f"💾 [MLflow] 成功将模型权重版本绑定至当前实验记录！")