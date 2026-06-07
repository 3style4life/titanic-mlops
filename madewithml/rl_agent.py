import os
import gymnasium as gym
from stable_baselines3 import PPO

def train_rl_agent(env_id="CartPole-v1", total_timesteps=30000):
    """
    实例化游戏环境，并使用 PPO 算法训练智能体
    """
    print(f"🎮 正在初始化虚拟博弈环境: {env_id} ...")
    # 创建标准的训练环境
    env = gym.make(env_id)

    print("🧠 正在基于 PyTorch 构建 PPO 神经网络智能体...")
    # MlpPolicy 代表使用普通的多层感知机作为大脑网络来观察状态
    # verbose=1 会在控制台打印出非常详细的训练指标
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, seed=42)

    print(f"🚀 智能体正在进入游戏环境开始“左右互搏”，预计尝试 {total_timesteps} 个动作步...")
    model.learn(total_timesteps=total_timesteps)
    print("🏋️ 训练结束！智能体已经摸索出了游戏套路。")

    # 固化保存大脑权重
    model_path = "ppo_cartpole_model"
    model.save(model_path)
    print(f"💾 [MLOps] 强化学习智能体权重已保存至: {model_path}.zip")

    env.close()
    return model_path