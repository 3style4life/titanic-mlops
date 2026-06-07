import gymnasium as gym
from stable_baselines3 import PPO

def evaluate_rl_agent(model_path, env_id="CartPole-v1", episodes=3):
    """
    加载保存好的强化学习模型，在测试环境里看它玩游戏的效果
    """
    print(f"\n🎬 正在唤醒保存好的游戏大师... 加载权重: {model_path}")

    # render_mode="human" 可以弹出真正的游戏画面窗口（由于服务器或系统差异，我们这里先用标准模式）
    # 如果你在本地运行，想要亲眼看弹窗动画，可以把下面这行改成 gym.make(env_id, render_mode="human")
    env = gym.make(env_id)

    model = PPO.load(model_path)

    for episode in range(episodes):
        # 初始化游戏，获取初始画面状态
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            # 让大脑根据当前看到的画面状态 obs，预测出最稳健的动作 action
            # deterministic=True 代表纯理智流，拒绝任何随机乱玩
            action, _states = model.predict(obs, deterministic=True)

            # 智能体在环境中按下这个按键
            obs, reward, done, truncated, info = env.step(action)

            # 累加这一局游戏的得分
            episode_reward += reward

        print(f"🏆 第 {episode + 1} 局游戏结束！AI 坚持了 {episode_reward} 个时间步！")

    env.close()