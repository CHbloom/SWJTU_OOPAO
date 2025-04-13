import sys
sys.path.append('/DATACENTER4/jiangbo.chai/OOPAO_TRWFS-master')

from stable_baselines3 import PPO
from tutorials.scao_system.myAoSystem.RL4AO.AdaptiveOpticsEnv import AOEnv
# 创建环境
env = AOEnv()

# 使用 PPO 算法进行训练
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 测试训练好的模型
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, _, info = env.step(action)
    env.render()