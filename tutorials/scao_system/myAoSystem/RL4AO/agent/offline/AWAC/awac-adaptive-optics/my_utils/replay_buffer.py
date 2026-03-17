import numpy as np
import torch
from gym import spaces

class ReplayBuffer:
    """
    一个支持字典(Dict)观察空间的回放缓冲区。
    它在内部存储原始的 numpy 数组，在采样时将它们批量转换为 PyTorch 张量。
    """
    def __init__(self, obs_space: spaces.Dict, action_dim: int, max_size=int(1e6)):
        """
        初始化回放缓冲区。
        :param obs_space: Gym 环境的字典观察空间。
        :param action_dim: 动作空间的维度 (对于 MultiDiscrete 是组件数量)。
        :param max_size: 缓冲区的最大容量。
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 根据 obs_space 初始化用于存储字典状态的 numpy 数组
        self.obs = {}
        for key, space in obs_space.spaces.items():
            self.obs[key] = np.zeros((max_size, *space.shape), dtype=space.dtype)
        
        self.next_obs = {}
        for key, space in obs_space.spaces.items():
            self.next_obs[key] = np.zeros((max_size, *space.shape), dtype=space.dtype)

        self.action = np.zeros((max_size, action_dim), dtype=np.int64)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, action, next_obs, reward, done):
        """
        向缓冲区添加一条原始经验。
        obs 和 next_obs 是包含 numpy 数组的字典。
        """
        for key in self.obs.keys():
            self.obs[key][self.ptr] = obs[key]
            self.next_obs[key][self.ptr] = next_obs[key]
        
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        从缓冲区中随机采样一个批次的数据，并转换为 PyTorch 张量。
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        # 将 numpy 字典批量转换为 tensor 字典
        obs_batch = {key: torch.tensor(self.obs[key][ind], dtype=torch.float32).to(self.device) for key in self.obs.keys()}
        next_obs_batch = {key: torch.tensor(self.next_obs[key][ind], dtype=torch.float32).to(self.device) for key in self.next_obs.keys()}

        action_batch = torch.tensor(self.action[ind], dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(self.reward[ind], dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(self.done[ind], dtype=torch.float32).to(self.device)

        return obs_batch, action_batch, next_obs_batch, reward_batch, done_batch