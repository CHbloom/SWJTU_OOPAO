import sys
sys.path.append('/home/jiangbo.chai/DATACENTER4/SWJTU_OOPAO/tutorials/scao_system/myAoSystem/RL4AO')
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
def parse_args():
    parser = argparse.ArgumentParser(description="Run AO system reinforcement learning experiment with different algorithms.")
    parser.add_argument("--agent", type=str, default="ppo_mid_fusion_R2", choices=["PPO", "SAC", "Dreamer-v3"], help="Reinforcement learning algorithm")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=0, help="seed") # AOEnv随机种子未实现------------
    parser.add_argument("--max_step", type=int, default=200, help="AOEnv max_step")
    parser.add_argument("--gainCL", type=float, default=0.6, help="Wavefront sensor gain")
    parser.add_argument("--sampling_rate", type=int, default=1000, help="Telescope sampling frequency (Hz)")
    parser.add_argument("--exposure_time", type=float, default=1, help="Camera exposure time (seconds)")
    parser.add_argument("--clock_rate", type=int, default=1000, help="Camera clock rate (Hz)")
    parser.add_argument("--lightRatio", type=float, default=0.3, help="wfs lightRatio")
    parser.add_argument("--paramFile", type=str, default="paramFile1", help="paramFile")
    return parser.parse_args()

class FeatureExtractor(nn.Module):
    """
    用于从观察空间中动态提取和融合特征的网络。
    可以处理任意数量的图像输入和向量输入。
    """
    def __init__(self, obs_space):
        super().__init__()
        
        self.cnns = nn.ModuleDict()
        self.cnn_fcs = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        
        total_features_dim = 0
        
        # 动态创建图像和向量处理模块
        for key, space in obs_space.spaces.items():
            if "img" in key:
                # --- 核心修改 1: 正确处理 (time_window, H, W) 形状 ---
                img_shape = space.shape
                time_window, H, W = img_shape
                
                # 将 time_window 视为输入通道
                cnn = nn.Sequential(
                    nn.Conv2d(time_window, 16, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                self.cnns[key] = cnn
                
                # 动态计算CNN输出维度并创建FC层
                with torch.no_grad():
                    # 创建正确的4D dummy_input: (N, C, H, W)
                    dummy_input = torch.zeros(1, time_window, H, W)
                    cnn_output_dim = cnn(dummy_input).shape[1]
                
                cnn_feature_dim = 128
                self.cnn_fcs[key] = nn.Linear(cnn_output_dim, cnn_feature_dim)
                total_features_dim += cnn_feature_dim

            elif "vec" in key:
                # --- 核心修改 2: 正确处理 (time_window, vec_dim) 形状 ---
                vec_shape = space.shape
                time_window, vec_dim = vec_shape
                mlp_feature_dim = 64
                # MLP的输入维度是 vec_dim, 我们将在 forward 中对时间维做平均
                mlp = nn.Sequential(
                    nn.Linear(vec_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, mlp_feature_dim),
                    nn.ReLU()
                )
                self.mlps[key] = mlp
                total_features_dim += mlp_feature_dim

        self.features_dim = total_features_dim

    def forward(self, obs):
        features = []
        # --- 核心修改 3: 重构 forward 逻辑 ---
        # 处理所有图像输入
        for key, cnn in self.cnns.items():
            # obs[key] 的形状是 (batch_size, time_window, H, W), 这正是CNN期望的
            img_obs = obs[key]
            img_feat = F.relu(self.cnn_fcs[key](cnn(img_obs)))
            features.append(img_feat)
            
        # 处理所有向量输入
        for key, mlp in self.mlps.items():
            # obs[key] 的形状是 (batch_size, time_window, vec_dim)
            # 对时间维度取平均，得到 (batch_size, vec_dim)
            vec_obs = obs[key].mean(dim=1) 
            vec_feat = mlp(vec_obs)
            features.append(vec_feat)
            
        return torch.cat(features, dim=1)

class Policy_Net(torch.nn.Module):
    def __init__(self, features_dim, action_dims, hidden_dim):
        super().__init__()
        self.fc = torch.nn.Linear(features_dim, hidden_dim)
        self.heads = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, dim) for dim in action_dims])

    def forward(self, x):
        x = F.relu(self.fc(x))
        logits = [head(x) for head in self.heads]
        return logits

class Value_Net(torch.nn.Module):
    def __init__(self, features_dim, hidden_dim):
        super(Value_Net, self).__init__()
        self.fc1 = torch.nn.Linear(features_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x) # 价值输出通常不加ReLU

class PPO:
    def __init__(self, observation_space, hidden_dim, action_dims, actor_lr, critic_lr, lamda, epochs, epsilon, gamma, device):
        self.action_dims = action_dims
        self.gamma = gamma
        self.lamda = lamda
        self.eps = epsilon
        self.epoch = epochs
        self.device = device

        # 1. 首先实例化特征提取器
        self.feature_extractor = FeatureExtractor(observation_space).to(self.device)
        # 2. 从实例中动态获取特征维度
        self.features_dim = self.feature_extractor.features_dim
        
        # 3. 使用动态维度初始化策略和价值网络
        self.Actor = Policy_Net(self.features_dim, action_dims, hidden_dim).to(self.device)
        self.Critic = Value_Net(self.features_dim, hidden_dim).to(self.device)
        
        # 演员优化器训练特征提取器和策略头
        actor_params = list(self.feature_extractor.parameters()) + list(self.Actor.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)
        # 评论家优化器只训练价值头
        self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=critic_lr)

    def compute_advantages(self, td_delta):
        td_delta = td_delta.cpu().detach().numpy()
        advantages = []
        a = 0
        for t in reversed(td_delta):
            a = self.lamda * self.gamma * a + t
            advantages.append(a)
        advantages.reverse()
        return torch.tensor(np.array(advantages), dtype=torch.float).to(self.device)

    def take_action(self, state, mode="train"):
        # 动态地将状态字典中的每个numpy数组转换为tensor
        state_tensor = {key: torch.tensor(np.array([value]), dtype=torch.float).to(self.device) 
                        for key, value in state.items()}
        
        with torch.no_grad():
            features = self.feature_extractor(state_tensor)
            action_logits = self.Actor(features)
        
        actions = []
        for i, logits in enumerate(action_logits):
            dist = torch.distributions.Categorical(logits=logits)
            if mode == "train":
                action = dist.sample()
            elif mode == "test":
                action = torch.argmax(logits)
            else:
                raise ValueError("Invalid mode. Use 'train' or 'test'.")
            actions.append(action.item())
        
        return actions

    def update(self, transition_dict):
        # 将字典列表转换为字典的张量 (这部分已是动态的，无需修改)
        states = {key: torch.tensor(np.array([d[key] for d in transition_dict['states']]), dtype=torch.float).to(self.device) for key in transition_dict['states'][0]}
        next_states = {key: torch.tensor(np.array([d[key] for d in transition_dict['next_states']]), dtype=torch.float).to(self.device) for key in transition_dict['next_states'][0]}
        
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            next_features = self.feature_extractor(next_states)
            td_target = rewards + self.Critic(next_features) * self.gamma * (1 - dones)
        
        features = self.feature_extractor(states)
        td_delta = td_target - self.Critic(features)
        advantage = self.compute_advantages(td_delta.detach()).to(self.device)

        action_logits = self.Actor(features.detach())
        old_log_probs = []
        for i, (logits, action) in enumerate(zip(action_logits, actions.T)):
            dist = torch.distributions.Categorical(logits=logits)
            old_log_probs.append(dist.log_prob(action))
        old_log_probs = torch.stack(old_log_probs, dim=1).sum(dim=1)

        for _ in range(self.epoch):
            features_new = self.feature_extractor(states)
            action_logits_new = self.Actor(features_new)
            
            new_log_probs = []
            for i, (logits, action) in enumerate(zip(action_logits_new, actions.T)):
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs.append(dist.log_prob(action))
            new_log_probs = torch.stack(new_log_probs, dim=1).sum(dim=1)

            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            actor_loss = torch.mean(-torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage))
            
            # 评论家损失使用与演员相同的特征，但分离计算图
            critic_loss = torch.mean(F.mse_loss(td_target.detach(), self.Critic(features_new.detach())))

            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            
            critic_loss.backward()
            actor_loss.backward()
            
            actor_params = list(self.feature_extractor.parameters()) + list(self.Actor.parameters())
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=0.5)
            
            self.critic_optimizer.step()
            self.actor_optimizer.step()

if __name__=="__main__":
    args = parse_args()
    # 根据参数动态导入环境
    if args.agent == "ppo_mid_fusion":
        from envs.AOEnv.AOEnvDISC_mid_fusion import AOEnv
    elif args.agent == "ppo_mid_fusion_R1":
        from envs.AOEnvR1.AOEnvR1DISC_mid_fusion import AOEnv
    elif args.agent == "ppo_mid_fusion_R2":
        from envs.AOEnvR2.AOEnvR2DISC_mid_fusion import AOEnv
        
    env = AOEnv(args)
    save_dir = utils.setup_dirs(args)

    # --- 关键改动 ---
    # 在访问 observation_space 之前，必须先调用 reset() 来初始化它
    env.reset(seed=args.seed)
    # ----------------

    actor_lr = 1e-4
    critic_lr = 5e-3
    gamma = 0.9
    lamda = 0.9
    epsilon = 0.2
    epochs = 10
    episodes = args.episodes
    
    # 现在可以安全地访问 observation_space
    observation_space = env.observation_space
    action_dims = env.action_space.nvec
    hidden_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PPO(observation_space, hidden_dim, action_dims, actor_lr, critic_lr, lamda, epochs, epsilon, gamma, device)
    return_list = []
    average_sr_list = []
    average_sr_list_old = []
    average_wfe_list = []

    best_mean_return = -np.inf
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(10):
        with tqdm(total=int(episodes/10), desc="Iteration %d" % i) as pbar:
            for episode in range(int(episodes/10)):
                # 训练循环开始时，再次调用 reset 获取初始状态
                state, _ = env.reset(seed=args.seed + episode) # 使用不同的种子以增加多样性
                done = False
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "rewards": [],
                    "next_states": [],
                    "dones": []
                }
                episode_return = 0
                episode_sr = []
                episode_sr_old = []
                episode_wfe = []
                while not done:
                    action = agent.take_action(state)
                    next_state, reward,  d1, d2, _ = env.step(action)
                    done = d1 or d2
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["rewards"].append(reward)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["dones"].append(done)
                    state = next_state
                    episode_return += reward
                    episode_sr.append(env.SR[env.current_step-1])
                    episode_sr_old.append(env.SR_old[env.current_step-1])
                    episode_wfe.append(env.residual[env.current_step-1])
                return_list.append(episode_return)
                average_sr_list.append(np.mean(episode_sr))
                average_sr_list_old.append(np.mean(episode_sr_old))
                average_wfe_list.append(np.mean(episode_wfe))
                
                agent.update(transition_dict)

                # 保存最佳模型
                current_mean_return = np.mean(return_list[-10:]) if len(return_list) > 0 else -np.inf
                if current_mean_return > best_mean_return:
                    best_mean_return = current_mean_return
                    torch.save({
                        'actor_state_dict': agent.Actor.state_dict(),
                        'critic_state_dict': agent.Critic.state_dict(),
                        'feature_extractor_state_dict': agent.feature_extractor.state_dict(),
                        'best_return': best_mean_return
                    }, f"./data/{save_dir}/train/models/best_model_{timestamp}.pth")

                if episode % 10 == 0 and len(return_list) >= 10:
                    pbar.set_postfix({'episode': '%d' % (episodes / 10 * i + episode + 1),
                                    'return': f"{episode_return:.1f}",
                                    'avg_SR': f"{np.mean(episode_sr):.3f}",
                                    'avg_WFE': f"{np.mean(episode_wfe):.1f}nm"
                                    })
                pbar.update(1)

    utils.save_training_data(return_list, average_sr_list, average_wfe_list, save_dir, average_sr_list_old)
    utils.test_disc_agent(agent, env, save_dir)