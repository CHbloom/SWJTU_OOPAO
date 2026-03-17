import torch
import torch.nn.functional as F
import sys
sys.path.append('/DATACENTER5/jicheng.liu/SWJTU_OOPAO/SWJTU_OOPAO/tutorials/scao_system/myAoSystem/RL4AO')
import numpy as np
import random
import argparse
import os
from tqdm import tqdm
from envs.AdaptiveOpticsEnvR2DISC import AOEnv
import utils
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def parse_args():
    parser = argparse.ArgumentParser(description="Run AO system reinforcement learning experiment with different algorithms.")
    parser.add_argument("--agent", type=str, default="SACDISC", choices=["PPO", "SAC", "Dreamer-v3"], help="Reinforcement learning algorithm")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--max_step", type=int, default=200, help="AOEnv max_step")
    parser.add_argument("--gainCL", type=float, default=0.6, help="Wavefront sensor gain")
    parser.add_argument("--sampling_rate", type=int, default=1000, help="Telescope sampling frequency (Hz)")
    parser.add_argument("--exposure_time", type=float, default=1, help="Camera exposure time (seconds)")
    parser.add_argument("--clock_rate", type=int, default=1000, help="Camera clock rate (Hz)")
    parser.add_argument("--lightRatio", type=float, default=0.3, help="wfs lightRatio")
    parser.add_argument("--paramFile", type=str, default="paramFile1", help="paramFile")
    return parser.parse_args()

class PolicyNetDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dims):
        super().__init__()
        self.fc = torch.nn.Linear(state_dim, hidden_dim)
        self.heads = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, dim) for dim in action_dims])

    def forward(self, x):
        x = F.relu(self.fc(x))
        logits = [head(x) for head in self.heads]
        probs = [F.softmax(logit, dim=-1) for logit in logits]
        return probs, logits

class QValueNetDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dims):
        super().__init__()
        self.action_dims = action_dims
        self.fc1 = torch.nn.Linear(state_dim + sum(action_dims), hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # a: batch_size x len(action_dims)，每个元素是离散动作索引
        a_onehot = []
        for i, dim in enumerate(self.action_dims):
            a_onehot.append(F.one_hot(a[:, i], num_classes=dim))
        a_onehot = torch.cat(a_onehot, dim=-1).float()
        cat = torch.cat([x, a_onehot], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class SACDiscrete:
    def __init__(self, state_dim, hidden_dim, action_dims, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device):
        self.actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dims).to(device)
        self.critic_1 = QValueNetDiscrete(state_dim, hidden_dim, action_dims).to(device)
        self.critic_2 = QValueNetDiscrete(state_dim, hidden_dim, action_dims).to(device)
        self.target_critic_1 = QValueNetDiscrete(state_dim, hidden_dim, action_dims).to(device)
        self.target_critic_2 = QValueNetDiscrete(state_dim, hidden_dim, action_dims).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True, device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.action_dims = action_dims

    def take_action(self, state, mode="train"):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs, _ = self.actor(state)

        actions = []
        for p in probs:
            if mode == "train":
                # 训练时采样动作
                action = torch.multinomial(p, 1).item()
            elif mode == "test":
                # 测试时选择最高概率动作
                action = torch.argmax(p).item()
            else:
                raise ValueError("Invalid mode. Use 'train' or 'test'.")
            actions.append(action)

        return actions

    def calc_target(self, rewards, next_states, dones, sample_n=16):
        batch_size = next_states.size(0)
        probs, _ = self.actor(next_states)
        device = next_states.device
        action_dim = len(self.action_dims)

        # 对每个状态采样sample_n组动作
        sampled_actions = []
        sampled_log_probs = []
        for _ in range(sample_n):
            actions = []
            log_probs = []
            for p in probs:
                dist = torch.distributions.Categorical(probs=p)
                a = dist.sample()
                actions.append(a)
                log_probs.append(dist.log_prob(a))
            actions = torch.stack(actions, dim=1)  # [batch, action_dim]
            log_probs = torch.stack(log_probs, dim=1).sum(dim=1, keepdim=True)  # [batch, 1]
            sampled_actions.append(actions)
            sampled_log_probs.append(log_probs)
        sampled_actions = torch.stack(sampled_actions, dim=1)  # [batch, sample_n, action_dim]
        sampled_log_probs = torch.stack(sampled_log_probs, dim=1)  # [batch, sample_n, 1]

        # 计算Q值
        q1 = []
        q2 = []
        for i in range(sample_n):
            a = sampled_actions[:, i, :]
            q1.append(self.target_critic_1(next_states, a))
            q2.append(self.target_critic_2(next_states, a))
        q1 = torch.stack(q1, dim=1)  # [batch, sample_n, 1]
        q2 = torch.stack(q2, dim=1)
        q = torch.min(q1, q2)
        log_alpha = self.log_alpha.exp()
        v = (q - log_alpha * sampled_log_probs).mean(dim=1)  # [batch, 1]
        td_target = rewards + self.gamma * v * (1 - dones)
        return td_target

    def update(self, transition_dict, sample_n=16):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = self.calc_target(rewards, next_states, dones, sample_n)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 策略损失采样近似
        probs, _ = self.actor(states)
        sampled_actions = []
        sampled_log_probs = []
        for _ in range(sample_n):
            actions = []
            log_probs = []
            for p in probs:
                dist = torch.distributions.Categorical(probs=p)
                a = dist.sample()
                actions.append(a)
                log_probs.append(dist.log_prob(a))
            actions = torch.stack(actions, dim=1)  # [batch, action_dim]
            log_probs = torch.stack(log_probs, dim=1).sum(dim=1, keepdim=True)  # [batch, 1]
            sampled_actions.append(actions)
            sampled_log_probs.append(log_probs)
        sampled_actions = torch.stack(sampled_actions, dim=1)  # [batch, sample_n, action_dim]
        sampled_log_probs = torch.stack(sampled_log_probs, dim=1)  # [batch, sample_n, 1]

        q1 = []
        q2 = []
        for i in range(sample_n):
            a = sampled_actions[:, i, :]
            q1.append(self.critic_1(states, a))
            q2.append(self.critic_2(states, a))
        q1 = torch.stack(q1, dim=1)  # [batch, sample_n, 1]
        q2 = torch.stack(q2, dim=1)
        q = torch.min(q1, q2)
        log_alpha = self.log_alpha.exp()
        actor_loss = -(log_alpha * sampled_log_probs - q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha自动调整
        entropy = -sampled_log_probs.mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

if __name__ == '__main__':
    args = parse_args()
    env = AOEnv(args)
    save_dir = utils.setup_dirs(args)
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec
    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = args.episodes
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -np.sum(np.log(action_dims))
    device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
    replay_buffer = utils.ReplayBuffer(buffer_size)
    agent = SACDiscrete(state_dim, hidden_dim, action_dims, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)

    return_list = []
    average_sr_list = []
    average_sr_list_old = []
    average_wfe_list = []
    best_mean_return = -np.inf
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                state, _ = env.reset(seed=0)
                episode_return = 0
                episode_sr = []
                episode_sr_old = []
                episode_wfe = []
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    episode_sr.append(env.SR[env.current_step-1])
                    episode_sr_old.append(env.SR_old[env.current_step-1])
                    episode_wfe.append(env.residual[env.current_step-1])
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict, sample_n=16)
                return_list.append(episode_return)
                average_sr_list.append(np.mean(episode_sr))
                average_sr_list_old.append(np.mean(episode_sr_old))
                average_wfe_list.append(np.mean(episode_wfe))
                if i_episode % 10 == 0 and len(return_list) >= 10:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                    'return': f"{episode_return:.1f}",
                                    'avg_SR': f"{np.mean(episode_sr):.3f}",
                                    'avg_WFE': f"{np.mean(episode_wfe):.1f}nm"
                                    })
                pbar.update(1)
    utils.save_training_data(return_list, average_sr_list, average_wfe_list, save_dir, average_sr_list_old)
    utils.test_disc_agent(agent, env, save_dir)
