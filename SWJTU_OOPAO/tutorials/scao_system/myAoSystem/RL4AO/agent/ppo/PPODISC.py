import sys
sys.path.append('/home/jiangbo.chai/DATACENTER4/SWJTU_OOPAO/tutorials/scao_system/myAoSystem/RL4AO')
import torch
import torch.nn.functional as F
from envs.AOEnvR1.AdaptiveOpticsEnvR1DISC import AOEnv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
def parse_args():
    parser = argparse.ArgumentParser(description="Run AO system reinforcement learning experiment with different algorithms.")
    parser.add_argument("--agent", type=str, default="PPODISC4R1", choices=["PPO", "SAC", "Dreamer-v3"], help="Reinforcement learning algorithm")
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

class Policy_Net(torch.nn.Module):
    def __init__(self, state_dim, action_dims, hidden_dim):
        super().__init__()
        self.fc = torch.nn.Linear(state_dim, hidden_dim)
        # 为每个维度创建一个独立的输出头部
        self.heads = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, dim) for dim in action_dims])

    def forward(self, x):
        x = F.relu(self.fc(x))
        # 对每个动作维度产生 logits
        logits = [head(x) for head in self.heads]
        return logits  # 返回每个维度的 logits 列表


class Value_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Value_Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dims, actor_lr, critic_lr, lamda, epochs, epsilon, gamma, device):
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.action_dim = len(action_dims)
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lamda = lamda
        self.eps = epsilon
        self.epoch = epochs
        self.device = device
        self.Actor = Policy_Net(state_dim, action_dims, hidden_dim).to(self.device)
        self.Critic = Value_Net(state_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=critic_lr)

    def compute_advantages(self, td_delta):
        td_delta = td_delta.cpu().detach().numpy()
        advantages = []
        a = 0
        for t in reversed(td_delta):
            a = self.lamda * self.gamma * a + t
            advantages.append(a)
        advantages.reverse()
        return torch.tensor(np.array(advantages), dtype=float).to(self.device)

    def take_action(self, state, mode="train"):
        # tmp_state = torch.Tensor(state)
        # print(torch.isnan(tmp_state).any(), torch.isinf(tmp_state).any(), tmp_state.max(), tmp_state.min())
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action_logits = self.Actor(state)
        
        # 为每个动作维度分别采样
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
        states, actions, rewards, next_states, dones = (
            torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device),
            torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device),
            torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device),
            torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device),
            torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device))
        td_target = rewards + self.Critic(next_states) * self.gamma * (1 - dones)
        td_delta = td_target - self.Critic(states)
        advantage = self.compute_advantages(td_delta.detach()).to(self.device)

        # print("reward:", rewards)
        # print("reward mean / std:", rewards.mean().item(), rewards.std().item())
        # print("reward max / min:", rewards.max().item(), rewards.min().item())


        action_logits = self.Actor(states)

        # 为每个动作维度计算 log_prob
        old_log_probs = []
        for i, (logits, action) in enumerate(zip(action_logits, actions.T)):
            dist = torch.distributions.Categorical(logits=logits.detach())
            old_log_probs.append(dist.log_prob(action))
        
        old_log_probs = torch.stack(old_log_probs, dim=1).sum(dim=1)

        for _ in range(self.epoch):
            action_logits = self.Actor(states)
            # 计算新的 log_prob
            new_log_probs = []
            for i, (logits, action) in enumerate(zip(action_logits, actions.T)):
                # print(logits)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs.append(dist.log_prob(action))
            new_log_probs = torch.stack(new_log_probs, dim=1).sum(dim=1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            actor_loss = torch.mean(-torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage))
            critic_loss = torch.mean(F.mse_loss(td_target.detach(), self.Critic(states)))

            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()
            self.actor_optimizer.step()


if __name__=="__main__":
    args = parse_args()
    # 创建环境
    env_name = 'AOEnv'
    env = AOEnv(args)
    save_dir = utils.setup_dirs(args)

    actor_lr = 1e-4
    critic_lr = 5e-3
    gamma = 0.9
    lamda = 0.9
    epsilon = 0.2
    epochs = 10
    episodes = args.episodes
    # 获取Space_Box维度
    env.reset(seed=0)
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec
    hidden_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    agent = PPO(state_dim, hidden_dim, action_dims, actor_lr, critic_lr, lamda, epochs, epsilon, gamma, device)
    return_list = []
    average_sr_list = []  # 一回合平均斯特列尔比
    average_sr_list_old = []
    average_wfe_list = []  # 波前误差

    best_mean_return = -np.inf
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(10):
        with tqdm(total=int(episodes/10), desc="Iteration %d" % i) as pbar:
            for episode in range(int(episodes/10)):
                state, _ = env.reset(seed=0)
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
                current_mean_return = np.mean(return_list[-10:])
                if current_mean_return > best_mean_return:
                    best_mean_return = current_mean_return
                    torch.save({
                        'actor_state_dict': agent.Actor.state_dict(),
                        'critic_state_dict': agent.Critic.state_dict(),
                        'optimizer_state_dict': agent.actor_optimizer.state_dict(),
                        'best_return': best_mean_return
                    }, f"./data/{save_dir}/train/models/best_model_{timestamp}.pth")

                if episode % 10 == 0 and len(return_list) >= 10:
                    pbar.set_postfix({'episode': '%d' % (episodes / 10 * i + episode + 1),
                                    'return': f"{episode_return:.1f}",
                                    'avg_SR': f"{np.mean(episode_sr):.3f}",
                                    'avg_WFE': f"{np.mean(episode_wfe):.1f}nm"
                                    })
                pbar.update(1)

    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('PPO-Clip on {}'.format(env_name))
    # plt.show()

    utils.save_training_data(return_list, average_sr_list, average_wfe_list, save_dir, average_sr_list_old)

    utils.test_disc_agent(agent, env, save_dir)