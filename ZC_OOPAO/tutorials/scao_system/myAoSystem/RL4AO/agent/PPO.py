import torch
import torch.nn.functional as F
from env.AdaptiveOpticsEnv import AOEnv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class Policy_Net(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Policy_Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_low = torch.tensor([0.1, 100, 0.1, 100], dtype=torch.float32).to(self.device)
        self.action_high = torch.tensor([1.0, 2000, 1.0, 2000], dtype=torch.float32).to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 匹配AO动作空间[low, high]
        mu = 0.5 * (self.action_high + self.action_low) + 0.5 * (self.action_high - self.action_low) * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))  # 用于确保输出为正数
        return mu, std


class Value_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Value_Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lamda, epochs, epsilon, gamma, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lamda = lamda
        self.eps = epsilon
        self.epoch = epochs
        self.device = device
        self.Actor = Policy_Net(state_dim, action_dim, hidden_dim).to(self.device)
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

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        mu, std = self.Actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample().view(4)
        return action.tolist()

    def update(self, transition_dict):
        states, actions, rewards, next_states, dones = (
            torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device),
            torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device),
            torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device),
            torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device),
            torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device))
        td_target = rewards + self.Critic(next_states) * self.gamma * (1 - dones)
        td_delta = td_target - self.Critic(states)
        advantage = self.compute_advantages(td_delta).to(self.device)
        mu, std = self.Actor(states)
        dist = torch.distributions.Normal(mu.detach(), std.detach())  # 旧的策略分布不需要计算梯度
        old_log_probs = dist.log_prob(actions)
        for _ in range(self.epoch):
            mu, std = self.Actor(states)
            dist = torch.distributions.Normal(mu, std)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            actor_loss = torch.mean(-torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage))
            critic_loss = torch.mean(F.mse_loss(td_target.detach(), self.Critic(states)))
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()



if __name__=="__main__":
    env_name = 'AOEnv'
    env = AOEnv()
    actor_lr = 1e-4
    critic_lr = 5e-3
    gamma = 0.9
    lamda = 0.9
    epsilon = 0.2
    epochs = 10
    episodes = 1000
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lamda, epochs, epsilon, gamma, device)
    return_list = []
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
                while not done:
                    action = agent.take_action(state)
                    next_state,  reward,  d1, d2, _ = env.step(action)
                    done = d1 or d2
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["rewards"].append(reward)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["dones"].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if episode % 10 == 0 and len(return_list) >= 10:
                    pbar.set_postfix({'episode': '%d' % (episodes / 10 * i + episode + 1),
                                    'return': np.mean(return_list[-10:])
                                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO-Clip on {}'.format(env_name))
    plt.show()