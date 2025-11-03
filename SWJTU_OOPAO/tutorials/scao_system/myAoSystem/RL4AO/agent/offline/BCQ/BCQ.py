import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class ReplayBuffer:
    """一个标准的经验回放缓冲区，用于存储离线数据集"""
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

class QNetwork(nn.Module):
    """标准的Q值网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        return self.l3(q)

class Generator(nn.Module):
    """
    生成器网络，用于行为克隆 (Behavior Cloning)。
    它的目标是模仿数据集中的行为策略。
    给定一个状态，它会输出每个离散动作的对数概率（logits）。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        g = F.relu(self.l1(state))
        g = F.relu(self.l2(g))
        return self.l3(g) # 输出logits

class BCQ:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        discount=0.99,
        tau=0.005,
        lr=3e-4,
        threshold=0.3 # BCQ的核心约束阈值
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q 网络
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_target_network = copy.deepcopy(self.q_network)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # 生成器网络 (用于行为克隆)
        self.generator = Generator(state_dim, action_dim, hidden_dim).to(self.device)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.threshold = threshold

    def take_action(self, state):
        """
        根据BCQ策略选择动作：
        1. 用生成器预测每个动作的概率。
        2. 筛选出概率高于阈值的动作。
        3. 在筛选出的动作中，选择Q值最大的一个。
        """
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            
            # 1. 用生成器得到动作的logits，并转换为概率
            action_logits = self.generator(state)
            action_probs = F.softmax(action_logits, dim=1)
            
            # 2. 筛选出概率高于阈值的动作
            # action_probs >= self.threshold 会得到一个布尔张量
            # (action_probs >= self.threshold).float() - 1e-8 会把False变成一个很大的负数
            # 这样在计算Q值时，不满足条件的动作的Q值就会变得非常小
            q_values = self.q_network(state)
            
            # 将不满足条件的动作的Q值设为一个非常小的值，使其不会被选中
            q_values[action_probs < self.threshold] = -1e8
            
            # 3. 在筛选后的动作中选择Q值最大的
            action = torch.argmax(q_values, dim=1).cpu().data.numpy().flatten()
            
        return action[0]

    def update(self, replay_buffer, batch_size=256):
        # 从回放区采样
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        # --- 1. 训练生成器 (行为克隆) ---
        # 目标是让生成器在给定状态下，能预测出数据集中对应的动作
        action_logits = self.generator(state)
        # 使用交叉熵损失，就像一个分类问题
        generator_loss = F.cross_entropy(action_logits, action.squeeze())

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.step()

        # --- 2. 训练Q网络 ---
        with torch.no_grad():
            # 计算目标Q值
            # 首先，用BCQ策略在next_state上选择动作
            next_action_logits = self.generator(next_state)
            next_action_probs = F.softmax(next_action_logits, dim=1)
            
            # 同样，筛选出概率高于阈值的动作
            next_q_values = self.q_target_network(next_state)
            next_q_values[next_action_probs < self.threshold] = -1e8
            
            # 从筛选后的动作中选择Q值最大的，得到 a'
            next_action = torch.argmax(next_q_values, dim=1).unsqueeze(1)
            
            # 用目标Q网络计算Q(s', a')
            target_q = self.q_target_network(next_state).gather(1, next_action)
            
            # 计算最终的目标 y = r + gamma * Q_target(s', a')
            target_q = reward + (1 - done) * self.discount * target_q

        # 计算当前Q值 Q(s, a)
        current_q = self.q_network(state).gather(1, action)

        # 计算Q损失 (MSE)
        q_loss = F.mse_loss(current_q, target_q)

        # 更新Q网络
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # --- 3. 软更新目标网络 ---
        for param, target_param in zip(self.q_network.parameters(), self.q_target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {"q_loss": q_loss.item(), "g_loss": generator_loss.item()}

if __name__ == '__main__':
    # === 离线数据加载 ===
    from BCQ_load_dreamer import offline_buffer, STATE_DIM, ACTION_DIM

    agent = BCQ(STATE_DIM, ACTION_DIM)
    print("\n开始离线训练BCQ智能体...")
    training_steps = 50000
    eval_freq = 5000

    for t in range(training_steps):
        losses = agent.update(offline_buffer, batch_size=128)
        if (t + 1) % eval_freq == 0:
            print(f"Step: {t+1}/{training_steps} | Q Loss: {losses['q_loss']:.3f} | G Loss: {losses['g_loss']:.3f}")

            # 评估：你可以用AOEnv环境做评估
            # 这里只给出伪代码，实际需根据你的AOEnv接口实现
            # from envs.AdaptiveOpticsEnvR1DISC import AOEnv
            # env = AOEnv(args)
            # avg_reward = 0
            # for _ in range(10):
            #     state = env.reset()
            #     done = False
            #     episode_reward = 0
            #     while not done:
            #         state_vec = np.concatenate([
            #             state['image'].flatten(),
            #             state['dmCoefs'].flatten(),
            #             state['wfsSingnal'].flatten(),
            #             state['turb_features'].flatten()
            #         ])
            #         action = agent.take_action(state_vec)
            #         state, reward, done, _ = env.step(action)
            #         episode_reward += reward
            #     avg_reward += episode_reward
            # avg_reward /= 10
            # print(f"评估平均奖励: {avg_reward:.2f}")