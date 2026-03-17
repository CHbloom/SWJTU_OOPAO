import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import argparse
from AdaptiveOpticsEnvR1DISC import AOEnv

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.int64)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
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
    def __init__(self, state_dim, action_dims, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.action_dims = action_dims
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, sum(action_dims))
    def forward(self, state):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        # 拆分为4段，每段分别softmax
        splits = torch.split(q, self.action_dims, dim=1)
        return splits  # 返回4个(batch,dim)的tuple

class Generator(nn.Module):
    def __init__(self, state_dim, action_dims, hidden_dim=256):
        super().__init__()
        self.action_dims = action_dims
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, sum(action_dims))
    def forward(self, state):
        g = F.relu(self.l1(state))
        g = F.relu(self.l2(g))
        g = self.l3(g)
        splits = torch.split(g, self.action_dims, dim=1)
        return splits

class BCQ:
    def __init__(self, state_dim, action_dims, hidden_dim=256, discount=0.99, tau=0.005, lr=3e-4, threshold=0.3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dims = action_dims
        self.num_actions = len(action_dims)
        self.discount = discount
        self.tau = tau
        self.threshold = threshold

        self.q_net = QNetwork(state_dim, action_dims, hidden_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q_net)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.generator = Generator(state_dim, action_dims, hidden_dim).to(self.device)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)

    def take_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, state_dim]
        with torch.no_grad():
            g_logits_splits = self.generator(state)
            q_vals_splits = self.q_net(state)
            
            action = []
            for i in range(self.num_actions):
                # 1. 获取生成器概率和Q值
                g_probs = F.softmax(g_logits_splits[i], dim=1)
                q_vals = q_vals_splits[i]
                
                # 2. 应用BCQ约束：将低概率动作的Q值设为负无穷
                q_vals[g_probs < self.threshold] = -1e8
                
                # 3. 在约束后的动作中选择Q值最大的
                act = torch.argmax(q_vals, dim=1).item()
                action.append(act)
        return np.array(action, dtype=np.int32)

    def update(self, buffer, batch_size=256):
        state, action, next_state, reward, done = buffer.sample(batch_size)
        batch_size = state.shape[0]

        # Generator loss
        g_logits = self.generator(state)
        g_loss = 0
        for i in range(len(self.action_dims)):
            g_loss += F.cross_entropy(g_logits[i], action[:, i])
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        with torch.no_grad():
            next_g_logits = self.generator(next_state)
            next_probs = [F.softmax(logit, dim=1) for logit in next_g_logits]
            next_q_vals = self.q_target(next_state)

            next_actions = []
            for i in range(len(self.action_dims)):
                q = next_q_vals[i].clone()
                q[next_probs[i] < self.threshold] = -1e8
                act = torch.argmax(q, dim=1)
                next_actions.append(act.unsqueeze(1))
            next_actions = torch.cat(next_actions, dim=1)

            # 目标 Q 值
            target_q = 0
            for i in range(len(self.action_dims)):
                target_q += next_q_vals[i].gather(1, next_actions[:, i].unsqueeze(1)).squeeze(1)
            target_q = reward.squeeze(1) + (1 - done.squeeze(1)) * self.discount * target_q

        # 当前 Q 值
        current_q = 0
        q_vals = self.q_net(state)
        for i in range(len(self.action_dims)):
            current_q += q_vals[i].gather(1, action[:, i].unsqueeze(1)).squeeze(1)

        # Q 损失
        q_loss = F.mse_loss(current_q, target_q)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # 软更新 target 网络
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q_loss": q_loss.item(),
            "g_loss": g_loss.item(),
            "q_mean": current_q.mean().item(),
            "q_max": current_q.max().item(),
        }
    

def decode_multidiscrete_onehot(onehot, dims):
    """把one-hot向量还原为MultiDiscrete的整数动作"""
    indices = []
    pointer = 0
    for dim in dims:
        sub = onehot[pointer:pointer+dim]
        indices.append(np.argmax(sub))
        pointer += dim
    return np.array(indices, dtype=np.int32)

def test_agent(agent, env, save_dir, episodes=20): # 1. 增加默认测试次数
    test_save_dir = os.path.join(save_dir, "test")
    os.makedirs(test_save_dir, exist_ok=True)
    
    all_returns = [] # 用于存储每次测试的回报

    print("\n" + "="*50)
    print(f"开始进行 {episodes} 轮性能测试...")
    print("="*50)

    for ep in range(episodes):
        # 2. 为每次测试提供不同种子，确保湍流不同
        state_dict = env.reset(seed=10000 + ep) 
        episode_return = 0
        done = False
        
        gainCL_list, sampling_rate_list, exposure_time_list, clock_rate_list = [], [], [], []

        while not done:
            # 将字典状态展平为向量
            state_vec = np.concatenate([
                state_dict['image'].flatten(),
                state_dict['dmCoefs'].flatten(),
                state_dict['wfsSingnal'].flatten(),
                state_dict['turb_features'].flatten()
            ])
            
            action = agent.take_action(state_vec)
            
            # 记录真实的动作值
            real_action = env.get_action_from_discrete(action)
            gainCL_list.append(real_action[0])
            sampling_rate_list.append(real_action[1])
            exposure_time_list.append(real_action[2])
            clock_rate_list.append(real_action[3])
            
            next_state_dict, reward, done, _, = env.step(action)
            state_dict = next_state_dict
            episode_return += reward
            
        all_returns.append(episode_return)
        print(f"  Test Episode {ep+1}/{episodes}, Return: {episode_return:.2f}")

        # --- 绘图部分保持不变，为每次测试生成图像 ---
        plt.figure()
        plt.plot(np.arange(env.max_step), env.SR, label='Strehl Ratio')
        plt.xlabel('Loop Index')
        plt.title(f'Strehl Ratio (Episode {ep+1}, Return={episode_return:.2f})')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(test_save_dir, f"SR_episode_{ep+1}.png"))
        plt.close()

        plt.figure(figsize=(12, 10))
        plt.subplot(221)
        plt.plot(gainCL_list)
        plt.title("gainCL")
        plt.grid()
        plt.subplot(222)
        plt.plot(sampling_rate_list)
        plt.title("sampling_rate")
        plt.grid()
        plt.subplot(223)
        plt.plot(exposure_time_list)
        plt.title("exposure_time")
        plt.grid()
        plt.subplot(224)
        plt.plot(clock_rate_list)
        plt.title("clock_rate")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(test_save_dir, f"actions_episode_{ep+1}.png"))
        plt.close()

    # 3. 在所有测试结束后，计算并打印统计结果
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    
    print("\n" + "="*50)
    print("      测试结果统计      ")
    print("="*50)
    print(f"总测试回合数: {episodes}")
    print(f"平均回报: {mean_return:.2f}")
    print(f"回报标准差 (波动): {std_return:.2f}")
    print(f"回报范围 (平均值±标准差): [{mean_return - std_return:.2f}, {mean_return + std_return:.2f}]")
    print(f"最高回报: {np.max(all_returns):.2f}")
    print(f"最低回报: {np.min(all_returns):.2f}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paramFile', type=str, default='paramFile1', help='Parameter file name')
    parser.add_argument('--max_step', type=int, default=200)
    parser.add_argument('--gainCL', type=float, default=0.6)
    parser.add_argument('--sampling_rate', type=int, default=1000)
    parser.add_argument('--exposure_time', type=float, default=1.0)
    parser.add_argument('--clock_rate', type=int, default=1000)
    parser.add_argument('--lightRatio', type=float, default=0.3)
    args = parser.parse_args()

    #数据加载
    DREAMER_TRAINDIR = '/home/jiangbo.chai/DATACENTER4/SWJTU_OOPAO/tutorials/scao_system/myAoSystem/RL4AO/agent/dreamerv3/logdir/AO1_R1DISC_lightRatio0.30/train_eps'
    sample_file = next(pathlib.Path(DREAMER_TRAINDIR).glob("*.npz"))
    sample = dict(np.load(sample_file, allow_pickle=True))
    img_shape = sample['image'][0].flatten().shape[0]
    dm_shape = sample['dmCoefs'][0].flatten().shape[0]
    wfs_shape = sample['wfsSingnal'][0].flatten().shape[0]
    turb_shape = sample['turb_features'][0].flatten().shape[0]
    STATE_DIM = img_shape + dm_shape + wfs_shape + turb_shape
    print(img_shape)
    print("STATE_DIM" , STATE_DIM)
    ACTION_DIM = 19+20+19+20
    action_dims = [19, 20, 19, 20]

    offline_buffer = ReplayBuffer(STATE_DIM, len(action_dims))
    npz_files = list(pathlib.Path(DREAMER_TRAINDIR).glob("*.npz"))
    print(f"找到 {len(npz_files)} 个 npz 文件")
    for npz_file in npz_files:
        episode = dict(np.load(npz_file, allow_pickle=True))
        images = episode['image']
        dmCoefs = episode['dmCoefs']
        wfsSingnal = episode['wfsSingnal']
        turb_features = episode['turb_features']
        actions_onehot = episode['action']
        rewards = episode['reward']
        dones = episode.get('is_terminal', episode.get('is_last'))
        states = [
            np.concatenate([
                img.flatten(),
                dm.flatten(),
                wfs.flatten(),
                turb.flatten()
            ])
            for img, dm, wfs, turb in zip(images, dmCoefs, wfsSingnal, turb_features)
        ]

        # 解码one-hot动作为(N, 4)的整数数组
        actions_decoded = np.array([decode_multidiscrete_onehot(a, action_dims) for a in actions_onehot])
        for i in range(len(states)-1):
            offline_buffer.add(states[i], actions_decoded[i], states[i+1], rewards[i+1], dones[i+1])
    print(f"总共装载了 {offline_buffer.size} 条数据到ReplayBuffer。")


    save_dir = f"/home/jiangbo.chai/DATACENTER4/SWJTU_OOPAO/tutorials/scao_system/myAoSystem/RL4AO/agent/offline/BCQ_offline_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    agent = BCQ(STATE_DIM, action_dims)
    eval_env = AOEnv(args)

    print("\n开始离线训练BCQ智能体...")
    training_steps = 50000
    eval_freq = 5000
    for t in range(training_steps):
        losses = agent.update(offline_buffer, batch_size=128)
        if (t + 1) % eval_freq == 0:
            print(f"Step: {t+1}/{training_steps} | Q Loss: {losses['q_loss']:.3f} | G Loss: {losses['g_loss']:.3f}")
    
    # 训练结束后，使用更新后的 test_agent 函数进行最终评估
    test_agent(agent, eval_env, save_dir, episodes=20)