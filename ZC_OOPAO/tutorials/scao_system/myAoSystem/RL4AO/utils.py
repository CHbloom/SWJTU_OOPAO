import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import collections
import random
from tqdm import tqdm


def moving_average(a, window_size):
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def setup_dirs(args):
    """创建结果保存目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.agent}_episodes{args.episodes}_gainCL{args.gainCL}_samplingRate{args.sampling_rate}"
    save_dir = save_dir + f"_exposureTime{args.exposure_time}_clockRate{args.clock_rate}_lightRatio{args.lightRatio}"
    save_dir = save_dir + f"_{timestamp}"
    train_dir = "./data/"+f"{save_dir}"+"/train/"
    test_dir = "./data/"+f"{save_dir}"+"/test/"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(f"{train_dir}/models", exist_ok=True)
    os.makedirs(f"{train_dir}/plots", exist_ok=True)
    os.makedirs(f"{train_dir}/logs", exist_ok=True)
    os.makedirs(f"{test_dir}/plots", exist_ok=True)

    return save_dir


def save_training_data(return_list, sr_list, wfe_list, save_dir,sr_list_old):
    """保存训练数据到文件"""
    save_dir = "./data/"+save_dir+"/train"
    # 保存原始数据
    np.savez(f"{save_dir}/logs/training_data_returns_sr_wfe_episode0_{len(return_list)}.npz",
             returns=np.array(return_list),
             strehl_ratio=np.array(sr_list),
             wavefront_error=np.array(wfe_list))
    window_size = 10
    return_list = moving_average(return_list, window_size)
    sr_list = moving_average(sr_list, window_size)
    wfe_list = moving_average(wfe_list, window_size)
    # 生成训练曲线图
    plt.figure(figsize=(12,4))
    
    plt.subplot(131)
    plt.plot(return_list)
    plt.title("Episode Returns")
    plt.xlabel("Episode")
    plt.grid()
    
    plt.subplot(132)
    plt.plot(sr_list)
    plt.title("Average Strehl Ratio")
    plt.xlabel("Episode")
    plt.grid()
    
    plt.subplot(133)
    plt.plot(wfe_list)
    plt.title("Average WFE (nm)")
    plt.xlabel("Episode")
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plots/training_curves.png")
    plt.close()

    plt.figure()

    plt.plot(sr_list_old)
    plt.title("Average Strehl Ratio old")
    plt.xlabel("Episode")
    plt.grid()
    plt.savefig(f"{save_dir}/plots/Average_SR_old.png")
    plt.close()

def test_agent(agent, env, save_dir, episodes=1):
    save_dir = "./data/"+save_dir+"/test/"
    x = 0
    while x<episodes:
        x+=1
        state, _ = env.reset()
        episode_return_agent = 0
        done = False
        gainCL_list = []
        sampling_rate_list = []
        exposure_time_list = []
        clock_rate_list = []
        while not done:
            action = agent.take_action(state)
            gainCL_list.append(np.clip(action[0],0.1,1))
            sampling_rate_list.append(np.clip(action[1],100,2000))
            exposure_time_list.append(np.clip(action[2],0.1,1))
            clock_rate_list.append(np.clip(action[3],100,2000))
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            episode_return_agent += reward
        print('return=',episode_return_agent)
        plt.figure()
        plt.plot(np.arange(env.max_step), env.SR, label='Strehl Ratio')
        plt.xlabel('Loop Index')
        plt.title(f'Strehl Ratio(return={episode_return_agent})')
        plt.legend()
        plt.grid()
        plt.savefig(f"{save_dir}/plots/SR_episode_{x}.png")
        plt.close()

        plt.figure()
        plt.plot(np.arange(env.max_step), env.SR_old, label='Strehl Ratio old')
        plt.xlabel('Loop Index')
        plt.title(f'Strehl Ratio(return={episode_return_agent})')
        plt.legend()
        plt.grid()
        plt.savefig(f"{save_dir}/plots/SR_episode_old{x}.png")
        plt.close()

        plt.figure(figsize=(12,12))
    
        plt.subplot(221)
        plt.plot(gainCL_list)
        plt.title("gainCL")
        plt.xlabel("Loop Index")
        plt.grid()
        
        plt.subplot(222)
        plt.plot(sampling_rate_list)
        plt.title("sampling_rate")
        plt.xlabel("Loop Index")
        plt.grid()
        
        plt.subplot(223)
        plt.plot(exposure_time_list)
        plt.title("exposure_time")
        plt.xlabel("Loop Index")
        plt.grid()

        plt.subplot(224)
        plt.plot(clock_rate_list)
        plt.title("clock_rate")
        plt.xlabel("Loop Index")
        plt.grid()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/plots/actions_episode_{x}.png")
        plt.close()
            
            
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
        
def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset(seed=1)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list