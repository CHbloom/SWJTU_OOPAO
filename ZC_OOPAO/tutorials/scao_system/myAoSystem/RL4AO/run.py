import argparse
from agent.PPO import PPO
from agent.SAC import SACContinuous
from env.AdaptiveOpticsEnv import AOEnv
import torch
import numpy as np
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
def parse_args():
    parser = argparse.ArgumentParser(description="Run AO system reinforcement learning experiment with different algorithms.")
    parser.add_argument("--agent", type=str, default="PPO", choices=["PPO", "SAC", "Dreamer-v3"], help="Reinforcement learning algorithm")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=0, help="seed") # AOEnv随机种子未实现------------
    parser.add_argument("--max_step", type=int, default=200, help="AOEnv max_step")
    parser.add_argument("--gainCL", type=float, default=0.6, help="Wavefront sensor gain")
    parser.add_argument("--sampling_rate", type=int, default=1000, help="Telescope sampling frequency (Hz)")
    parser.add_argument("--exposure_time", type=float, default=1, help="Camera exposure time (seconds)")
    parser.add_argument("--clock_rate", type=int, default=1000, help="Camera clock rate (Hz)")
    parser.add_argument("--lightRatio", type=float, default=0.3, help="wfs lightRatio")
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    # 创建环境
    env = AOEnv(args)
    save_dir = utils.setup_dirs(args)
    # # 根据选择的算法初始化代理
    if args.agent == "PPO":
        actor_lr = 1e-4
        critic_lr = 5e-3
        gamma = 0.9
        lamda = 0.9
        epsilon = 0.2
        epochs = 10
        episodes = args.episodes
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        hidden_dim = 256
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lamda, epochs, epsilon, gamma, device)
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
                        next_state,  reward,  d1, d2, _ = env.step(action)
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
        # 最终保存
        torch.save(agent.Actor.state_dict(), f"./data/{save_dir}/train/models/final_actor_{timestamp}.pth")
        torch.save(agent.Critic.state_dict(), f"./data/{save_dir}/train/models/final_critic_{timestamp}.pth")
        utils.save_training_data(return_list, average_sr_list, average_wfe_list, save_dir, average_sr_list_old)

        utils.test_agent(agent, env, save_dir)
        
    elif args.agent == "SAC":
        env_name = 'AOEvn'
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        actor_lr = 3e-4
        critic_lr = 3e-3
        alpha_lr = 3e-4
        num_episodes = args.episodes
        hidden_dim = 128
        gamma = 0.99
        tau = 0.005  # 软更新参数
        buffer_size = 100000
        minimal_size = 1000
        batch_size = 64
        target_entropy = -env.action_space.shape[0]
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")

        replay_buffer = utils.ReplayBuffer(buffer_size)
        agent = SACContinuous(state_dim, hidden_dim, action_dim, 
                            actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                            gamma, device)

        return_list = []
        average_sr_list = []  # 一回合平均斯特列尔比
        average_sr_list_old = []
        average_wfe_list = []  # 波前误差

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
                            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                            'dones': b_d}
                            agent.update(transition_dict)
                    return_list.append(episode_return)
                    average_sr_list.append(np.mean(episode_sr))
                    average_sr_list_old.append(np.mean(episode_sr_old))
                    average_wfe_list.append(np.mean(episode_wfe))
                    # # 保存最佳模型
                    # current_mean_return = np.mean(return_list[-10:])
                    # if current_mean_return > best_mean_return:
                    #     best_mean_return = current_mean_return
                    #     torch.save({
                    #         'actor_state_dict': agent.actor.state_dict(),
                    #         'critic_state_dict': agent.critic.state_dict(),
                    #         'optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    #         'best_return': best_mean_return
                    #     }, f"./data/{save_dir}/train/models/best_model_{timestamp}.pth")

                    if i_episode % 10 == 0 and len(return_list) >= 10:
                        pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                        'return': f"{episode_return:.1f}",
                                        'avg_SR': f"{np.mean(episode_sr):.3f}",
                                        'avg_WFE': f"{np.mean(episode_wfe):.1f}nm"
                                        })
                    pbar.update(1)
        # # 最终保存
        # torch.save(agent.Actor.state_dict(), f"./data/{save_dir}/train/models/final_actor_{timestamp}.pth")
        # torch.save(agent.Critic.state_dict(), f"./data/{save_dir}/train/models/final_critic_{timestamp}.pth")
        utils.save_training_data(return_list, average_sr_list, average_wfe_list, save_dir, average_sr_list_old)

        utils.test_agent(agent, env, save_dir)

    elif args.agent == "Dreamer-v3":
        pass