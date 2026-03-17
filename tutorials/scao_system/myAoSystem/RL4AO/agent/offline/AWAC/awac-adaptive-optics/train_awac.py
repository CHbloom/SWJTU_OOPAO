import numpy as np
import torch
import argparse
import os
import pathlib
from datetime import datetime
import sys
# --- 核心修改：导入 matplotlib 用于绘图 ---
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append('/DATACENTER5/jicheng.liu/SWJTU_OOPAO/SWJTU_OOPAO/tutorials/scao_system/myAoSystem/RL4AO')
from agent.awac import AWAC
from envs.AdaptiveOpticsEnvR2DISC import AOEnv
from skimage.transform import resize
from my_utils.helpers import save_model
from my_utils.replay_buffer import ReplayBuffer

# --- 新增：从 utils.py 移植过来的辅助函数，用于将离散动作索引转换为实际值 ---
def get_action_from_discrete(discrete_action):
    """
    将离散动作转化为实际的控制值。
    注意：这里的数值范围需要和您的环境定义保持一致。
    """
    # 这些值应该与您的环境或数据生成过程中的离散化方式相匹配
    gain_values = np.arange(0.1, 1.05, 0.05)
    sampling_rate_values = np.arange(100, 2100, 100)
    exposure_time_values = np.arange(0.1, 1.05, 0.05)
    clock_rate_values = np.arange(100, 2100, 100)

    # 使用离散的动作索引来获取实际值
    gain = gain_values[discrete_action[0]]
    sampling_rate = sampling_rate_values[discrete_action[1]]
    exposure_time = exposure_time_values[discrete_action[2]]
    clock_rate = clock_rate_values[discrete_action[3]]

    return np.array([gain, sampling_rate, exposure_time, clock_rate])

def decode_multidiscrete_onehot(onehot, dims):
    """把one-hot向量还原为MultiDiscrete的整数动作"""
    indices = []
    pointer = 0
    for dim in dims:
        sub = onehot[pointer:pointer+dim]
        indices.append(np.argmax(sub))
        pointer += dim
    return np.array(indices, dtype=np.int32)

# --- 核心修改：大幅增强 evaluate_agent 函数，增加绘图功能 ---
def evaluate_agent(agent, env, save_dir, episodes=10, eval_type="offline"):
    """
    评估智能体在环境中的性能，并保存详细的性能曲线图。
    """
    test_save_dir = os.path.join(save_dir, "test_results", eval_type)
    os.makedirs(test_save_dir, exist_ok=True)
    
    all_returns = []
    all_avg_sr = []
    all_avg_wfe = []
    print("\n" + "="*50)
    print(f"开始进行 {episodes} 轮性能评估 ({eval_type})...")
    print("="*50)

    for ep in range(episodes):
        state, _ = env.reset(seed=10000 + ep) # 使用不同种子以评估鲁棒性
        done = False
        total_reward = 0
        
        # --- 修改：为每个回合初始化空的 history 列表 ---
        actions_history = []
        reward_history = []
        sr_history = []
        wfe_history = []

        while not done:
            action_indices = agent.take_action(state, eval_mode=True) # 测试时使用贪心策略
            next_state, reward, terminated, truncated, _ = env.step(action_indices)
            done = terminated or truncated
            
            total_reward += reward
            reward_history.append(reward) 
            # --- 核心修改：在每一步都记录 SR 和 WFE ---
            # 假设 env.current_step 从 1 开始计数
            current_step_index = env.current_step - 1
            if current_step_index < len(env.SR):
                sr_history.append(env.SR[current_step_index])
            if current_step_index < len(env.residual):
                wfe_history.append(env.residual[current_step_index])
            
            state = next_state
            
            # 仅记录动作历史
            actions_history.append(get_action_from_discrete(action_indices))
        
        # --- 核心修改：移除循环后的赋值操作 ---
        # sr_history = env.SR  (已在循环中处理)
        # wfe_history = env.residual (已在循环中处理)

        all_returns.append(total_reward)
        # 检查 history 列表是否为空，避免 np.mean 报错
        avg_sr = np.mean(sr_history) if sr_history else 0
        avg_wfe = np.mean(wfe_history) if wfe_history else 0
        
        all_avg_sr.append(avg_sr)
        all_avg_wfe.append(avg_wfe)

        print(f"评估回合 {ep+1}/{episodes}: 总回报 = {total_reward:.2f}, 平均SR = {avg_sr:.3f}, 平均WFE = {avg_wfe:.2f} nm")

        # --- 开始绘图 ---
        actions_history = np.array(actions_history)

        # 1. 绘制 SR, WFE 和 Reward 曲线
        plt.figure(figsize=(18, 5))
        
        # SR 曲线
        plt.subplot(1, 3, 1)
        plt.plot(sr_history, label='Strehl Ratio')
        plt.title(f'Episode {ep+1} Strehl Ratio (Avg: {np.mean(sr_history):.3f})')
        plt.xlabel('Step')
        plt.ylabel('SR')
        plt.grid(True)
        plt.legend()

        # WFE 曲线
        plt.subplot(1, 3, 2)
        plt.plot(wfe_history, label='WFE (nm)', color='orange')
        plt.title(f'Episode {ep+1} WFE (Avg: {np.mean(wfe_history):.2f} nm)')
        plt.xlabel('Step')
        plt.ylabel('Wavefront Error (nm)')
        plt.grid(True)
        plt.legend()

        # Reward 曲线
        plt.subplot(1, 3, 3)
        plt.plot(reward_history, label='Step Reward', color='green')
        plt.title(f'Episode {ep+1} Step Rewards (Total: {total_reward:.2f})')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(test_save_dir, f"perf_episode_{ep+1}.png"))
        plt.close()

        # 2. 绘制动作曲线
        plt.figure(figsize=(12, 8))
        action_labels = ['Gain', 'Sampling Rate', 'Exposure Time', 'Clock Rate']
        for i in range(actions_history.shape[1]):
            plt.subplot(2, 2, i + 1)
            plt.plot(actions_history[:, i])
            plt.title(f'Action: {action_labels[i]}')
            plt.xlabel('Step')
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(test_save_dir, f"actions_episode_{ep+1}.png"))
        plt.close()

    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    mean_sr = np.mean(all_avg_sr)
    std_sr = np.std(all_avg_sr)
    mean_wfe = np.mean(all_avg_wfe)
    std_wfe = np.std(all_avg_wfe)
    
    print("\n" + "="*60)
    print(f"      评估结果统计 ({eval_type})      ")
    print("="*60)
    print(f"总测试回合数: {episodes}")
    print(f"回报 (Return):")
    print(f"  - 平均值: {mean_return:.2f}, 标准差: {std_return:.2f}")
    print(f"  - 最高: {np.max(all_returns):.2f}, 最低: {np.min(all_returns):.2f}")
    print(f"平均斯特列尔比 (SR):")
    print(f"  - 平均值: {mean_sr:.3f}, 标准差: {std_sr:.3f}")
    print(f"平均波前误差 (WFE, nm):")
    print(f"  - 平均值: {mean_wfe:.2f}, 标准差: {std_wfe:.2f}")
    print("="*60)
    
    # --- 新增：绘制最终评估结果的统计图 ---
    plt.figure(figsize=(15, 5))
    
    # 子图1: 总回报的箱形图
    plt.subplot(1, 3, 1)
    plt.boxplot(all_returns, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title(f'Total Returns\n(Mean: {mean_return:.2f} ± {std_return:.2f})')
    plt.ylabel('Return')
    plt.grid(True, axis='y')

    # 子图2: 平均SR的箱形图
    plt.subplot(1, 3, 2)
    plt.boxplot(all_avg_sr, vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    plt.title(f'Average Strehl Ratio\n(Mean: {mean_sr:.3f} ± {std_sr:.3f})')
    plt.ylabel('SR')
    plt.grid(True, axis='y')

    # 子图3: 平均WFE的箱形图
    plt.subplot(1, 3, 3)
    plt.boxplot(all_avg_wfe, vert=True, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    plt.title(f'Average WFE (nm)\n(Mean: {mean_wfe:.2f} ± {std_wfe:.2f})')
    plt.ylabel('WFE (nm)')
    plt.grid(True, axis='y')

    plt.suptitle(f'Evaluation Summary ({eval_type} - {episodes} episodes)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局为总标题留出空间
    plt.savefig(os.path.join(test_save_dir, "summary_boxplot.png"))
    plt.close()

    # 保存评估回报的原始数据
    np.savez(os.path.join(test_save_dir, f"eval_summary.npz"), 
             returns=all_returns,
             avg_srs=all_avg_sr,
             avg_wfes=all_avg_wfe)
             
    return mean_return

def main(args):
    """
    主训练函数，执行AWAC的离线预训练和在线微调。
    """
    print("--- 开始 main 函数 ---", flush=True)
    # --- 1. 初始化环境和智能体 ---
    save_dir = f"/DATACENTER5/jicheng.liu/SWJTU_OOPAO/SWJTU_OOPAO/tutorials/scao_system/myAoSystem/RL4AO/agent/offline/AWAC/awac_results/{args.paramFile}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    eval_env = AOEnv(args)
    
    # --- 核心修改 2: 直接从环境获取 observation_space 和 action_dims ---
    observation_space = eval_env.observation_space
    action_dims = eval_env.action_space.nvec.tolist()
    action_dim_flat = len(action_dims)

    print(f"从环境获取的 Observation Space: {observation_space}")
    print(f"从环境获取的动作维度: {action_dims}")

    # 使用 observation_space 初始化 Agent，并传入融合参数 ---
    agent = AWAC(
        observation_space=observation_space, 
        action_dims=action_dims, 
        hidden_dim=256, 
        lr=args.lr, 
        awac_lambda=args.awac_lambda,
        fusion_mode=args.fusion_mode,
        use_film=not args.no_film
    )
    
    # --- 核心修改 4: 使用支持字典的 ReplayBuffer ---
    replay_buffer = ReplayBuffer(observation_space, action_dim_flat, max_size=int(1e6))

    # --- 数据加载 ---
    print("正在加载离线数据集...")
    npz_files = list(pathlib.Path(args.offline_dataset_dir).glob("*.npz"))
    print(f"找到 {len(npz_files)} 个 npz 数据文件。")
    
    total_transitions = 0
    # --- 核心修改：定义目标图像分辨率 ---
    target_image_res = 64 

    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            # 提取数据流
            images_hwt = data['image']          # 形状: (N, H, W, T) -> (201, 120, 120, 3)
            dm_coefs_flat = data['dmCoefs']     # 形状: (N, T*Dim) -> (201, 2568)
            wfs_signal_flat = data['wfsSingnal']# 形状: (N, T*Dim) -> (201, 4632)
            actions_onehot = data['action']
            rewards = data['reward']
            dones = data['is_terminal']
            
            actions_decoded = np.array([decode_multidiscrete_onehot(a, action_dims) for a in actions_onehot])

            # 将图像从 (H, W, T) 转为 (T, H, W)
            images_thw = np.transpose(images_hwt, (0, 3, 1, 2)) # -> (201, 3, 120, 120)

            num_transitions = len(images_thw) - 1
            for i in range(num_transitions):
                # --- 核心修改：在添加数据前进行 resize ---
                original_img = images_thw[i] # (3, 120, 120)
                resized_img = np.array([resize(img_ch, (target_image_res, target_image_res)) for img_ch in original_img]) # -> (3, 64, 64)

                state = {
                    'far_img': resized_img,
                    'slope_vec': wfs_signal_flat[i].reshape((3, -1)),
                    'dm_vec': dm_coefs_flat[i].reshape((3, -1))
                }
                
                original_next_img = images_thw[i+1]
                resized_next_img = np.array([resize(img_ch, (target_image_res, target_image_res)) for img_ch in original_next_img])

                next_state = {
                    'far_img': resized_next_img,
                    'slope_vec': wfs_signal_flat[i+1].reshape((3, -1)),
                    'dm_vec': dm_coefs_flat[i+1].reshape((3, -1))
                }
                
                action = actions_decoded[i]
                reward = rewards[i+1]
                done = dones[i+1]
                
                replay_buffer.add(state, action, next_state, reward, done)

            total_transitions += num_transitions
        except KeyError as e:
            print(f"加载文件 {npz_file} 失败: 键名错误 - {e}。请检查 .npz 文件内容。")
        except Exception as e:
            print(f"加载文件 {npz_file} 失败: {e}")
            
    print(f"成功从离线数据集中加载了 {total_transitions} 条转移数据到回放缓冲区。")

    # --- 3. 离线预训练阶段 ---
    print("\n" + "="*50)
    print("开始离线预训练...")
    print("="*50)
    for t in range(args.offline_steps):
        loss_info = agent.update(replay_buffer, args.batch_size)
        
        if (t + 1) % 1000 == 0:
            print(f"离线训练步骤 [{t+1}/{args.offline_steps}], Critic Loss: {loss_info['critic_loss']:.4f}, Actor Loss: {loss_info['actor_loss']:.4f}")
        
        if (t + 1) % args.eval_freq == 0:
            evaluate_agent(agent, eval_env, save_dir, episodes=5, eval_type=f"offline_step_{t+1}")
            # 保存预训练模型
            save_model(agent.actor, os.path.join(save_dir, f"actor_offline_{t+1}.pth"))
            save_model(agent.critic, os.path.join(save_dir, f"critic_offline_{t+1}.pth"))

    print("离线预训练完成。")

    # --- 4. 在线微调阶段 ---
    print("\n" + "="*50)
    print("开始在线微调...")
    print("="*50)
    
    online_env = AOEnv(args) # 为在线交互创建一个新的环境实例
    state, _ = online_env.reset()
    
    for t in range(args.online_steps):
        # 与环境交互
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = online_env.step(action)
        done = terminated or truncated
        
        # 将新经验存入缓冲区
        replay_buffer.add(state, action, next_state, reward, done)
        
        # 更新状态
        state = next_state
        if done:
            state, _ = online_env.reset()

        # 从缓冲区采样进行更新
        loss_info = agent.update(replay_buffer, args.batch_size)

        if (t + 1) % 1000 == 0:
            print(f"在线微调步骤 [{t+1}/{args.online_steps}], Critic Loss: {loss_info['critic_loss']:.4f}, Actor Loss: {loss_info['actor_loss']:.4f}")

        if (t + 1) % args.eval_freq == 0:
            evaluate_agent(agent, eval_env, save_dir, episodes=10, eval_type=f"online_step_{t+1}")
            # 保存微调后的模型
            save_model(agent.actor, os.path.join(save_dir, f"actor_online_{t+1}.pth"))
            save_model(agent.critic, os.path.join(save_dir, f"critic_online_{t+1}.pth"))

    print("在线微调完成。")
    
    # --- 5. 最终评估 ---
    print("\n训练结束，进行最终性能评估。")
    evaluate_agent(agent, eval_env, save_dir, episodes=20, eval_type="final")


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="使用AWAC算法训练自适应光学智能体")
    
    # 环境相关参数
    parser.add_argument('--paramFile', type=str, default='1-1', help='自适应光学系统参数文件名')
    parser.add_argument('--max_step', type=int, default=200, help='每个回合的最大步数')
    parser.add_argument('--gainCL', type=float, default=0.6, help='闭环初始增益')
    parser.add_argument('--sampling_rate', type=int, default=1000, help='初始采样率')
    parser.add_argument('--exposure_time', type=float, default=1.0, help='初始曝光时间')
    parser.add_argument('--clock_rate', type=int, default=1000, help='初始时钟频率')
    parser.add_argument('--lightRatio', type=float, default=0.6, help='WFS分光比')

    # 训练流程参数
    parser.add_argument('--offline_dataset_dir', type=str, default='/DATACENTER5/jicheng.liu/SWJTU_OOPAO/SWJTU_OOPAO/tutorials/scao_system/myAoSystem/RL4AO/agent/dreamerv3/logdir/WindSpeed_Direction_Experiment2/1-1/train_eps', help='离线数据集目录路径')
    parser.add_argument('--offline_steps', type=int, default=5000, help='离线预训练的总步数')
    parser.add_argument('--online_steps', type=int, default=5000, help='在线微调的总步数')
    parser.add_argument('--eval_freq', type=int, default=5000, help='评估频率（每N步评估一次）')
    
    # AWAC算法超参数
    parser.add_argument('--batch_size', type=int, default=256, help='训练批次大小')
    parser.add_argument('--lr', type=float, default=5e-4, help='优化器的学习率')
    parser.add_argument('--awac_lambda', type=float, default=3.0, help='AWAC优势加权的温度系数')

    # --- 新增：模型架构参数 ---
    parser.add_argument('--fusion_mode', type=str, default='concat', choices=['concat', 'mamba'], help='选择特征融合模块 (concat 或 mamba)')
    parser.add_argument('--no_film', action='store_true', help='如果使用mamba融合，禁用FiLM调制')

    args = parser.parse_args()
    main(args)