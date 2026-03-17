'''
测试训练好的模型，将结果保存到logdir_test
'''

import argparse
import os
import pathlib
import sys
import gym.spaces
from tqdm import trange
os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml
import gym

sys.path.append(str(pathlib.Path(__file__).parent))

import tools
import envs.wrappers as wrappers
from dreamer import Dreamer
import torch
# from envs.AdaptiveOpticsEnvR1DISC import AOEnv
from envs.AdaptiveOpticsEnvR2DISC import AOEnv
import matplotlib.pyplot as plt



def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    # print(episodes)cd
    # for episode in episodes.values():
    #     for k,v in episode.items():
    #             print('    ',k,len(v))
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


acts = None
def make_env(config, mode, id, args):
    # args.paramFile = 'WindSpeed_Direction_Experiment2/1-1'
    env = AOEnv(args)
    global acts
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        acts = env.action_space
        env = wrappers.OneHotMultiDiscreteAction(env) #离散
    else:
        env = wrappers.NormalizeActions(env) #连续
    
    env = wrappers.TimeLimit(env, config.time_limit)
    # env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    
    return env


def main(config, args):
    # tools.set_seed_everywhere(config.seed)
    config.action_repeat = 1
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    # 创建环境
    # env = AOEnv(args)
    # env = wrappers.NormalizeActions(env)
    # acts = env.action_space
    # print("Action Space", acts)
    # config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    env = make_env(config,'',0,args)
    global acts
    if isinstance(acts, gym.spaces.MultiDiscrete):
        nvec = acts.nvec.tolist()
        config.shape = {"action"+str(i):nvec[i] for i in range(len(nvec))}
        config.num_actions = sum(nvec)
    else:
        acts = env.action_space
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        config.shape = config.num_actions

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
    agent.eval()
    print(f"✅ 成功加载模型 from {logdir}/latest.pt")
    
    # === 4. 执行推理评估 ===
    def add_batch_dim(obs_dict):
        return {
            k: torch.tensor(v, dtype=torch.float32)[None].to(config.device) if isinstance(v, np.ndarray) else v
            for k, v in obs_dict.items()
    }
    def act_to_origin(action):
        '''归一化动作->原始动作'''
        action[0] = (action[0] + 1) / 2 * (1 - 0.1) + 0.1
        action[1] = (action[1] + 1) / 2 * (2000 - 100) + 100
        action[2] = (action[2] + 1) / 2 * (1 - 0.1) + 0.1
        action[3] = (action[3] + 1) / 2 * (2000 - 100) + 100

        return action
    
    def decode_onehot(acts, onehot):
        """将拼接的onehot向量转换为 MultiDiscrete 索引"""
        indices = []
        pointer = 0
        # print('onehot ',onehot)
        for dim in acts.nvec:
            sub = onehot[pointer:pointer + dim]
            index = np.argmax(sub)
            ref = np.zeros_like(sub)
            ref[index] = 1
            if not np.allclose(ref, sub):
                raise ValueError(f"Invalid one-hot segment: {sub}")
            indices.append(index)
            pointer += dim
        return np.array(indices, dtype=np.int32)

    test_dir = os.path.join(config.logdir, "test")
    os.makedirs(test_dir, exist_ok=True)

    episodes = 2
    returns = []
    # for ep in trange(episodes, desc="Evaluating"):
    #     obs = env.reset()
    #     done = False
    #     total_reward = 0.0
    #     state = None  # Dreamer recurrent state
    #     obs["is_first"] = torch.tensor([True], dtype=torch.bool, device=config.device)
    #     obs = add_batch_dim(obs)
    #     reset = obs["is_first"]  # is_first indicator, batch size = 1

    #     gainCL_list = []
    #     sampling_rate_list = []
    #     exposure_time_list = []
    #     clock_rate_list = []
    #     while not done:
    #         # 推理模式调用 Dreamer
    #         policy_output, state = agent(obs, reset, state, training=False)
    #         action = policy_output["action"]
    #         # 注意：Dreamer 的 action 是 torch.Tensor，需转 numpy
    #         action_np = action[0].cpu().numpy()
    #         obs, reward, done, info = env.step(action_np)
    #         if isinstance(acts, gym.spaces.MultiDiscrete):
    #             gain_idx, sampling_idx, exposure_idx, clock_idx = decode_onehot(acts, action_np)
    #             gain = env.gain_values[gain_idx]
    #             sampling_rate = env.sampling_rate_values[sampling_idx]
    #             exposure_time = env.exposure_time_values[exposure_idx]
    #             clock_rate = env.clock_rate_values[clock_idx]
    #             action_np = np.array([gain, sampling_rate, exposure_time, clock_rate])
    #         else:
    #             action_np = act_to_origin(action_np)
    #         obs["is_first"] = torch.tensor([done], dtype=torch.bool, device=config.device)
    #         obs = add_batch_dim(obs)
    #         reset = obs["is_first"]
    #         total_reward += reward

    #         gainCL_list.append(np.clip(action_np[0],0.1,1))
    #         sampling_rate_list.append(np.clip(action_np[1],100,2000))
    #         exposure_time_list.append(np.clip(action_np[2],0.1,1))
    #         clock_rate_list.append(np.clip(action_np[3],100,2000))

    #     returns.append(total_reward)
    #     print(f"Episode {ep + 1} Return: {total_reward:.2f}")
    #     np.save(f"{config.logdir}/test/SR_test_on{args.paramFile}_{ep + 1}.npy", sum(env.actual_SR) )
    #     plt.figure()
    #     plt.plot(np.arange(env.max_step), env.actual_SR, label='Strehl Ratio')
    #     plt.xlabel('Loop Index')
    #     plt.title(f'Strehl Ratio(env={args.paramFile},SR_mean={sum(env.actual_SR)/len(env.actual_SR):.3f})')
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig(f"{config.logdir}/test/SR_test_on{args.paramFile}_{ep + 1}.png")
    #     plt.close()

    #     # plt.figure()
    #     # plt.plot(np.arange(env.max_step), env.SR_old, label='Strehl Ratio old')
    #     # plt.xlabel('Loop Index')
    #     # plt.title(f'Strehl Ratio(return={total_reward:.2f},avg_SR={sum(env.SR_old)/len(env.SR_old):.2f})')
    #     # plt.legend()
    #     # plt.grid()
    #     # plt.savefig(f"{config.logdir}/test/SR_episode_old{ep + 1}.png")
    #     # plt.close()

    #     plt.figure(figsize=(12,12))
    
    #     plt.subplot(221)
    #     plt.plot(gainCL_list)
    #     plt.title("gainCL")
    #     plt.xlabel("Loop Index")
    #     plt.grid()
        
    #     plt.subplot(222)
    #     plt.plot(sampling_rate_list)
    #     plt.title("sampling_rate")
    #     plt.xlabel("Loop Index")
    #     plt.grid()
        
    #     plt.subplot(223)
    #     plt.plot(exposure_time_list)
    #     plt.title("exposure_time")
    #     plt.xlabel("Loop Index")
    #     plt.grid()

    #     plt.subplot(224)
    #     plt.plot(clock_rate_list)
    #     plt.title("clock_rate")
    #     plt.xlabel("Loop Index")
    #     plt.grid()
        
    #     plt.tight_layout()
    #     plt.savefig(f"{config.logdir}/test/actions_episode_{ep + 1}.png")
    #     plt.close()

    # print(f"\n🎯 平均评估回报: {np.mean(env.actual_SR):.3f} ± {np.std(env.actual_SR):.3f}")

    # === 推理速度测试配置 ===
    import time
    warmup_steps = 10  # 预热步数，排除初次加载的延迟
    total_inference_time = 0.0
    inference_count = 0
    for ep in trange(episodes, desc="Evaluating"):
        obs = env.reset()
        done = False
        state = None
        obs["is_first"] = torch.tensor([True], dtype=torch.bool, device=config.device)
        obs = add_batch_dim(obs)
        reset = obs["is_first"]

        while not done:
            # --- 1. 准备测速 ---
            # 仅测量 agent(obs, reset, state) 这一行
            torch.cuda.synchronize() # 同步 GPU
            
            start_time = time.perf_counter()

            # --- 2. 模型核心前向传播 ---
            with torch.no_grad(): # 确保推理模式不计算梯度，加速且省显存
                policy_output, state = agent(obs, reset, state, training=False)
                action = policy_output["action"]
            
            # --- 3. 结束测速 ---
            torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start_time
            
            # 跳过预热步数进行统计
            if inference_count > warmup_steps:
                total_inference_time += elapsed
            
            inference_count += 1

            # --- 4. 后续环境交互 (不计入推理时间) ---
            action_np = action[0].cpu().numpy()
            obs, reward, done, info = env.step(action_np)
            
            # ... 原有的 action_np 处理逻辑 (decode_onehot 或 act_to_origin) ...
            
            obs["is_first"] = torch.tensor([done], dtype=torch.bool, device=config.device)
            obs = add_batch_dim(obs)
            reset = obs["is_first"]

    # === 5. 输出结果 ===
    if inference_count > warmup_steps:
        valid_count = inference_count - warmup_steps
        avg_time_ms = (total_inference_time / valid_count) * 1000
        fps = 1.0 / (total_inference_time / valid_count)
        
        print(f"\n" + "="*30)
        print(f"🚀 模型推理性能报告 (Inference Only):")
        print(f"平均耗时: {avg_time_ms:.3f} ms / step")
        print(f"推理帧率: {fps:.2f} FPS (Steps Per Second)")
        print("="*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")

    parser.add_argument("--agent", type=str, default="PPO", choices=["PPO", "SAC", "Dreamer-v3"], help="Reinforcement learning algorithm")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=0, help="seed") # AOEnv随机种子未实现------------
    parser.add_argument("--maxStep", type=int, default=500, help="AOEnv max_step")
    parser.add_argument("--gainCL", type=float, default=0.6, help="Wavefront sensor gain")
    parser.add_argument("--samplingRate", type=int, default=1000, help="Telescope sampling frequency (Hz)")
    parser.add_argument("--exposureTime", type=float, default=1, help="Camera exposure time (seconds)")
    parser.add_argument("--clockRate", type=int, default=1000, help="Camera clock rate (Hz)")
    parser.add_argument("--lightRatio", type=float, default=0.6, help="wfs lightRatio")
    parser.add_argument("--paramFile", type=str, default="paramFile1", help="paramFile")
    parser.add_argument("--k", type=int, default=3, help="state queue length")
    parser.add_argument("--delay", type=int, default=0, help="delay")

    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    # name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    name_list = ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    args.size = defaults.get("size")
    main(parser.parse_args(remaining),args)
