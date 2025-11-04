import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import gym
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from torch import nn
from torch import distributions as torchd
# from envs.AdaptiveOpticsEnvR2 import AOEnv
# from envs.AdaptiveOpticsEnvDISC import AOEnv
# from envs.AdaptiveOpticsEnvR1DISC import AOEnv
from envs.AdaptiveOpticsEnvR2DISC import AOEnv

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        print(torch.cuda.current_device(), config.device)  # 应输出 1（对应 cuda:1）
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)

        embed = self._wm.encoder(obs)

        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


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
    
    env = AOEnv(args)
    global acts
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        acts = env.action_space
        env = wrappers.OneHotMultiDiscreteAction(env) #离散
    else:
        env = wrappers.NormalizeActions(env) #连续
    
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    
    return env

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

def test_after_train(env,agent,config):
    def add_batch_dim(obs_dict):
        return {
            k: torch.tensor(v, dtype=torch.float32)[None].to(config.device) if isinstance(v, np.ndarray) else v
            for k, v in obs_dict.items()
    }
    test_dir = os.path.join(config.logdir, "test_after_train")
    os.makedirs(test_dir, exist_ok=True)
    episodes = 3
    returns = []
    for ep in trange(episodes, desc="Evaluating"):
        obs = env.reset()
        done = False
        total_reward = 0.0
        state = None  # Dreamer recurrent state
        obs["is_first"] = torch.tensor([True], dtype=torch.bool, device=config.device)
        obs = add_batch_dim(obs)
        reset = obs["is_first"]  # is_first indicator, batch size = 1

        gainCL_list = []
        sampling_rate_list = []
        exposure_time_list = []
        clock_rate_list = []
        while not done:
            # 推理模式调用 Dreamer
            policy_output, state = agent(obs, reset, state, training=False)
            action = policy_output["action"]

            # 注意：Dreamer 的 action 是 torch.Tensor，需转 numpy
            action_np = action[0].cpu().numpy()

            obs, reward, done, info = env.step(action_np)
            if isinstance(acts, gym.spaces.MultiDiscrete):
                gain_idx, sampling_idx, exposure_idx, clock_idx = decode_onehot(acts, action_np)
                gain = env.gain_values[gain_idx]
                sampling_rate = env.sampling_rate_values[sampling_idx]
                exposure_time = env.exposure_time_values[exposure_idx]
                clock_rate = env.clock_rate_values[clock_idx]
                action_np = np.array([gain, sampling_rate, exposure_time, clock_rate])
            else:
                action_np = act_to_origin(action_np)
            obs["is_first"] = torch.tensor([done], dtype=torch.bool, device=config.device)
            obs = add_batch_dim(obs)
            reset = obs["is_first"]
            total_reward += reward

            gainCL_list.append(np.clip(action_np[0],0.1,1))
            sampling_rate_list.append(np.clip(action_np[1],100,2000))
            exposure_time_list.append(np.clip(action_np[2],0.1,1))
            clock_rate_list.append(np.clip(action_np[3],100,2000))

        returns.append(total_reward)
        print(f"Episode {ep + 1} Return: {total_reward:.2f}")
        np.save(f"{config.logdir}/test_after_train/SR_episode_{ep + 1}.npy", np.array(env.SR)) 
        plt.figure()
        plt.plot(np.arange(env.max_step), env.SR, label='Strehl Ratio')
        plt.xlabel('Loop Index')
        plt.title(f'Strehl Ratio(return={total_reward:.2f},avg_SR={sum(env.SR)/len(env.SR):.2f})')
        plt.legend()
        plt.grid()
        plt.savefig(f"{config.logdir}/test_after_train/SR_episode_{ep + 1}.png")
        plt.close()

        plt.figure()
        plt.plot(np.arange(env.max_step), env.SR_old, label='Strehl Ratio old')
        plt.xlabel('Loop Index')
        plt.title(f'Strehl Ratio(return={total_reward:.2f},avg_SR={sum(env.SR_old)/len(env.SR_old):.2f})')
        plt.legend()
        plt.grid()
        plt.savefig(f"{config.logdir}/test_after_train/SR_episode_old{ep + 1}.png")
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
        plt.savefig(f"{config.logdir}/test_after_train/actions_episode_{ep + 1}.png")
        plt.close()

    print(f"\n🎯 平均评估回报: {np.mean(returns):.2f} ± {np.std(returns):.2f}")



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
    make = lambda mode, id: make_env(config, mode, id, args)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]

    global acts
    if isinstance(acts, gym.spaces.MultiDiscrete):
        nvec = acts.nvec.tolist()
        config.shape = {"action"+str(i):nvec[i] for i in range(len(nvec))}
        config.num_actions = sum(nvec)
    else:
        acts = train_envs[0].action_space
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        config.shape = config.num_actions
    print("Action Space", acts)
    print(config.shape)
        

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if isinstance(acts, gym.spaces.Discrete):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        elif isinstance(acts, gym.spaces.MultiDiscrete):
            nvec = acts.nvec.tolist()
            total_dim = sum(nvec)
            logits = torch.zeros(config.envs, total_dim)
            random_actor = tools.MultiOneHotDist(logits, nvec)
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
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

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    env = AOEnv(args)
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        env = wrappers.OneHotMultiDiscreteAction(env) #离散
    else:
        env = wrappers.NormalizeActions(env) #连续
    test_after_train(env,agent,config)
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")

    parser.add_argument("--agent", type=str, default="PPO", choices=["PPO", "SAC", "Dreamer-v3"], help="Reinforcement learning algorithm")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=0, help="seed") # AOEnv随机种子未实现------------
    parser.add_argument("--maxStep", type=int, default=200, help="AOEnv max_step")
    parser.add_argument("--gainCL", type=float, default=0.6, help="Wavefront sensor gain")
    parser.add_argument("--samplingRate", type=int, default=100, help="Telescope sampling frequency (Hz)")
    parser.add_argument("--exposureTime", type=float, default=0.1, help="Camera exposure time (seconds)")
    parser.add_argument("--clockRate", type=int, default=100, help="Camera clock rate (Hz)")
    parser.add_argument("--lightRatio", type=float, default=0.1, help="wfs lightRatio")
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
