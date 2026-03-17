import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from gym import spaces

# --- 核心修改 1：从 fusion_modules 导入解耦的模块 ---
from .fusion_modules import ImageEncoder, VectorEncoder, ConcatFusion, AOMambaFusion

# --- 核心修改 2：定义新的、可插拔的特征提取器 ---
class MultiModalFeatureExtractor(nn.Module):
    def __init__(self, obs_space, feature_dim=256, fusion_mode='concat', use_film=True):
        super().__init__()
        
        # 1. 初始化独立编码器
        self.encoders = nn.ModuleDict()
        vector_keys = []
        for key, space in obs_space.spaces.items():
            if 'img' in key:
                self.encoders['image'] = ImageEncoder(time_window=space.shape[0], feature_dim=feature_dim)
            elif 'vec' in key:
                self.encoders[key] = VectorEncoder(input_dim=space.shape[1], time_window=space.shape[0], feature_dim=feature_dim)
                vector_keys.append(key)
        
        self.vector_keys = sorted(vector_keys)
        num_modalities = 1 + len(self.vector_keys)

        # 2. 根据 fusion_mode 选择并初始化融合模块
        if fusion_mode == 'concat':
            self.fusion_module = ConcatFusion(feature_dim=feature_dim, num_modalities=num_modalities)
        elif fusion_mode == 'mamba':
            self.fusion_module = AOMambaFusion(
                feature_dim=feature_dim,
                num_vectors=len(self.vector_keys),
                use_film=use_film
            )
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        
        self.features_dim = self.fusion_module.output_dim

    def forward(self, obs):
        # obs 是一个字典: {'far_img': tensor, 'slope_vec': tensor, 'dm_vec': tensor}
        image_feature = self.encoders['image'](obs['far_img'])
        vector_features = [self.encoders[key](obs[key]) for key in self.vector_keys]
        fused_features = self.fusion_module(image_feature, vector_features)
        return fused_features

# --- Actor 和 Critic (无需修改) ---
class Actor(nn.Module):
    def __init__(self, features_dim, action_dims, hidden_dim=256):
        super().__init__()
        self.action_dims = action_dims
        self.net = nn.Sequential(
            nn.Linear(features_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, sum(action_dims))
        )
    def forward(self, features):
        return torch.split(self.net(features), self.action_dims, dim=1)
    def sample(self, features):
        logits_tuple = self.forward(features)
        actions, log_probs = [], 0
        for logits in logits_tuple:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            actions.append(action.unsqueeze(1))
            log_probs += dist.log_prob(action)
        return torch.cat(actions, dim=1), log_probs.unsqueeze(1)

class Critic(nn.Module):
    def __init__(self, features_dim, action_dim_total, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features_dim + action_dim_total, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, features, action_one_hot):
        return self.net(torch.cat([features, action_one_hot], 1))

# --- AWAC 主类 ---
class AWAC:
    def __init__(self, observation_space, action_dims, hidden_dim=256, discount=0.99, lr=3e-4, tau=0.005, awac_lambda=1.0, fusion_mode='concat', use_film=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dims = action_dims
        self.action_dim_total = sum(action_dims)
        self.num_action_components = len(action_dims)
        self.discount, self.tau, self.awac_lambda = discount, tau, awac_lambda

        # --- 核心修改 3：使用新的 MultiModalFeatureExtractor ---
        self.feature_extractor = MultiModalFeatureExtractor(
            observation_space,
            feature_dim=hidden_dim,
            fusion_mode=fusion_mode,
            use_film=use_film
        ).to(self.device)
        
        features_dim = self.feature_extractor.features_dim

        self.actor = Actor(features_dim, action_dims, hidden_dim).to(self.device)
        self.critic = Critic(features_dim, self.action_dim_total, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # 将特征提取器的参数也加入优化器
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + list(self.feature_extractor.parameters()), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def _to_one_hot(self, action):
        batch_size = action.shape[0]
        one_hots = []
        for i, dim in enumerate(self.action_dims):
            component_action = action[:, i].unsqueeze(1)
            one_hot = torch.zeros(batch_size, dim, device=self.device).scatter_(1, component_action, 1)
            one_hots.append(one_hot)
        return torch.cat(one_hots, dim=1)

    def take_action(self, state_dict, eval_mode=False):
        obs_tensor = {k: torch.from_numpy(v).unsqueeze(0).float().to(self.device) for k, v in state_dict.items()}
        with torch.no_grad():
            features = self.feature_extractor(obs_tensor)
            logits_tuple = self.actor(features)
            if eval_mode:
                # print(f"Action probabilities: {[F.softmax(logits, dim=1).cpu().numpy() for logits in logits_tuple]}")
                # 测试时选概率最大的动作（贪心）
                actions = [torch.argmax(logits, dim=1).item() for logits in logits_tuple]
            else:
                # 训练时按分布采样（探索）
                actions = [torch.multinomial(F.softmax(logits, dim=1), 1).item() for logits in logits_tuple]
        return np.array(actions, dtype=np.int32)


    # --- 核心修改：简化 update 函数以处理已批处理的数据 ---
    def update(self, buffer, batch_size=256):
        # 1. 从 Replay Buffer 采样一个批次的数据。
        state_dict, action, next_state_dict, reward, done = buffer.sample(batch_size)
        
        # --- 更新 Critic (Q-Network) ---
        # 首先计算 Critic 的损失
        with torch.no_grad():
            next_features = self.feature_extractor(next_state_dict)
            next_action, _ = self.actor.sample(next_features)
            next_action_one_hot = self._to_one_hot(next_action)
            target_q = self.critic_target(next_features, next_action_one_hot)
            target_q = reward + (1 - done) * self.discount * target_q
        
        # 这里的 features 会被用于 Critic 和 Actor 的更新
        features = self.feature_extractor(state_dict)
        action_one_hot = self._to_one_hot(action)
        current_q = self.critic(features, action_one_hot)
        
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 优化 Critic (注意：这里只更新 Critic)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 更新 Actor ---
        # 重新计算 features，但这次是为了 Actor 的更新，所以梯度需要流过 feature_extractor
        features_for_actor = self.feature_extractor(state_dict)
        
        with torch.no_grad():
            # 重新计算 current_q，但这次不需要梯度
            current_q_detached = self.critic(features_for_actor, action_one_hot)
            
            # 计算 V(s)
            sampled_action_for_v, _ = self.actor.sample(features_for_actor)
            v_s = self.critic(features_for_actor, self._to_one_hot(sampled_action_for_v))
            
            # 核心修改：从 current_q_detached 计算优势，切断梯度流向 Critic
            advantage = current_q_detached - v_s
            weights = torch.exp(advantage / self.awac_lambda).clamp(max=100.0) # .detach() is implicitly done by no_grad

        action_logits_tuple = self.actor(features_for_actor)
        log_probs = 0
        for i in range(self.num_action_components):
            log_softmax_dist = F.log_softmax(action_logits_tuple[i], dim=1)
            log_probs += log_softmax_dist.gather(1, action[:, i].unsqueeze(1))

        # Actor 的损失会同时更新 Actor 和 FeatureExtractor
        actor_loss = -torch.mean(weights * log_probs)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 软更新目标网络 ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "advantage_mean": advantage.mean().item()
        }