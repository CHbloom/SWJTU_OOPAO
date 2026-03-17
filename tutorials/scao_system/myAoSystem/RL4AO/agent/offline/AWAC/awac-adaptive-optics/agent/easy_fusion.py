import torch
import torch.nn as nn
# --- 特征提取器 ---
class FeatureExtractor(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.cnns = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        total_features_dim = 0

        for key, space in obs_space.spaces.items():
            if len(space.shape) == 3:  # 图像 (T, H, W)
                # PyTorch CNN期望 (B, C, H, W)，这里的 T 扮演了 C 的角色
                in_channels = space.shape[0]
                self.cnns[key] = nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                    nn.Flatten(),
                )
                with torch.no_grad():
                    dummy_input = torch.zeros(1, *space.shape)
                    cnn_out_dim = self.cnns[key](dummy_input).shape[1]
                total_features_dim += cnn_out_dim
            elif len(space.shape) == 2: # 向量 (T, Dim)
                self.mlps[key] = nn.Flatten()
                total_features_dim += space.shape[0] * space.shape[1]
        
        self.features_dim = total_features_dim

    def forward(self, obs):
        features = []
        for key, cnn in self.cnns.items():
            features.append(cnn(obs[key]))
        for key, mlp in self.mlps.items():
            features.append(mlp(obs[key]))
        return torch.cat(features, dim=1)