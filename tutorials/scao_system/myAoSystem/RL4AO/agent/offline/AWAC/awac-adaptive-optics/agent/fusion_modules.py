import torch
import torch.nn as nn

# --- 核心修改 1：从本地文件导入 Mamba 实现 ---
try:
    # 使用相对导入，从同目录下的 mamba_minimal.py 导入
    from .mamba_minimal import ResidualBlock, ModelArgs
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    # 定义一个虚拟类以防导入失败，避免程序崩溃
    class ResidualBlock: pass
    class ModelArgs: pass

# --- 1. 独立编码器 ---

class ImageEncoder(nn.Module):
    """轻量级CNN，用于将图像 (B, T, H, W) 编码为特征向量 (B, D)"""
    def __init__(self, time_window=3, feature_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(time_window, 32, kernel_size=5, stride=2), # (B, 32, 30, 30)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), # (B, 64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2), # (B, 128, 6, 6)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2), # (B, 256, 2, 2)
            nn.ReLU(),
            nn.Flatten(), # (B, 256 * 2 * 2) = (B, 1024)
        )
        self.projector = nn.Linear(1024, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def forward(self, image_obs):
        # image_obs shape: (B, T, H, W) e.g. (B, 3, 64, 64)
        x = self.cnn(image_obs)
        x = self.projector(x)
        return self.ln(x)

class VectorEncoder(nn.Module):
    """小型MLP，用于将时序向量 (B, T, Dim) 编码为特征向量 (B, D)"""
    def __init__(self, input_dim, time_window=3, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), # (B, T * Dim)
            nn.Linear(time_window * input_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.ln = nn.LayerNorm(feature_dim)

    def forward(self, vector_obs):
        # vector_obs shape: (B, T, Dim)
        x = self.net(vector_obs)
        return self.ln(x)

# --- 2. FiLM 调制层 ---

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) 层.
    使用一个向量特征来调制另一个特征（通常是图像特征）。
    """
    def __init__(self, modulator_dim, feature_dim):
        super().__init__()
        # 生成 gamma 和 beta 的网络
        self.generator = nn.Linear(modulator_dim, feature_dim * 2)

    def forward(self, feature_to_modulate, modulator_feature):
        # 生成 gamma 和 beta
        gamma_beta = self.generator(modulator_feature)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        
        # 应用 FiLM: y = gamma * x + beta
        return gamma * feature_to_modulate + beta

# --- 3. 核心融合模块 ---

# --- 新增：将您的拼接方案封装成一个模块 ---
class ConcatFusion(nn.Module):
    """
    简单的拼接融合模块。
    接收编码后的特征列表，并将它们在特征维度上拼接起来。
    """
    def __init__(self, feature_dim=256, num_modalities=3):
        super().__init__()
        # 最终的输出维度是所有模态特征维度之和
        self.output_dim = feature_dim * num_modalities

    def forward(self, image_feature, vector_features: list):
        # 将图像特征和所有向量特征放入一个列表
        all_features = [image_feature] + vector_features
        # 在最后一个维度（特征维度）上进行拼接
        return torch.cat(all_features, dim=1)


class AOMambaFusion(nn.Module):
    """
    自适应光学 Mamba 融合模块.
    """
    def __init__(self, feature_dim=256, num_vectors=2, use_film=True,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("Local mamba_minimal.py not found or failed to import. Cannot use AOMambaFusion.")
        
        self.feature_dim = feature_dim
        self.use_film = use_film
        
        # 如果使用 FiLM，需要一个 FiLM 层
        if self.use_film:
            self.film_layer = FiLMLayer(modulator_dim=feature_dim * num_vectors, feature_dim=feature_dim)

        # --- 核心修改 2：使用 ModelArgs 和 ResidualBlock 来构建 Mamba 层 ---
        # 1. 创建 Mamba 参数对象
        mamba_args = ModelArgs(
            d_model=feature_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            n_layer=1,      # 我们只需要一个 Mamba 块
            vocab_size=-1   # 在我们的场景下不需要
        )

        # 2. 使用 ResidualBlock 来包装 MambaBlock (这是 mamba-minimal 的推荐用法)
        self.mamba_block = ResidualBlock(mamba_args)
        
        # 为每个模态创建一个可学习的位置/类型嵌入
        num_modalities = 1 + num_vectors
        self.modality_embeddings = nn.Parameter(torch.randn(1, num_modalities, feature_dim))
        
        # 最终的输出维度
        self.output_dim = feature_dim

    def forward(self, image_feature, vector_features: list):
        # image_feature: (B, D)
        # vector_features: [(B, D), (B, D), ...]
        
        # 1. (可选) FiLM 调制
        if self.use_film:
            modulator = torch.cat(vector_features, dim=1)
            image_feature = self.film_layer(image_feature, modulator)

        # 2. 构建模态序列
        modal_sequence = torch.stack([image_feature] + vector_features, dim=1)
        
        # 3. 添加模态嵌入
        modal_sequence += self.modality_embeddings
        
        # 4. Mamba 深度融合 (调用 ResidualBlock)
        fused_sequence = self.mamba_block(modal_sequence)
        
        # 5. 聚合输出
        fused_feature = fused_sequence[:, 0, :]
        
        return fused_feature