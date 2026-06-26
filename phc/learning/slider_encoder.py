"""
StyleSlider — Style Encoder (slider modality)
=============================================
Phase 1 W3 第一刀。对应 docs/07 §1.3 / docs/08 D10/D12。

作用：把 6 维 style slider（W3 离散 {0,1} / W5 连续 [0,1]）映射到 32-dim style latent z_style。
z_style 三条去路（见 docs/11 §3.0 框架图）：
  1. concat 进 obs（W3 先验证）/ FiLM 注入 base hidden（W6）
  2. residual head 的输入
  3. conditional AMP 判别器的条件输入（W4）

设计要点：
  - 输入维度 6（D12: energy/mood/step_width/gait_type/smoothness/arm_swing）
  - 输出维度 32（D10 统一 style latent）
  - 小 MLP，参数量 << base（base=[1024,512] ≈ 1.5M；这里 ~4K，<0.3%）
  - orthogonal init（PPO 常用，与 rl_games 默认风格一致；base 用 const mu_init，但那是 mu head 不是 encoder）
  - W5 连续化时训练时对输入 slider 加噪扰动，让 latent 空间连续可插值

注意（本地草案 → 服务器落地）：
  - 本地 ~/PHC 是 ZhengyiLuo 原版，无服务器上的 ipdb/import 修复（phc_reproduction_notes §1 E3/E4）。
    此文件是独立新模块，不碰 PHC 原文件，落地时直接 cp 到 phc/learning/ 即可。
  - device 跟随调用方（env/policy 在 self.device），本模块不持 device，forward 时输入已在正确 device。
"""
import torch
import torch.nn as nn


class SliderEncoder(nn.Module):
    """6-dim slider → 32-dim style latent.

    Args:
        slider_dim: 输入 slider 维度，默认 6（D12）。
        latent_dim: 输出 latent 维度，默认 32（D10）。
        hidden_dims: encoder MLP 隐藏层，默认 [32, 32]（小网络，防过拟合 6→32 映射）。
        noise_std: W5 训练时输入加噪 std（0=关闭，W3 离散阶段保持 0）。
    """

    def __init__(self, slider_dim: int = 6, latent_dim: int = 32,
                 hidden_dims=(32, 32), noise_std: float = 0.0):
        super().__init__()
        self.slider_dim = slider_dim
        self.latent_dim = latent_dim
        self.noise_std = noise_std

        layers = []
        in_dim = slider_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        layers += [nn.Linear(in_dim, latent_dim)]
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """orthogonal init（PPO 常用）。末层 gain=0.01 让初始 z_style 接近 0，启动稳。"""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # 末层小 gain：初始 z_style ≈ 0，冻结 base 不受 style 干扰，W3 验证 pipeline 通
        last = self.mlp[-1]
        nn.init.orthogonal_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)

    def forward(self, slider: torch.Tensor) -> torch.Tensor:
        """slider: (B, slider_dim) 或 (num_envs, slider_dim) → z_style (B, latent_dim).

        W5 训练时若 noise_std>0，对输入加高斯噪扰动（仅 train 模式），让 latent 连续可插值。
        W3 离散阶段 noise_std=0，输入是 {0,1} one-hot-ish 标签。
        """
        if self.training and self.noise_std > 0:
            slider = slider + torch.randn_like(slider) * self.noise_std
        return self.mlp(slider)


# ============================================================================
# 落地集成备忘（W3 在服务器 /root/PHC 执行）
# ============================================================================
# 1. cp 本文件到 phc/learning/slider_encoder.py
# 2. 每个 env 分配一个风格标签（从 data/style_labels.csv 采样的 S1/S2/S3 clip）：
#    env 初始化时存 self.style_label (num_envs, 6)，W3 先全部用同一风格或随机验证 pipeline。
# 3. 每个 step: z_style = encoder(self.style_label)  # (num_envs, 32)
# 4. z_style 三条去路：
#    a) concat 进 obs: humanoid_im.py:694 _compute_observations 拼 z_style（见 patches/obs_inject.md）
#    b) residual head: 见 policy/style_residual.py
#    c) conditional AMP: W4 amp_network_*_builder.py 加 z_style 条件输入
# 5. encoder 参数随 PPO 一起训（base 冻结，encoder+residual+FiLM 可训）。
# ============================================================================
