"""
StyleSlider — Style Residual Head（冻结 base + bounded residual）
==================================================================
Phase 1 W3。对应 master_plan §0/§3.3（Bounded Style Adaptation for Frozen PHC）、
docs/21 §3（bounded controller-deviation）。

作用：在冻结的 PHC **单 primitive base**（从 phc_3 提取的 P(1)，见
extract_p1_from_phc3.py）输出 mu 上叠加 **bounded** style-conditional residual，
让风格能影响动作，而不改动 base 权重（保内容保真 + 消融变量隔离）。

公式（master_plan §3.3，doc 21 §3）：
    action = frozen_base(s_t) + α · tanh( style_residual(s_t, z_style) )
    - frozen_base = 提取的单 primitive（actor_mlp + mu），参数 requires_grad=False
    - style_residual = 小 MLP，输入 (obs, z_style)，输出 action 维度残差
    - tanh 对 residual 施加硬界 ∈[-1,1]：bounded 是**结构保证**，非仅 α 软约束
      （doc 21 §3 关键设计，挡"内容保真/稳定性"叙事的结构级武器）
    - α 可学习，init=0：启动时 residual 项=0，action=纯 base，训练稳

重定位后的注入点（与旧版不同！旧版写 MCP composer，现已废弃）：
    单 policy 版 amp_network_builder.py eval_actor，is_continuous 分支：
      非 RNN: line ~142  mu = self.mu_act(self.mu(a_out))
      RNN   : line ~117  mu = self.mu_act(self.mu(a_out))
    → 在 self.mu(a_out) 出 mu 后、return 前叠加 bounded residual。

架构事实（已核对 AAA/phc 代码）：
    - 重定位后 base = 提取的单 primitive（标准 amp 网络，AMPNetworkBuilder）
    - eval_actor (amp_network_builder.py:57):
        obs → actor_cnn(空) → actor_mlp → mu = self.mu(a_out)  (is_continuous)
    - mu 经 _action_to_pd_targets (humanoid_im.py:1094) → pd_tar = ref_dof_pos + scale·action
      residual 叠加在 mu 上后经 _pd_action_scale 缩放进 PD target，不改 _action_to_pd_targets。
    - 所以 residual 直接叠加在 mu 上：mu_style = mu + α·tanh(residual)

注意：
    - z_style 来源 = env.z_style_buf（obs_inject.md §3.1），通过 obs_dict 传入 policy
    - z_style **不进 obs**（保 945 维，base 第一层权重可加载冻结）——方案 B
    - 本文件只定义 StyleResidual module；接入单 policy builder 的胶水代码见底部备忘
    - 冻结 base：actor_mlp / mu 全部 requires_grad=False
"""
import torch
import torch.nn as nn


class StyleResidual(nn.Module):
    """Style-conditional **bounded** action residual.

    输入 obs(self_obs+task_obs, 945) + z_style(32) → residual(action_dim, 69)。
    residual 经 tanh 硬界后再乘 α，叠加到 base mu 上。
    小 MLP，参数量 < base 5%（§8.1 约束）。

    Args:
        obs_dim: base 输入 obs 维度（945 = self_obs + task_obs）。
        z_style_dim: style latent 维度（32, D10）。
        action_dim: base 输出 action 维度（69, SMPL humanoid dof）。
        hidden_dim: residual MLP 隐藏维（128，远小于 base [1024,512] 或 [2048,...,512]）。
    """

    def __init__(self, obs_dim: int = 945, z_style_dim: int = 32,
                 action_dim: int = 69, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_style_dim = z_style_dim
        self.action_dim = action_dim

        # (obs, z_style) 拼接 → residual（未界）
        in_dim = obs_dim + z_style_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self._init_weights()
        # α 可学习，init=0：启动 residual 项=0，action=纯 base，不破坏 base 行为
        self.style_alpha = nn.Parameter(torch.zeros(1))

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1e-3)  # 小 init，residual 启动接近 0
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
        """obs: (B, 945), z_style: (B, 32) → 未界 residual: (B, 69).

        注意：返回的是 tanh **前**的 residual；tanh 界在调用处（eval_actor）施加，
        与 α 一起：mu_style = mu + α·tanh(residual)。这样 ablation "unbounded"
        版只需在调用处去掉 tanh 即可（doc 21 §7.2 Unbounded residual 对照）。
        """
        x = torch.cat([obs, z_style], dim=-1)
        return self.mlp(x)


# ============================================================================
# 落地集成备忘（W3 服务器 /root/PHC，单 policy 版）
# ============================================================================
# 0. 前置：先用 extract_p1_from_phc3.py 提取单 primitive 作 base，配 im_big/im config。
#
# 1. cp 本文件到 phc/learning/style_residual.py
#
# 2. 改 phc/learning/amp_network_builder.py 的 Network.__init__（单 policy 版，非 MCP），加：
#    self.style_enabled = True
#    self.style_residual = StyleResidual(
#        obs_dim=self.self_obs_size + self.task_obs_size,  # 945
#        z_style_dim=32, action_dim=action_dim)            # action_dim=69
#    self.style_alpha = self.style_residual.style_alpha    # 暴露给 optimizer
#    # 冻结 base（actor_mlp + mu）：
#    for p in self.actor_mlp.parameters(): p.requires_grad_(False)
#    for p in self.mu.parameters():         p.requires_grad_(False)
#
# 3. 改 eval_actor (amp_network_builder.py:57)，is_continuous 分支叠加 bounded residual：
#    # 非 RNN 分支 (~line 142):
#    if self.is_continuous:
#        mu = self.mu_act(self.mu(a_out))
#        if self.style_enabled:
#            z_style = obs_dict['z_style']                       # env.z_style_buf 经 runner 传入
#            residual = self.style_residual(obs, z_style)        # 未界 residual (B,69)
#            mu = mu + self.style_alpha * torch.tanh(residual)   # ★ bounded: tanh 硬界
#        if self.space_config['fixed_sigma']:
#            sigma = mu * 0.0 + self.sigma_act(self.sigma)
#        else:
#            sigma = self.sigma_act(self.sigma(a_out))
#        return mu, sigma
#    # RNN 分支 (~line 117) 同理在 mu = self.mu_act(self.mu(a_out)) 后叠加。
#    #
#    # ★ ablation "Unbounded residual" 对照 (doc 21 §7.2)：去掉 torch.tanh 即可。
#
# 4. z_style 传入路径：env.z_style_buf → runner obs_dict['z_style'] → eval_actor
#    需改 im_amp_players.py / im_amp.py 把 z_style_buf 加进 obs_dict（与 obs 同步每步更新）。
#    humanoid_im.py:694 _compute_observations 末尾更新 z_style_buf（不进 obs，独立 buffer）。
#
# 5. optimizer：style_residual + style_alpha + SliderEncoder 进 rl_games optimizer
#    （base 冻结 requires_grad=False 自动排除）。reward 含 −λ_res·‖Δa‖² 惩罚
#    （master_plan 抉择2），Δa = α·tanh(residual)，配合 tanh 硬界双保险防 adapter 过偏。
# ============================================================================
