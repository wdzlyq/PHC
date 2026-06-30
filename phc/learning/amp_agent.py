from phc.utils.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgym.torch_utils import *

import time
import os
from datetime import datetime
import numpy as np
from torch import optim
import torch
from torch import nn
from phc.env.tasks.humanoid_amp_task import HumanoidAMPTask

import learning.replay_buffer as replay_buffer
import learning.common_agent as common_agent

from tensorboardX import SummaryWriter
import copy
from phc.utils.torch_utils import project_to_norm
import learning.amp_datasets as amp_datasets
from phc.learning.loss_functions import kl_multi
from smpl_sim.utils.math_utils import LinearAnneal

def load_my_state_dict(target, saved_dict):
    for name, param in saved_dict.items():
        if name not in target:
            continue

        if target[name].shape == param.shape:
            target[name].copy_(param)


# === AAA Step C3: local response proxy MLP（必须与 code/scripts/train_g_local.py 一致）===
# 为什么：C2 训好的 g_local(obs, action_delta)->Δstep_width_at_gait_event 要在 agent 侧当 dense
#   surrogate 冻结加载，梯度经 action_delta 回到 structured scalar head/slider_encoder（plan §680-700）。
#   架构必须与 train_g_local.LocalResponseMLP 逐键匹配，否则 load_state_dict 失败。
# 做什么：复制 train_g_local.py 的 LocalResponseMLP 定义；hidden 由 ckpt['config']['hidden'] 决定。
class AAALocalResponseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
# === AAA end ===


class AMPAgent(common_agent.CommonAgent):

    def __init__(self, base_name, config):
        super().__init__(base_name, config)


        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.ppo_device)  # Override and get new value

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        norm_disc_reward = config.get('norm_disc_reward', False)
        if (norm_disc_reward):
            self._disc_reward_mean_std = RunningMeanStd((1,)).to(self.ppo_device)
        else:
            self._disc_reward_mean_std = None

        self.temp_running_mean = self.vec_env.env.task.temp_running_mean # use temp running mean to make sure the obs used for training is the same as calc gradient.

        # === AAA W5: step_width canary loss switches (default off, zero-regression) ===
        # 为什么：C+ reward-shaping-only 已证实 style reward 能计算但传不到 adapter；下一轮需要把
        #   style reward 从总 reward 拆出独立 advantage，并用 residual contrast loss 显式打开
        #   slider→style_residual 信息通道。默认全 0，保持旧 W4/W5 训练行为不变。
        # 做什么：读取 loss 权重与 contrast 设置；实际 loss 在 calc_gradients gated 加入。
        self._aaa_style_adv_coef = float(config.get('aaa_style_adv_coef', 0.0))
        self._aaa_contrast_coef = float(config.get('aaa_contrast_coef', 0.0))
        self._aaa_contrast_delta_coef = float(config.get('aaa_contrast_delta_coef', 0.0))
        self._aaa_contrast_kl_coef = float(config.get('aaa_contrast_kl_coef', 0.0))
        self._aaa_contrast_alpha_floor = float(config.get('aaa_contrast_alpha_floor', 0.1))
        self._aaa_contrast_dim = int(config.get('aaa_contrast_dim', 1))  # step_width dim
        # === AAA W5 P2: learned differentiable step_width proxy (default off) ===
        # 为什么：EMA canary 证明 reward 量纲已修，但真实 sweep 仍 flat；rollout reward→GAE→PPO
        #   不能把 step_width 方向稳定写进 residual。P2 用 g(obs, action)->step_width_norm 监督拟合
        #   在线物理指标，再用 g 的可微方向 loss 直通 style_residual。
        # 做什么：proxy 自带 optimizer 仅训练 proxy；direction loss 冻结 proxy 权重，只让梯度流向
        #   policy/action/residual。所有 coef 默认 0，零回归。
        self._aaa_proxy_train_coef = float(config.get('aaa_proxy_train_coef', 0.0))
        self._aaa_proxy_dir_coef = float(config.get('aaa_proxy_dir_coef', 0.0))
        self._aaa_proxy_lr = float(config.get('aaa_proxy_lr', 1e-3))
        self._aaa_proxy_hidden = int(config.get('aaa_proxy_hidden', 256))
        self._aaa_proxy_margin = float(config.get('aaa_proxy_margin', 0.01))
        self._aaa_proxy_grad_clip = float(config.get('aaa_proxy_grad_clip', 1.0))
        self._aaa_proxy_dir_warmup_epochs = int(config.get('aaa_proxy_dir_warmup_epochs', 0))
        self._aaa_step_proxy = None
        self._aaa_step_proxy_opt = None
        # === AAA W5 P2b: analytic/action-space step_width proxy (default off) ===
        # 为什么：learned g(obs, action) 几乎不看 action；先用最小可控的髋部动作差异代理 step_width。
        # SMPL actuator order: L_Hip_x/y/z=0/1/2, R_Hip_x/y/z=12/13/14。
        self._aaa_joint_step_coef = float(config.get('aaa_joint_step_coef', 0.0))
        self._aaa_joint_step_margin = float(config.get('aaa_joint_step_margin', 0.05))
        self._aaa_joint_step_left_idx = int(config.get('aaa_joint_step_left_idx', 0))
        self._aaa_joint_step_right_idx = int(config.get('aaa_joint_step_right_idx', 12))
        self._aaa_joint_step_left_sign = float(config.get('aaa_joint_step_left_sign', 1.0))
        self._aaa_joint_step_right_sign = float(config.get('aaa_joint_step_right_sign', -1.0))
        # === AAA Step C: structured scalar head loss（默认关闭，零回归）===
        # 为什么：Step B 证明固定 D2_axis0 动作方向可控；Step C 只训练一个 scalar head 读 slider，
        #   再投到该方向，避免重新打开无结构 69D residual 自由度。
        self._aaa_structured_step_coef = float(config.get('aaa_structured_step_coef', 0.0))
        self._aaa_structured_step_margin = float(config.get('aaa_structured_step_margin', 0.05))
        self._aaa_structured_step_amp = float(config.get('aaa_structured_step_amp', 0.1))
        # === AAA Step C3: g_local dense surrogate（默认关闭，零回归）===
        # 为什么：C1 structured_step_loss 用的是 action-projection margin（标量内积），只约束方向维
        #   投影差；C3 改用 C2 训好的 g_local(obs, delta)->Δstep_width_event 作 dense differentiable
        #   surrogate，把"高 slider 的 delta 应导致更大真实 step_width"这一物理响应直接写成可微 loss，
        #   梯度经 delta_action 回到 scalar head，不靠 PPO advantage（plan §486-498, §667-679）。
        # 做什么：读 g_local cfg；模型在 _aaa_ensure_g_local 懒加载（首次 train_minibatch），
        #   统计量（obs_mean/std、delta_scale、y_mean/std）随模型一起冻结，requires_grad=False。
        self._aaa_g_local_coef = float(config.get('aaa_g_local_coef', 0.0))
        self._aaa_g_local_margin = float(config.get('aaa_g_local_margin', 0.03))
        self._aaa_g_local_path = str(config.get('aaa_g_local_path', ''))
        self._aaa_g_local_feature_mode = str(config.get('aaa_g_local_feature_mode', 'obs_delta'))
        self._aaa_g_local_grad_clip = float(config.get('aaa_g_local_grad_clip', 0.0))
        self._aaa_g_local = None          # 冻结的 AAALocalResponseMLP，懒加载
        self._aaa_g_local_stats = None    # dict: obs_mean/std/delta_scale/y_mean/y_std/delta_start/...
        # === AAA end ===
        # === AAA 诊断C: 梯度审计计数器（gated by AAA_GRAD_AUDIT env var，默认关零回归）===
        self._aaa_grad_audit_count = 0
        # === AAA end ===

        kin_lr = float(self.vec_env.env.task.kin_lr)
        
        # ZL Hack
        if self.vec_env.env.task.fitting:
            print("#################### Fitting and freezing!! ####################")
            # checkpoint = torch_ext.load_checkpoint(self.vec_env.env.task.models_path[0])
            # self.set_stats_weights(checkpoint)  # loads mean std. essential for distilling knowledge. will not load if has a shape mismatch.
            self.freeze_state_weights()  # freeze the mean stds.
            # load_my_state_dict(self.model.state_dict(), checkpoint['model'])  # loads everything (model, std, ect.). that can be load from the last model.
            # self.value_mean_std # not freezing value function though.
        
        return
    
    def set_stats_weights(self, weights):
        if self.normalize_input:
            if weights['running_mean_std']['running_mean'].shape == self.running_mean_std.state_dict()['running_mean'].shape:
                self.running_mean_std.load_state_dict(weights['running_mean_std'])
            else:
                print("shape mismatch, can not load input mean std")
                
        if self.normalize_value:
            self.value_mean_std.load_state_dict(weights['reward_mean_std'])

        if self.has_central_value:
            self.central_value_net.set_stats_weights(weights['assymetric_vf_mean_std'])
 
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])
            
        if self._normalize_amp_input:
            if weights['amp_input_mean_std']['running_mean'].shape == self._amp_input_mean_std.state_dict()['running_mean'].shape:
                self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])
            else:
                print("shape mismatch, can not load AMP mean std")
            

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.load_state_dict(weights['disc_reward_mean_std'])
            
    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        
        if "kin_optimizer" in self.__dict__:
            print("!!!saving kin_optimizer!!! Remove this message asa p!!")
            state['kin_optimizer'] = self.kin_optimizer.state_dict()
        if self._aaa_step_proxy is not None:
            state['aaa_step_proxy'] = self._aaa_step_proxy.state_dict()
            state['aaa_step_proxy_opt'] = self._aaa_step_proxy_opt.state_dict()

        return state

    def set_full_state_weights(self, weights):
        if getattr(self.model.a2c_network, 'style_enabled', False):
            # === AAA W3: 从提取的单 primitive base ckpt 初始化（非 resume）===
            # model: 容错加载（base 键匹配加载；style 模块 slider_encoder/style_residual/
            #   style_alpha/aaa_style_label 保留 init，因 ckpt 无这些键）。
            # optimizer: 跳过（ckpt optimizer 是 phc_3 全 pnn 结构，与新 model 不匹配），
            #   用 fresh optimizer 只训 style（base requires_grad=False → grad None → Adam skip）。
            # epoch/frame: 重置 0（fresh training）。
            load_my_state_dict(self.model.state_dict(), weights['model'])
            self.set_stats_weights(weights)
            self.epoch_num = 0
            self.frame = 0
            self.last_mean_rewards = -100500
            print("[AAA W3] init base from ckpt (tolerant load), fresh optimizer, epoch=0")
            return
        super().set_full_state_weights(weights)
        if "kin_optimizer" in weights:
            print("!!!loading kin_optimizer!!! Remove this message asa p!!")
            self.kin_optimizer.load_state_dict(weights['kin_optimizer'])
        

    def freeze_state_weights(self):
        if self.normalize_input:
            self.running_mean_std.freeze()
        if self.normalize_value:
            self.value_mean_std.freeze()
        if self.has_central_value:
            raise NotImplementedError()
        if self.mixed_precision:
            raise NotImplementedError()

    def unfreeze_state_weights(self):
        if self.normalize_input:
            self.running_mean_std.unfreeze()
        if self.normalize_value:
            self.value_mean_std.unfreeze()
        if self.has_central_value:
            raise NotImplementedError()
        if self.mixed_precision:
            raise NotImplementedError()

    def init_tensors(self):
        super().init_tensors()
        self._build_amp_buffers()

            
        return

    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.eval()

        return

    def set_train(self):
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.train()

        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state['amp_input_mean_std'] = self._amp_input_mean_std.state_dict()

        if (self._norm_disc_reward()):
            state['disc_reward_mean_std'] = self._disc_reward_mean_std.state_dict()

        return state

    def get_action_values(self, obs):
        # === AAA W4: 注入当前 slider 给 policy eval_actor（style_residual 条件化，w4 patch 补漏）===
        # 为什么：common_agent.get_action_values 的 input_dict 不透传 obs 额外 key，policy eval_actor
        #   拿不到 slider → style_residual 用固定 style 0 → policy 学不到风格（W4 核心）。
        #   故 override 此处注入。slider 来源 = env.style_labels（reset 时已按 motion 更新，
        #   play_steps/play_steps_rnn 调用时即是当前步风格，与 infos['slider'] 同源）。
        obs_orig = obs['obs']
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            "obs_orig": obs_orig,
            'rnn_states': self.rnn_states
        }
        if getattr(self.model.a2c_network, 'style_enabled', False):
            input_dict['slider'] = self.vec_env.env.task.style_labels  # (num_envs, 6)
        # === AAA end ===
        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                cv_input = {
                    'is_train': False,
                    'states': states,
                }
                value = self.get_central_value(cv_input)
                res_dict['values'] = value
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    # === AAA W5 P2: learned differentiable step_width proxy helpers ===
    def _aaa_ensure_step_proxy(self, obs_dim, action_dim):
        if self._aaa_step_proxy is not None:
            return
        self._aaa_step_proxy = nn.Sequential(
            nn.Linear(obs_dim + action_dim, self._aaa_proxy_hidden),
            nn.ELU(),
            nn.Linear(self._aaa_proxy_hidden, self._aaa_proxy_hidden),
            nn.ELU(),
            nn.Linear(self._aaa_proxy_hidden, 1),
        ).to(self.ppo_device)
        self._aaa_step_proxy_opt = optim.Adam(self._aaa_step_proxy.parameters(), lr=self._aaa_proxy_lr)

    def _aaa_step_proxy_forward(self, obs, action):
        self._aaa_ensure_step_proxy(obs.shape[-1], action.shape[-1])
        return self._aaa_step_proxy(torch.cat([obs, action], dim=-1))

    def _aaa_train_step_proxy(self, obs, action, target):
        if self._aaa_proxy_train_coef <= 0.0:
            return torch.zeros((), device=self.ppo_device), torch.zeros((), device=self.ppo_device)
        valid = torch.isfinite(target).squeeze(-1)
        if valid.sum() == 0:
            return torch.zeros((), device=self.ppo_device), torch.zeros((), device=self.ppo_device)
        pred = self._aaa_step_proxy_forward(obs.detach()[valid], action.detach()[valid])
        tgt = target.detach()[valid].clamp(0.0, 1.0)
        loss = torch.mean((pred - tgt) ** 2)
        self._aaa_step_proxy_opt.zero_grad()
        loss.backward()
        if self._aaa_proxy_grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self._aaa_step_proxy.parameters(), self._aaa_proxy_grad_clip)
        self._aaa_step_proxy_opt.step()
        mae = torch.mean(torch.abs(pred.detach() - tgt))
        return loss.detach(), mae.detach()
    # === AAA end ===

    # === AAA Step C3: g_local 懒加载 + forward ===
    def _aaa_ensure_g_local(self, obs_dim, action_dim):
        # 为什么：g_local 模型依赖 ckpt 内统计量与 hidden，需在首次拿到 obs/action 维度后加载；
        #   放 __init__ 时机太早（ppo_device/网络维度未定）。懒加载 + 一次校验，避免每个 minibatch 重读。
        # 做什么：读 ckpt -> 建 AAALocalResponseMLP(ckpt['config']['hidden']) -> load_state_dict ->
        #   冻结(eval+requires_grad=False) -> 缓存 stats 张量到 ppo_device。校验 feature_mode/维度一致。
        if self._aaa_g_local is not None:
            return
        if self._aaa_g_local_coef <= 0.0 or not self._aaa_g_local_path:
            return
        import torch as _torch
        _path = self._aaa_g_local_path
        if not os.path.exists(_path):
            raise FileNotFoundError(f"[AAA g_local] ckpt not found: {_path}")
        _ckpt = _torch.load(_path, map_location=self.ppo_device)
        _stats = _ckpt['stats']
        _cfg = _ckpt.get('config', {})
        _hidden = int(_cfg.get('hidden', 256))
        _feat_mode = str(_stats.get('feature_mode', 'obs_delta'))
        if _feat_mode != self._aaa_g_local_feature_mode:
            raise ValueError(
                f"[AAA g_local] feature_mode mismatch: cfg={self._aaa_g_local_feature_mode} "
                f"but ckpt={_feat_mode}")
        if int(_stats.get('obs_dim', -1)) != int(obs_dim):
            raise ValueError(
                f"[AAA g_local] obs_dim mismatch: cfg/env={obs_dim} but ckpt={_stats.get('obs_dim')}")
        if int(_stats.get('action_dim', -1)) != int(action_dim):
            raise ValueError(
                f"[AAA g_local] action_dim mismatch: env={action_dim} but ckpt={_stats.get('action_dim')}")
        _model = AAALocalResponseMLP(int(_stats['input_dim']), hidden=_hidden).to(self.ppo_device)
        _model.load_state_dict(_ckpt['model_state'])
        _model.eval()
        for _p in _model.parameters():
            _p.requires_grad_(False)
        self._aaa_g_local = _model
        # 缓存统计量到 ppo_device（非 nn.Buffer，因 g_local 不进 self.model；随 agent 生命周期常驻）
        self._aaa_g_local_stats = {
            'obs_mean': _torch.as_tensor(_stats['obs_mean'], device=self.ppo_device).float(),
            'obs_std': _torch.as_tensor(_stats['obs_std'], device=self.ppo_device).float(),
            'delta_scale': float(_stats['delta_scale']),
            'y_mean': float(_stats['y_mean']),
            'y_std': float(_stats['y_std']),
            'delta_start': int(_stats.get('delta_start', int(_stats['obs_dim']))),
            'obs_dim': int(_stats['obs_dim']),
            'action_dim': int(_stats['action_dim']),
        }
        print(f"[AAA g_local] loaded frozen g_local from {_path} "
              f"(feature_mode={_feat_mode}, input_dim={int(_stats['input_dim'])}, "
              f"hidden={_hidden}, best_epoch={_ckpt.get('best_epoch')})", flush=True)

    def _aaa_g_local_forward(self, obs, delta_action):
        # 为什么：g_local 输入是 (obs_feat, delta_feat)（obs_delta 模式），输出归一化 Δstep_width，
        #   须反标准化回真实单位做 margin 比较（plan §695）。梯度穿过 delta_action 回 scalar head。
        # 做什么：标准化 -> forward -> 反标准化；obs 用 obs_mean/std，delta 用 delta_scale。
        _s = self._aaa_g_local_stats
        _obs_feat = (obs - _s['obs_mean']) / _s['obs_std']
        _delta_feat = delta_action / _s['delta_scale']
        if self._aaa_g_local_feature_mode == 'obs_delta':
            _x = torch.cat([_obs_feat, _delta_feat], dim=-1)
        elif self._aaa_g_local_feature_mode == 'delta_phase':
            # C2 delta_phase 模式无 obs；agent 侧未采集 phase，此处不应走到（cfg 选 obs_delta）。
            # 保留分支仅为对称；若启用需补 phase 特征。
            _x = _delta_feat
        else:  # full
            _x = torch.cat([_obs_feat, _delta_feat], dim=-1)  # phase 缺省，与 ckpt 不匹配会报错
        _pred_norm = self._aaa_g_local(_x)
        return _pred_norm * _s['y_std'] + _s['y_mean']
    # === AAA end ===

    def play_steps_rnn(self):
        self.set_eval()
        mb_rnn_states = []
        epinfos = []
        self.experience_buffer.tensor_dict['values'].fill_(0)
        self.experience_buffer.tensor_dict['rewards'].fill_(0)
        self.experience_buffer.tensor_dict['dones'].fill_(1)
        step_time = 0.0

        update_list = self.update_list

        batch_size = self.num_agents * self.num_actors
        mb_rnn_masks = None

        mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states = self.init_rnn_step(batch_size, mb_rnn_states) # mb_rnn_states means "memory bank" rnn states

        ### ZL
        done_indices = []
        terminated_flags = torch.zeros(self.num_actors, device=self.device)
        reward_raw = torch.zeros(1, device=self.device)

        for n in range(self.horizon_length):
            
            
            
            self.obs = self.env_reset(done_indices)
            
            # self.rnn_states[0][:, :, -1] = n; print('debugg!!!!')
            # self.rnn_states[0][:, :, -2] = torch.arange(self.num_actors)
            
            seq_indices, full_tensor = self.process_rnn_indices(mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states)  # this should upate mb_rnn_states
            if full_tensor:
                break
            
            if self.has_central_value:
                self.central_value_net.pre_step_rnn(self.last_rnn_indices, self.last_state_indices)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            
            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data_rnn('obses', indices, play_mask, self.obs['obs'])

            for k in update_list:
                self.experience_buffer.update_data_rnn(k, indices, play_mask, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data_rnn('states', indices[::self.num_agents], play_mask[::self.num_agents] // self.num_agents, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            
                
            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()
            self.experience_buffer.update_data_rnn('rewards', indices, play_mask, shaped_rewards)
            self.experience_buffer.update_data_rnn('next_obses', indices, play_mask, self.obs['obs'])
            self.experience_buffer.update_data_rnn('dones', indices, play_mask, self.dones.byte())
            self.experience_buffer.update_data_rnn('amp_obs', indices, play_mask, infos['amp_obs'])
            # === AAA W4: slider 同步进 RNN buffer（前向兼容，im_big 非 RNN 不走此路）===
            self.experience_buffer.update_data_rnn('slider', indices, play_mask, infos['slider'])
            # === AAA W5: C+ style reward 单独进 RNN buffer ===
            # 为什么：style-specific PPO advantage 需要 env 侧 metric-anchored r_style 的原始序列；
            #   RNN 路径虽非 im_big 主线，也保持 buffer 字段一致。
            _aaa_style_reward = infos.get('aaa_cplus_style_reward', None)
            if _aaa_style_reward is None:
                _aaa_style_reward = torch.zeros_like(rewards)
            if _aaa_style_reward.dim() == 1:
                _aaa_style_reward = _aaa_style_reward.unsqueeze(-1)
            self.experience_buffer.update_data_rnn('style_reward', indices, play_mask, _aaa_style_reward)
            _aaa_step_norm = infos.get('aaa_cplus_step_width_norm', None)
            if _aaa_step_norm is None:
                _aaa_step_norm = torch.full_like(rewards, float('nan'))
            if _aaa_step_norm.dim() == 1:
                _aaa_step_norm = _aaa_step_norm.unsqueeze(-1)
            self.experience_buffer.update_data_rnn('step_width_norm', indices, play_mask, _aaa_step_norm)
            # === AAA end ===
            # === AAA W4: residual norm 惩罚 ‖Δa‖²（RNN 前向兼容）===
            _aaa_delta = self.model.a2c_network._last_delta_a
            _aaa_res_pen = (_aaa_delta ** 2).sum(dim=-1, keepdim=True)
            self.experience_buffer.update_data_rnn('res_penalty', indices, play_mask, _aaa_res_pen)
            # === AAA end ===

            ### ZL
            terminated = infos['terminate'].float()
            terminated_flags += terminated
            reward_raw_mean = infos['reward_raw'].mean(dim=0)

            if reward_raw.shape != reward_raw_mean.shape:
                reward_raw = reward_raw_mean
            else:
                reward_raw += reward_raw_mean

            terminated = terminated.unsqueeze(-1)
            input_dict = {"obs": self.obs['obs'], "rnn_states": self.rnn_states}
            next_vals = self._eval_critic(input_dict)  # ZL this has issues? (maybe not, since we are passing the states in.)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data_rnn('next_values', indices, play_mask, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.process_rnn_dones(all_done_indices, indices, seq_indices)

            if self.has_central_value:
                self.central_value_net.post_step_rnn(all_done_indices)

            self.algo_observer.process_infos(infos, done_indices)

            fdones = self.dones.float()
            not_dones = 1.0 - self.dones.float()

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)

            done_indices = done_indices[:, 0]
            

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_style_rewards = self.experience_buffer.tensor_dict['style_reward']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        mb_slider = self.experience_buffer.tensor_dict['slider']  # AAA W4: 取 slider 喂 conditional disc（reward 路径）
        mb_res_pen = self.experience_buffer.tensor_dict['res_penalty']  # AAA W4: ‖Δa‖²（residual norm 惩罚）
        amp_rewards = self._calc_amp_rewards(mb_amp_obs, mb_slider)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        # === AAA probe: 记录三分量 reward 均值（gated by AAA_PROBE 环境变量，零回归）===
        # 为什么：W4 第二步诊断 α 卡住，需看 content/disc/respen 三力各自量级 + α 净梯度方向。
        # 做什么：combine 后 original task reward 已被覆盖，从 buffer 取；disc/respen 仍在作用域。
        if os.environ.get('AAA_PROBE'):
            self._probe_r_content = float(self.experience_buffer.tensor_dict['rewards'].mean())
            self._probe_r_disc = float(amp_rewards['disc_rewards'].mean())
            self._probe_r_respen = float(mb_res_pen.mean())
        # === AAA end ===
        # === AAA W4: residual norm 惩罚 −λ_res·‖Δa‖²（tanh 硬界双保险，w4 patch §8.3 / 抉择2）===
        # 为什么：bounded 结构的软约束补充——tanh 硬界限 Δa∈[−α,α]，norm 惩罚再加经济成本防 adapter 过偏。
        mb_rewards = mb_rewards - self._res_lambda * mb_res_pen
        # === AAA end ===
        

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values
        # === AAA W5: style-specific advantage（无 critic baseline，默认 coef=0 时只记录不使用）===
        # 为什么：C+ v2 说明 style reward 混在总 reward 中会被 content/disc 稀释；单独 GAE 后
        #   calc_gradients 可用 aaa_style_adv_coef 加一条 style PPO actor loss。
        _aaa_zero_values = torch.zeros_like(mb_values)
        mb_style_advs = self.discount_values(mb_fdones, _aaa_zero_values, mb_style_rewards, _aaa_zero_values)
        # === AAA end ===
        
        # self.experience_buffer.tensor_dict['actions']: is num_env, Batch, feat. That's why we swap and flatten, mb_rnn_states is already in that format. 
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list) # swap to step, num_envs, feat
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['style_advantages'] = a2c_common.swap_and_flatten01(mb_style_advs)
        batch_dict['rnn_states'] = mb_rnn_states
        
        batch_dict['rnn_masks'] = mb_rnn_masks # ZL: this should be swap and flattened, but it's all ones for now
        batch_dict['terminated_flags'] = terminated_flags
        batch_dict['reward_raw'] =reward_raw / self.horizon_length
        
        batch_dict['played_frames'] = n * self.num_actors * self.num_agents
        batch_dict['step_time'] = step_time
        

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        batch_dict['mb_rewards'] = a2c_common.swap_and_flatten01(mb_rewards)
        batch_dict['style_rewards'] = a2c_common.swap_and_flatten01(mb_style_rewards)
        
        return batch_dict

    def play_steps(self):
        self.set_eval()
        humanoid_env = self.vec_env.env.task

        epinfos = []
        done_indices = []
        update_list = self.update_list
        terminated_flags = torch.zeros(self.num_actors, device=self.device)
        reward_raw = torch.zeros(1, device=self.device)
        for n in range(self.horizon_length):

            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
                
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])
            
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
                
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])
            # === AAA W4: slider 同步进 buffer（仿 amp_obs，w4 patch §1.2）===
            self.experience_buffer.update_data('slider', n, infos['slider'])
            # === AAA W5: C+ style reward 单独进 buffer ===
            # 为什么：C+ v2 失败说明 style reward 混进总 reward 后被 content/disc 淹没；
            #   单独保存 env 计算的 metric-anchored r_style，后面可构造独立 style advantage。
            _aaa_style_reward = infos.get('aaa_cplus_style_reward', None)
            if _aaa_style_reward is None:
                _aaa_style_reward = torch.zeros_like(rewards)
            if _aaa_style_reward.dim() == 1:
                _aaa_style_reward = _aaa_style_reward.unsqueeze(-1)
            self.experience_buffer.update_data('style_reward', n, _aaa_style_reward)
            _aaa_step_norm = infos.get('aaa_cplus_step_width_norm', None)
            if _aaa_step_norm is None:
                _aaa_step_norm = torch.full_like(rewards, float('nan'))
            if _aaa_step_norm.dim() == 1:
                _aaa_step_norm = _aaa_step_norm.unsqueeze(-1)
            self.experience_buffer.update_data('step_width_norm', n, _aaa_step_norm)
            # === AAA end ===
            # === AAA W4: residual norm 惩罚 ‖Δa‖²（w4 patch §8.3）===
            # Δa = α·tanh(Δπ) 由 eval_actor 缓存在 _last_delta_a（对应当步 action），这里取算平方和存盘。
            _aaa_delta = self.model.a2c_network._last_delta_a  # (num_envs, 69)
            _aaa_res_pen = (_aaa_delta ** 2).sum(dim=-1, keepdim=True)  # (num_envs, 1)
            self.experience_buffer.update_data('res_penalty', n, _aaa_res_pen)
            # === AAA end ===

                
            terminated = infos['terminate'].float()
            terminated_flags += terminated

            reward_raw_mean = infos['reward_raw'].mean(dim=0)
            if reward_raw.shape != reward_raw_mean.shape:
                reward_raw = reward_raw_mean
            else:
                reward_raw += reward_raw_mean
            terminated = terminated.unsqueeze(-1)

            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)
            
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_style_rewards = self.experience_buffer.tensor_dict['style_reward']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        mb_slider = self.experience_buffer.tensor_dict['slider']  # AAA W4: 取 slider 喂 conditional disc（reward 路径）
        mb_res_pen = self.experience_buffer.tensor_dict['res_penalty']  # AAA W4: ‖Δa‖²（residual norm 惩罚）
        amp_rewards = self._calc_amp_rewards(mb_amp_obs, mb_slider)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        # === AAA probe: 记录三分量 reward 均值（gated by AAA_PROBE 环境变量，零回归）===
        # 为什么：W4 第二步诊断 α 卡住，需看 content/disc/respen 三力各自量级 + α 净梯度方向。
        # 做什么：combine 后 original task reward 已被覆盖，从 buffer 取；disc/respen 仍在作用域。
        if os.environ.get('AAA_PROBE'):
            self._probe_r_content = float(self.experience_buffer.tensor_dict['rewards'].mean())
            self._probe_r_disc = float(amp_rewards['disc_rewards'].mean())
            self._probe_r_respen = float(mb_res_pen.mean())
        # === AAA end ===
        # === AAA W4: residual norm 惩罚 −λ_res·‖Δa‖²（tanh 硬界双保险，w4 patch §8.3 / 抉择2）===
        # 为什么：bounded 结构的软约束补充——tanh 硬界限 Δa∈[−α,α]，norm 惩罚再加经济成本防 adapter 过偏。
        mb_rewards = mb_rewards - self._res_lambda * mb_res_pen
        # === AAA end ===
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values
        # === AAA W5: style-specific advantage（无 critic baseline，默认 coef=0 时只记录不使用）===
        # 为什么：C+ v2 说明 style reward 混在总 reward 中会被 content/disc 稀释；单独 GAE 后
        #   calc_gradients 可用 aaa_style_adv_coef 加一条 style PPO actor loss。
        _aaa_zero_values = torch.zeros_like(mb_values)
        mb_style_advs = self.discount_values(mb_fdones, _aaa_zero_values, mb_style_rewards, _aaa_zero_values)
        # === AAA end ===

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['style_advantages'] = a2c_common.swap_and_flatten01(mb_style_advs)
        batch_dict['terminated_flags'] = terminated_flags
        batch_dict['reward_raw'] =reward_raw / self.horizon_length
        batch_dict['played_frames'] = self.batch_size
        
        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)
        batch_dict['mb_rewards'] = a2c_common.swap_and_flatten01(mb_rewards)
        batch_dict['style_rewards'] = a2c_common.swap_and_flatten01(mb_style_rewards)
        
        return batch_dict

    def prepare_dataset(self, batch_dict):
        
        
        dataset_dict = super().prepare_dataset(batch_dict)
        dataset_dict['amp_obs'] = batch_dict['amp_obs']
        dataset_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        dataset_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']
        # === AAA W4: 三路 slider 进 dataset，供 minibatch 切片喂 conditional disc ===
        dataset_dict['slider'] = batch_dict['slider']                      # agent 分支
        dataset_dict['demo_slider'] = batch_dict['demo_slider']            # demo 分支
        dataset_dict['amp_replay_slider'] = batch_dict['amp_replay_slider']  # replay 分支
        # === AAA W5: style-specific PPO 字段进 dataset ===
        # 为什么：super().prepare_dataset 不保证透传自定义字段；calc_gradients 需要按同一 sample_idx
        #   切出 style_advantages，才能和 action_log_probs 对齐。
        dataset_dict['style_advantages'] = batch_dict['style_advantages']
        dataset_dict['style_rewards'] = batch_dict['style_rewards']
        dataset_dict['step_width_norm'] = batch_dict['step_width_norm']
        # === AAA end ===
        # === AAA end ===

            
        self.dataset.update_values_dict(dataset_dict, rnn_format = True, horizon_length = self.horizon_length, num_envs = self.num_actors)
        # self.dataset.update_values_dict(dataset_dict)

        return

    def train_epoch(self):
        self.pre_epoch(self.epoch_num)
        play_time_start = time.time()

        ### ZL: do not update state weights during play

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self._update_amp_demos()
        num_obs_samples = batch_dict['amp_obs'].shape[0]
        # === AAA W4: demo 取 amp_obs + slider（w4 patch §3）===
        _demo_sampled = self._amp_obs_demo_buffer.sample(num_obs_samples)
        batch_dict['amp_obs_demo'] = _demo_sampled['amp_obs']
        batch_dict['demo_slider'] = _demo_sampled['slider']
        # === AAA end ===

        if (self._amp_replay_buffer.get_total_count() == 0):
            batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
            batch_dict['amp_replay_slider'] = batch_dict['slider']  # AAA W4: 首次无 replay，用当步 slider
        else:
            # === AAA W4: replay 取 amp_obs + slider（保证 replay 分支条件化正确）===
            _replay_sampled = self._amp_replay_buffer.sample(num_obs_samples)
            batch_dict['amp_obs_replay'] = _replay_sampled['amp_obs']
            batch_dict['amp_replay_slider'] = _replay_sampled['slider']
            # === AAA end ===

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        # if self.is_rnn:
        # frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)
            
        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        self._store_replay_amp_obs(batch_dict['amp_obs'], batch_dict['slider'])  # AAA W4: 同步存 slider

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        train_info['terminated_flags'] = batch_dict['terminated_flags']
        train_info['reward_raw'] = batch_dict['reward_raw']
        train_info['mb_rewards'] = batch_dict['mb_rewards']
        train_info['style_rewards'] = batch_dict['style_rewards']
        train_info['returns'] = batch_dict['returns']
        self._record_train_batch_info(batch_dict, train_info)
        self.post_epoch(self.epoch_num)
        # === AAA probe: 每 epoch 打印 α 值 + α 净梯度 + 三分量 reward 均值（gated，零回归）===
        # 为什么：α 卡 -0.026，需确认是哪股力主导。grad=最后 minibatch 的 PPO 净梯度（含三力），
        #   sign 持续为负→被压向 0；≈0→无方向性杠杆。配合 disc-only run 隔离 force 3。
        if os.environ.get('AAA_PROBE'):
            _sa = self.model.a2c_network.style_alpha
            _g = _sa.grad
            _gv = float(_g.item()) if (_g is not None) else float('nan')
            _style_r = float(batch_dict['style_rewards'].mean()) if 'style_rewards' in batch_dict else float('nan')
            _style_loss = torch_ext.mean_list(train_info.get('aaa_style_actor_loss', [torch.tensor(float('nan'))])).item()
            _contrast_loss = torch_ext.mean_list(train_info.get('aaa_contrast_loss', [torch.tensor(float('nan'))])).item()
            _proxy_loss = torch_ext.mean_list(train_info.get('aaa_proxy_train_loss', [torch.tensor(float('nan'))])).item()
            _proxy_mae = torch_ext.mean_list(train_info.get('aaa_proxy_mae', [torch.tensor(float('nan'))])).item()
            _proxy_dir = torch_ext.mean_list(train_info.get('aaa_proxy_dir_loss', [torch.tensor(float('nan'))])).item()
            _proxy_gap = torch_ext.mean_list(train_info.get('aaa_proxy_dir_gap', [torch.tensor(float('nan'))])).item()
            _proxy_ag = torch_ext.mean_list(train_info.get('aaa_proxy_action_grad', [torch.tensor(float('nan'))])).item()
            _proxy_agmax = torch_ext.mean_list(train_info.get('aaa_proxy_action_grad_max', [torch.tensor(float('nan'))])).item()
            _joint_loss = torch_ext.mean_list(train_info.get('aaa_joint_step_loss', [torch.tensor(float('nan'))])).item()
            _joint_gap = torch_ext.mean_list(train_info.get('aaa_joint_step_gap', [torch.tensor(float('nan'))])).item()
            _structured_loss = torch_ext.mean_list(train_info.get('aaa_structured_step_loss', [torch.tensor(float('nan'))])).item()
            _structured_gap = torch_ext.mean_list(train_info.get('aaa_structured_step_gap', [torch.tensor(float('nan'))])).item()
            _glocal_loss = torch_ext.mean_list(train_info.get('aaa_g_local_loss', [torch.tensor(float('nan'))])).item()
            _glocal_gap = torch_ext.mean_list(train_info.get('aaa_g_local_gap', [torch.tensor(float('nan'))])).item()
            _glocal_plow = torch_ext.mean_list(train_info.get('aaa_g_local_pred_low', [torch.tensor(float('nan'))])).item()
            _glocal_phigh = torch_ext.mean_list(train_info.get('aaa_g_local_pred_high', [torch.tensor(float('nan'))])).item()
            print(f'[AAA probe] Ep{self.epoch_num} alpha={_sa.item():+.5f} grad={_gv:+.3e} '
                  f'r_content={getattr(self,"_probe_r_content",float("nan")):.4f} '
                  f'r_disc={getattr(self,"_probe_r_disc",float("nan")):.4f} '
                  f'r_style={_style_r:.4f} r_respen={getattr(self,"_probe_r_respen",float("nan")):.4f} '
                  f'style_loss={_style_loss:+.4f} contrast={_contrast_loss:+.4f} '
                  f'proxy_mse={_proxy_loss:.5f} proxy_mae={_proxy_mae:.5f} '
                  f'proxy_dir={_proxy_dir:+.5f} proxy_gap={_proxy_gap:+.5f} '
                  f'proxy_action_grad={_proxy_ag:.5e} proxy_action_grad_max={_proxy_agmax:.5e} '
                  f'joint_loss={_joint_loss:+.5f} joint_gap={_joint_gap:+.5f} '
                  f'structured_loss={_structured_loss:+.5f} structured_gap={_structured_gap:+.5f} '
                  f'glocal_loss={_glocal_loss:+.5f} glocal_gap={_glocal_gap:+.5f} '
                  f'glocal_plow={_glocal_plow:+.5f} glocal_phigh={_glocal_phigh:+.5f}', flush=True)
        # === AAA end ===
        
        return train_info

    def pre_epoch(self, epoch_num):
        # === AAA 方案A v2: α schedule + clamp（gated by AAA_ALPHA_MODE，默认 none 零回归）===
        # 为什么：disc 探针定因——自然平衡 α~0.02 太小（content+respen 压制 + disc 杠杆自限），
        #   风格显现需 |α|~0.3；forced-α=-0.3 sweep 已证架构能产可控风格（step_width +139%、
        #   elbow_rom +52%）。但方案A从 phc_p1_extracted 重训时 residual 是 fresh init（gain 1e-3），
        #   直接大 α = 大随机扰动 → destabilize，故先 ramp 0→target 让 residual 学映射再放大；
        #   ramp 结束后 clamp 下界锁 |α|≥clamp，防被 content 拉回 0.02（探针观测到的失败模式）。
        # 做什么：每 epoch 开头按 mode 调 style_alpha.data。
        #   none（默认）=不干预，smoke/w3 零回归；schedule=ramp+clamp（方案A 重训用）；
        #   warmup=只 ramp 到 target，不 clamp（W5 step_width canary 用，避免大 α 放大 slider-independent residual）。
        #   fixed_positive=固定正 α 做 Step 6 方向诊断，验证反向 sweep 是否来自 α/residual 符号闭环。
        #   ramp 期覆盖 α.data 会抹掉 PPO 上一 epoch 的 α 更新——这是 schedule 本意（强制轨迹）；
        #   PPO 的 Adam m/v 不受影响，grad 继续反传，ramp 结束后 PPO 自由推（仅受 clamp 下界约束）。
        _alpha_mode = os.environ.get('AAA_ALPHA_MODE', 'none')
        if _alpha_mode in ['schedule', 'warmup', 'fixed_positive'] and hasattr(self.model, 'a2c_network') \
                and hasattr(self.model.a2c_network, 'style_alpha'):
            _sa = self.model.a2c_network.style_alpha
            if _alpha_mode == 'fixed_positive':
                # === AAA W5 Step 6: fixed positive alpha direction diagnosis ===
                # 为什么：Ep150 真实 rollout 中 step_width 随 slider 稳定反向，而 adapter 已强读 slider；
                #   需要隔离 α 符号，确认反向是否来自 α/residual 的符号组合，而不是继续加训或扩维。
                # 做什么：用 AAA_ALPHA_VALUE（默认 +0.10）每 epoch 固定 α，并关闭该参数梯度，
                #   让本次诊断只训练 residual/encoder，不让 optimizer 在 epoch 内把 α 推回负方向。
                _value = abs(float(os.environ.get('AAA_ALPHA_VALUE', '0.10')))
                _sa.requires_grad_(False)
                _sa.data.fill_(_value)
                # === AAA end ===
            else:
                if not _sa.requires_grad:
                    _sa.requires_grad_(True)
                _target = float(os.environ.get('AAA_ALPHA_TARGET', '-0.1' if _alpha_mode == 'warmup' else '-0.3'))
                _ramp_ep = int(os.environ.get('AAA_ALPHA_RAMP_EP', '50'))
                _clamp = float(os.environ.get('AAA_ALPHA_CLAMP', '0.2'))
                if _ramp_ep > 0 and epoch_num <= _ramp_ep:
                    # 线性 ramp 0 → target（epoch 0 → 0，epoch ramp_ep → target）
                    _sa.data.fill_(_target * (epoch_num / _ramp_ep))
                elif _alpha_mode == 'schedule':
                    # ramp 结束后只锁下界 |α|≥clamp（符号跟随 target），PPO 自由推
                    if _target < 0:
                        _sa.data.clamp_(max=-_clamp)   # α ≤ -clamp（负方向 |α|≥clamp）
                    else:
                        _sa.data.clamp_(min=_clamp)
        # === AAA end ===

        # print("freeze running mean/std")

        if self.vec_env.env.task.humanoid_type in ["smpl", "smplh", "smplx"]:
            humanoid_env = self.vec_env.env.task
            if (epoch_num > 1) and epoch_num % humanoid_env.shape_resampling_interval == 1: # + 1 to evade the evaluations. 
            # if (epoch_num > 0) and epoch_num % humanoid_env.shape_resampling_interval == 0 and not (epoch_num % (self.save_freq)): # Remove the resampling for this. 
                # Different from AMP, always resample motion no matter the motion type.
                print("Resampling Shape")
                humanoid_env.resample_motions()
                # self.current_rewards # Fixing these values such that they do not get whacked by the
                # self.current_lengths
            if humanoid_env.getup_schedule:
                humanoid_env.update_getup_schedule(epoch_num, getup_udpate_epoch=humanoid_env.getup_udpate_epoch)
                if epoch_num > humanoid_env.getup_udpate_epoch:  # ZL fix janky hack
                    self._task_reward_w = 0.5
                    self._disc_reward_w = 0.5
                else:
                    self._task_reward_w = 0
                    self._disc_reward_w = 1

        self.running_mean_std_temp = copy.deepcopy(self.running_mean_std)  # Freeze running mean/std, so that the actor does not use the updated mean/std
        self.running_mean_std_temp.freeze()

    def post_epoch(self, epoch_num):
        self.running_mean_std_temp = copy.deepcopy(self.running_mean_std)  # Unfreeze running mean/std
        self.running_mean_std_temp.freeze()
        

    def _preproc_obs(self, obs_batch, use_temp=False):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v, use_temp = use_temp)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0

        if self.normalize_input:
            obs_batch_proc = obs_batch[:, :self.running_mean_std.mean_size]
            if use_temp:
                obs_batch_out = self.running_mean_std_temp(obs_batch_proc)
                obs_batch_orig = self.running_mean_std(obs_batch_proc)  # running through mean std, but do not use its value. use temp
            else:
                obs_batch_out = self.running_mean_std(obs_batch_proc)  # running through mean std, but do not use its value. use temp
            obs_batch_out = torch.cat([obs_batch_out, obs_batch[:, self.running_mean_std.mean_size:]], dim=-1)

        return obs_batch_out

    def calc_gradients(self, input_dict):
        
        self.set_train()
        humanoid_env = self.vec_env.env.task

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        style_advantage = input_dict.get('style_advantages', None)
        step_width_target = input_dict.get('step_width_norm', None)
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch_processed = self._preproc_obs(obs_batch, use_temp=self.temp_running_mean)
        input_dict['obs_processed'] = obs_batch_processed

        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        amp_slider = input_dict['slider'][0:self._amp_minibatch_size]  # AAA W4: agent 分支 slider

        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)
        amp_replay_slider = input_dict['amp_replay_slider'][0:self._amp_minibatch_size]  # AAA W4: replay 分支 slider

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)
        demo_slider = input_dict['demo_slider'][0:self._amp_minibatch_size]  # AAA W4: demo 分支 slider

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip
        
        self.train_result = {}
        
        batch_dict = {'is_train': True, 'amp_steps': self.vec_env.env.task._num_amp_obs_steps, \
            'prev_actions': actions_batch, 'obs': obs_batch_processed, 'amp_obs': amp_obs, 'amp_obs_replay': amp_obs_replay, 'amp_obs_demo': amp_obs_demo, \
                "obs_orig": obs_batch,
                # === AAA W4: 三路 slider 传给 model forward → conditional disc（amp_models.py）===
                # disc 的 slider 切到 _amp_minibatch_size(4096) 对齐 amp_obs；policy 的 slider 必须对齐
                # obs_batch(minibatch_size=16384，与 amp_minibatch_size 不同)，故用全量 input_dict['slider']。
                'amp_slider': amp_slider, 'amp_replay_slider': amp_replay_slider, 'demo_slider': demo_slider,
                # AAA W4: policy eval_actor 用 'slider' key（style_residual 条件化），对齐 obs_batch 行数（全量，不切 amp_minibatch）
                'slider': input_dict['slider'],
                # === AAA end ===
                }
    
        rnn_masks = None
        rnn_len = self.horizon_length
        rnn_len = 1
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = rnn_len
            
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict) # current model if RNN, has BPTT enabled. 
            
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            disc_agent_logit = res_dict['disc_agent_logit']
            disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
            disc_demo_logit = res_dict['disc_demo_logit']

            # === AAA W5 P2: train learned step_width proxy on rollout labels ===
            # 为什么：proxy 必须先拟合真实 rollout step_width_norm，direction loss 才能作为可信的
            #   可微信号。这里用采样 action 对应的真实标签监督，独立 optimizer 更新 proxy。
            if step_width_target is not None:
                proxy_train_loss, proxy_mae = self._aaa_train_step_proxy(obs_batch_processed, actions_batch, step_width_target)
            else:
                proxy_train_loss = torch.zeros((), device=self.ppo_device)
                proxy_mae = torch.zeros((), device=self.ppo_device)
            # === AAA end ===

            if not rnn_masks is None:
                rnn_mask_bool = rnn_masks.squeeze().bool()
                old_action_log_probs_batch, action_log_probs, advantage, values, entropy, mu, sigma, return_batch, old_mu_batch, old_sigma_batch = \
                    old_action_log_probs_batch[rnn_mask_bool], action_log_probs[rnn_mask_bool], advantage[rnn_mask_bool], values[rnn_mask_bool], \
                        entropy[rnn_mask_bool], mu[rnn_mask_bool], sigma[rnn_mask_bool], return_batch[rnn_mask_bool], old_mu_batch[rnn_mask_bool], old_sigma_batch[rnn_mask_bool]
                if style_advantage is not None:
                    style_advantage = style_advantage[rnn_mask_bool]
                
                # flatten values for computing loss
                
                
            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            # === AAA W5: style-specific PPO actor loss（默认 coef=0 零回归）===
            # 为什么：C+ reward-shaping-only 失败说明 style reward 混进总 reward 后被 content/disc 稀释；
            #   这里用单独 GAE 得到的 style_advantage 再加一条 PPO actor loss，让 step_width metric
            #   有独立策略梯度通道。
            if self._aaa_style_adv_coef > 0.0 and style_advantage is not None:
                _style_adv = style_advantage
                if self.normalize_advantage:
                    # === AAA W5: keep signed style direction during advantage scaling ===
                    # 为什么：step_width contrast canary 中 r_style 近似常数，mean-zero normalize 把 style actor
                    #   loss 压成 0；signed improvement reward 已经提供正/负方向，不能再减 batch mean。
                    # 做什么：只按 std 缩放以控制量级，保留 advantage 的符号和均值方向。
                    _style_adv = _style_adv / (_style_adv.std() + 1e-8)
                    # === AAA end ===
                style_a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, _style_adv, curr_e_clip)
                style_a_loss = torch.mean(style_a_info['actor_loss'])
            else:
                style_a_loss = torch.zeros((), device=self.ppo_device)
            # === AAA end ===

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)

            a_loss = torch.mean(a_loss)
            c_loss = torch.mean(c_loss)
            b_loss = torch.mean(b_loss)
            entropy = torch.mean(entropy)

            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
            disc_loss = disc_info['disc_loss']

            # === AAA W5: residual contrast loss（默认 coef=0 零回归）===
            # 为什么：当前最硬失败点是同 obs 切 slider 时 Δa 只有 1.10% 变化；物理 reward 链路太长，
            #   需要一条直接作用到 slider_encoder + style_residual 的可微信号，强制 adapter 读 slider。
            # 做什么：同一 obs 构造 step_width low/high 两个 slider，最大化 Δa 差异；同时用 delta/kl
            #   约束防止无意义大动作。alpha_floor 在 α 过小时给 residual 有效梯度，不改变真实 eval_actor。
            if self._aaa_contrast_coef > 0.0 and getattr(self.model.a2c_network, 'style_enabled', False):
                _slider_mid = input_dict['slider'].clone()
                _slider_low = _slider_mid.clone()
                _slider_high = _slider_mid.clone()
                _slider_low[:, self._aaa_contrast_dim] = 0.0
                _slider_high[:, self._aaa_contrast_dim] = 1.0
                _sa = self.model.a2c_network.style_alpha
                _sign = torch.sign(_sa.detach())
                if torch.abs(_sign).item() < 0.5:
                    _sign = torch.tensor(-1.0, device=self.ppo_device)
                _alpha_eff = _sign * torch.clamp(torch.abs(_sa.detach()), min=self._aaa_contrast_alpha_floor)
                _delta_low = self.model.a2c_network.eval_style_delta(obs_batch_processed, _slider_low, alpha_override=_alpha_eff)
                _delta_high = self.model.a2c_network.eval_style_delta(obs_batch_processed, _slider_high, alpha_override=_alpha_eff)
                contrast_loss = -torch.norm(_delta_high - _delta_low, dim=-1).mean()
                contrast_delta_loss = ((_delta_low ** 2).sum(dim=-1) + (_delta_high ** 2).sum(dim=-1)).mean()
                contrast_kl_loss = ((mu - old_mu_batch) ** 2).sum(dim=-1).mean()
            else:
                contrast_loss = torch.zeros((), device=self.ppo_device)
                contrast_delta_loss = torch.zeros((), device=self.ppo_device)
                contrast_kl_loss = torch.zeros((), device=self.ppo_device)
            # === AAA end ===

            # === AAA W5 P2: proxy direction loss ===
            # 为什么：EMA canary 已证 reward 标量能算对但 sweep 仍 flat；这里用冻结 proxy 直接约束
            #   同一 obs 下 high slider action 的 predicted step_width 高于 low slider action。
            _proxy_dir_enabled = self._aaa_proxy_dir_coef > 0.0 and self.epoch_num >= self._aaa_proxy_dir_warmup_epochs
            if _proxy_dir_enabled and getattr(self.model.a2c_network, 'style_enabled', False):
                _slider_mid = input_dict['slider'].clone()
                _slider_low = _slider_mid.clone()
                _slider_high = _slider_mid.clone()
                _slider_low[:, self._aaa_contrast_dim] = 0.0
                _slider_high[:, self._aaa_contrast_dim] = 1.0
                _sa = self.model.a2c_network.style_alpha
                _sign = torch.sign(_sa.detach())
                if torch.abs(_sign).item() < 0.5:
                    _sign = torch.tensor(-1.0, device=self.ppo_device)
                _alpha_eff = _sign * torch.clamp(torch.abs(_sa.detach()), min=self._aaa_contrast_alpha_floor)
                _delta_mid = self.model.a2c_network.eval_style_delta(obs_batch_processed, _slider_mid)
                _delta_low = self.model.a2c_network.eval_style_delta(obs_batch_processed, _slider_low, alpha_override=_alpha_eff)
                _delta_high = self.model.a2c_network.eval_style_delta(obs_batch_processed, _slider_high, alpha_override=_alpha_eff)
                _base_mu = mu - _delta_mid
                self._aaa_ensure_step_proxy(obs_batch_processed.shape[-1], mu.shape[-1])
                _prev_requires_grad = [p.requires_grad for p in self._aaa_step_proxy.parameters()]
                for p in self._aaa_step_proxy.parameters():
                    p.requires_grad_(False)
                _pred_low = self._aaa_step_proxy_forward(obs_batch_processed, _base_mu + _delta_low)
                _pred_high = self._aaa_step_proxy_forward(obs_batch_processed, _base_mu + _delta_high)
                for p, req in zip(self._aaa_step_proxy.parameters(), _prev_requires_grad):
                    p.requires_grad_(req)
                proxy_dir_loss = torch.relu(self._aaa_proxy_margin - (_pred_high - _pred_low)).mean()
                proxy_dir_gap = (_pred_high - _pred_low).mean()
                # 诊断 g 是否真的看 action：用 detach 分支测 ∂g/∂action，避免影响 actor 梯度图。
                _diag_action = (_base_mu + _delta_high).detach().requires_grad_(True)
                _diag_pred = self._aaa_step_proxy_forward(obs_batch_processed.detach(), _diag_action)
                _diag_grad = torch.autograd.grad(_diag_pred.mean(), _diag_action, retain_graph=False, create_graph=False)[0]
                proxy_action_grad = _diag_grad.norm(dim=-1).mean()
                proxy_action_grad_max = _diag_grad.abs().max()
            else:
                proxy_dir_loss = torch.zeros((), device=self.ppo_device)
                proxy_dir_gap = torch.zeros((), device=self.ppo_device)
                proxy_action_grad = torch.zeros((), device=self.ppo_device)
                proxy_action_grad_max = torch.zeros((), device=self.ppo_device)
            # === AAA end ===

            # === AAA W5 P2b: analytic/action-space step_width direction loss ===
            # 为什么：naive learned proxy 的 action gradient 接近 0；这里直接让 high step_width slider
            #   在左右髋部外展代理分数上高于 low slider，给 residual 一条明确动作方向。
            if self._aaa_joint_step_coef > 0.0 and getattr(self.model.a2c_network, 'style_enabled', False):
                _slider_mid = input_dict['slider'].clone()
                _slider_low = _slider_mid.clone()
                _slider_high = _slider_mid.clone()
                _slider_low[:, self._aaa_contrast_dim] = 0.0
                _slider_high[:, self._aaa_contrast_dim] = 1.0
                _sa = self.model.a2c_network.style_alpha
                _sign = torch.sign(_sa.detach())
                if torch.abs(_sign).item() < 0.5:
                    _sign = torch.tensor(-1.0, device=self.ppo_device)
                _alpha_eff = _sign * torch.clamp(torch.abs(_sa.detach()), min=self._aaa_contrast_alpha_floor)
                _delta_mid = self.model.a2c_network.eval_style_delta(obs_batch_processed, _slider_mid)
                _delta_low = self.model.a2c_network.eval_style_delta(obs_batch_processed, _slider_low, alpha_override=_alpha_eff)
                _delta_high = self.model.a2c_network.eval_style_delta(obs_batch_processed, _slider_high, alpha_override=_alpha_eff)
                _base_mu = mu - _delta_mid
                _a_low = _base_mu + _delta_low
                _a_high = _base_mu + _delta_high
                _score_low = (
                    self._aaa_joint_step_left_sign * _a_low[:, self._aaa_joint_step_left_idx]
                    + self._aaa_joint_step_right_sign * _a_low[:, self._aaa_joint_step_right_idx]
                )
                _score_high = (
                    self._aaa_joint_step_left_sign * _a_high[:, self._aaa_joint_step_left_idx]
                    + self._aaa_joint_step_right_sign * _a_high[:, self._aaa_joint_step_right_idx]
                )
                joint_step_gap = (_score_high - _score_low).mean()
                joint_step_loss = torch.relu(self._aaa_joint_step_margin - (_score_high - _score_low)).mean()
            else:
                joint_step_loss = torch.zeros((), device=self.ppo_device)
                joint_step_gap = torch.zeros((), device=self.ppo_device)
            # === AAA end ===

            # === AAA Step C: structured scalar direction loss ===
            # 为什么：fixed canary 已确认 D2_axis0 是 action-rescuable 方向；这里只要求 high slider
            #   的 scalar-projected Δa 比 low slider 高 margin，梯度只需进入 scalar head/slider encoder。
            _net = self.model.a2c_network
            _structured_enabled = (
                self._aaa_structured_step_coef > 0.0
                and getattr(_net, 'style_enabled', False)
                and hasattr(_net, 'eval_structured_delta')
            )
            if _structured_enabled:
                _slider_mid = input_dict['slider'].clone()
                _slider_low = _slider_mid.clone()
                _slider_high = _slider_mid.clone()
                _slider_low[:, self._aaa_contrast_dim] = 0.0
                _slider_high[:, self._aaa_contrast_dim] = 1.0
                _delta_low = _net.eval_structured_delta(
                    obs_batch_processed, _slider_low, amp_override=self._aaa_structured_step_amp)
                _delta_high = _net.eval_structured_delta(
                    obs_batch_processed, _slider_high, amp_override=self._aaa_structured_step_amp)
                _direction = _net.aaa_structured_direction.view(1, -1)
                _score_low = (_delta_low * _direction).sum(dim=-1)
                _score_high = (_delta_high * _direction).sum(dim=-1)
                structured_step_gap = (_score_high - _score_low).mean()
                structured_step_loss = torch.relu(
                    self._aaa_structured_step_margin - (_score_high - _score_low)).mean()
            else:
                structured_step_loss = torch.zeros((), device=self.ppo_device)
                structured_step_gap = torch.zeros((), device=self.ppo_device)
            # === AAA end ===

            # === AAA Step C3: g_local dense surrogate loss（默认关闭，零回归）===
            # 为什么：C1 的 structured_step_loss 用 action 投影内积做 margin，未真正接上"action->step_width"
            #   物理响应；C3 用 C2 训好的 g_local(obs, delta)->Δstep_width_event 当 dense 可微 surrogate，
            #   约束 high slider 的 delta 经 g_local 预测的 Δstep_width 比 low slider 高 margin，
            #   梯度经 delta_action 直通 structured scalar head/slider_encoder（plan §486-498, §667-679）。
            # 做什么：构造 slider low/high -> eval_structured_delta 得 delta_low/high（D2_axis0 方向）->
            #   g_local 预测 -> ReLU(margin-(pred_high-pred_low)).mean()。g_local 冻结，梯度只回 scalar head。
            _g_local_enabled = (
                self._aaa_g_local_coef > 0.0
                and getattr(_net, 'style_enabled', False)
                and hasattr(_net, 'eval_structured_delta')
            )
            if _g_local_enabled:
                try:
                    self._aaa_ensure_g_local(
                        obs_batch.shape[-1],
                        int(_net.aaa_structured_direction.shape[0]))
                except Exception as _e:
                    print(f"[AAA g_local] ensure failed (disabling for this run): {_e}", flush=True)
                    self._aaa_g_local_coef = 0.0
            if _g_local_enabled and self._aaa_g_local is not None:
                _gl_slider_mid = input_dict['slider'].clone()
                _gl_slider_low = _gl_slider_mid.clone()
                _gl_slider_high = _gl_slider_mid.clone()
                _gl_slider_low[:, self._aaa_contrast_dim] = 0.0
                _gl_slider_high[:, self._aaa_contrast_dim] = 1.0
                _gl_delta_low = _net.eval_structured_delta(
                    obs_batch_processed, _gl_slider_low, amp_override=self._aaa_structured_step_amp)
                _gl_delta_high = _net.eval_structured_delta(
                    obs_batch_processed, _gl_slider_high, amp_override=self._aaa_structured_step_amp)
                # g_local 训练在 Gate-0 raw onset_obs 上（env_reset 原始，未 RMS 归一），故喂 raw obs_batch，
                # g_local 内部用自己的 dataset obs_mean/std 归一；不能用 obs_batch_processed（agent RMS 归一）。
                _gl_pred_low = self._aaa_g_local_forward(obs_batch, _gl_delta_low)
                _gl_pred_high = self._aaa_g_local_forward(obs_batch, _gl_delta_high)
                g_local_step_gap = (_gl_pred_high - _gl_pred_low).mean()
                g_local_step_loss = torch.relu(
                    self._aaa_g_local_margin - (_gl_pred_high - _gl_pred_low)).mean()
                g_local_pred_low = _gl_pred_low.detach().mean()
                g_local_pred_high = _gl_pred_high.detach().mean()
            else:
                g_local_step_loss = torch.zeros((), device=self.ppo_device)
                g_local_step_gap = torch.zeros((), device=self.ppo_device)
                g_local_pred_low = torch.zeros((), device=self.ppo_device)
                g_local_pred_high = torch.zeros((), device=self.ppo_device)
            # === AAA end ===

            # === AAA 诊断C: 梯度审计（gated by AAA_GRAD_AUDIT，默认关零回归）===
            # 为什么：codex 4 轮 proxy/reward 改动 + Claude expC 大α长训后 sweep 仍 flat/失稳，
            #   需确认各 style loss 分量是否真进 style_residual/slider_encoder/style_alpha，
            #   以及谁的梯度量级主导（reward 路径是否被 contrast/PPO 覆盖？）。
            # 做什么：用 torch.autograd.grad（retain_graph）分别算每个 loss 分量对三组参数的
            #   grad norm，不动 param.grad、不干扰主 backward。只打印前 N 个 minibatch。
            if os.environ.get('AAA_GRAD_AUDIT', '0') == '1' and \
               self._aaa_grad_audit_count < int(os.environ.get('AAA_GRAD_AUDIT_N', '3')) and \
               self.epoch_num >= int(os.environ.get('AAA_GRAD_AUDIT_START_EP', '60')):
                # 注：retain_graph=True 的多次 autograd.grad 会累积显存（8 loss × 3 group = 24 次），
                # 1024 env 下会 OOM。用 AAA_GRAD_AUDIT_ENVS 降 num_envs 或减少 group 数规避。
                self._aaa_grad_audit_count += 1
                _net = self.model.a2c_network
                if getattr(_net, 'style_enabled', False):
                    _audit_param_groups = {
                        'style_residual': list(_net.style_residual.parameters()),
                        'structured_scalar': list(getattr(_net, 'aaa_structured_scalar_head', nn.Module()).parameters()),
                    }
                    _audit_losses = {
                        'a_loss(PPO)': a_loss,
                        'disc_loss': disc_loss,
                        'style_a_loss': style_a_loss,
                        'contrast_loss': contrast_loss,
                        'contrast_delta_loss': contrast_delta_loss,
                        'contrast_kl_loss': contrast_kl_loss,
                        'joint_step_loss': joint_step_loss,
                        'structured_step_loss': structured_step_loss,
                        'g_local_step_loss': g_local_step_loss,
                        'proxy_dir_loss': proxy_dir_loss,
                    }
                    print(f"[AAA grad_audit] === minibatch #{self._aaa_grad_audit_count} ===", flush=True)
                    _n_losses = len(_audit_losses)
                    _i_loss = 0
                    for _lname, _lobj in _audit_losses.items():
                        _i_loss += 1
                        for _gname, _gparams in _audit_param_groups.items():
                            try:
                                _grads = torch.autograd.grad(_lobj, _gparams,
                                                             retain_graph=True, allow_unused=True)
                                _gn_sq = 0.0
                                for _g in _grads:
                                    if _g is not None:
                                        _gn_sq += float((_g.detach() ** 2).sum().item())
                                print(f"[AAA grad_audit] {_lname:>22} -> {_gname:<16} grad_norm={_gn_sq ** 0.5:.6e}", flush=True)
                            except RuntimeError as _e:
                                print(f"[AAA grad_audit] {_lname:>22} -> {_gname:<16} ERR={_e}", flush=True)
            # === AAA end ===

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                + self._disc_coef * disc_loss \
                + self._aaa_style_adv_coef * style_a_loss \
                + self._aaa_contrast_coef * contrast_loss \
                + self._aaa_contrast_delta_coef * contrast_delta_loss \
                + self._aaa_contrast_kl_coef * contrast_kl_loss \
                + self._aaa_proxy_dir_coef * proxy_dir_loss \
                + self._aaa_joint_step_coef * joint_step_loss \
                + self._aaa_structured_step_coef * structured_step_loss \
                + self._aaa_g_local_coef * g_local_step_loss
            
            
            a_clip_frac = torch.mean(a_info['actor_clipped'].float())

            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            a_info['aaa_style_actor_loss'] = style_a_loss.detach()
            a_info['aaa_contrast_loss'] = contrast_loss.detach()
            a_info['aaa_contrast_delta_loss'] = contrast_delta_loss.detach()
            a_info['aaa_contrast_kl_loss'] = contrast_kl_loss.detach()
            a_info['aaa_proxy_train_loss'] = proxy_train_loss.detach()
            a_info['aaa_proxy_mae'] = proxy_mae.detach()
            a_info['aaa_proxy_dir_loss'] = proxy_dir_loss.detach()
            a_info['aaa_proxy_dir_gap'] = proxy_dir_gap.detach()
            a_info['aaa_proxy_action_grad'] = proxy_action_grad.detach()
            a_info['aaa_proxy_action_grad_max'] = proxy_action_grad_max.detach()
            a_info['aaa_joint_step_loss'] = joint_step_loss.detach()
            a_info['aaa_joint_step_gap'] = joint_step_gap.detach()
            a_info['aaa_structured_step_loss'] = structured_step_loss.detach()
            a_info['aaa_structured_step_gap'] = structured_step_gap.detach()
            a_info['aaa_g_local_loss'] = g_local_step_loss.detach()
            a_info['aaa_g_local_gap'] = g_local_step_gap.detach()
            a_info['aaa_g_local_pred_low'] = g_local_pred_low.detach()
            a_info['aaa_g_local_pred_high'] = g_local_pred_high.detach()
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        
        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = kl_dist.mean()
        
                
        #TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
          
        self.train_result.update( {'entropy': entropy, 'kl': kl_dist, 'last_lr': self.last_lr, 'lr_mul': lr_mul, 'b_loss': b_loss})
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)
            
        return

    def _load_config_params(self, config):
        super()._load_config_params(config)

        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        self._amp_observation_space = self.env_info['amp_observation_space']
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])
        assert (self._amp_minibatch_size <= self.minibatch_size)

        self._disc_coef = config['disc_coef']
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape
        
        config['task_obs_size_detail'] = self.vec_env.env.task.get_task_obs_size_detail()
        if self.vec_env.env.task.has_task:
            config['self_obs_size'] = self.vec_env.env.task.get_self_obs_size()
            config['task_obs_size'] = self.vec_env.env.task.get_task_obs_size()

        return config

    def _init_train(self):
        super()._init_train()
        self._init_amp_demo_buf()
        return


    def _oracle_loss(self, obs):
        oracle_a, _ = self.oracle_model.a2c_network.eval_actor({"obs": obs})
        model_a, _ = self.model.a2c_network.eval_actor({"obs": obs})
        oracle_loss = (oracle_a - model_a).pow(2).mean(dim=-1) * 50
        return {'oracle_loss': oracle_loss}

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        '''
        disc_agent_logit: replay and current episode logit (fake examples)
        disc_demo_logit: disc_demo_logit logit 
        obs_demo: gradient penalty demo obs (real examples)
        '''
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights)) # make weight small??
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit), create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]

        ### ZL Hack for zeroing out gradient penalty on the shape (406,)
        # if self.vec_env.env.task.__dict__.get("smpl_humanoid", False):
        #     humanoid_env = self.vec_env.env.task
        #     B, feat_dim = disc_demo_grad.shape
        #     shape_obs_dim = 17
        #     if humanoid_env.has_shape_obs:
        #         amp_obs_dim = int(feat_dim / humanoid_env._num_amp_obs_steps)
        #         for i in range(humanoid_env._num_amp_obs_steps):
        #             disc_demo_grad[:,
        #                            ((i + 1) * amp_obs_dim -
        #                             shape_obs_dim):((i + 1) * amp_obs_dim)] = 0

        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)

        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        # print(f"agent_loss: {disc_loss_agent.item():.3f}  | disc_loss_demo {disc_loss_demo.item():.3f}")
        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty.detach(),
            'disc_logit_loss': disc_logit_loss.detach(),
            'disc_agent_acc': disc_agent_acc.detach(),
            'disc_demo_acc': disc_demo_acc.detach(),
            'disc_agent_logit': disc_agent_logit.detach(),
            'disc_demo_logit': disc_demo_logit.detach()
        }
        return disc_info
    
    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self, num_samples):
        # === AAA W4: env 返回 (amp_obs_demo, demo_slider)，透传 slider（w4 patch §3）===
        amp_obs_demo, demo_slider = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo, demo_slider
        # === AAA end ===

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['amp_obs'] = torch.zeros(batch_shape + self._amp_observation_space.shape, device=self.ppo_device)
        # === AAA W4: slider buffer（复用 amp_obs buffer 路径，w4 patch §1.2）===
        # 为什么：slider 与 amp_obs 同形状前导（batch_shape），随 amp_obs 一起 update_data/swap_and_flatten，
        #   训练时取 mb_slider 喂 conditional disc（reward 路径）+ minibatch 喂 disc（训练路径）。
        #   加进 tensor_list 使 play_steps 的 get_transformed_list 把 slider 塞进 batch_dict（供 replay 存盘）。
        self.experience_buffer.tensor_dict['slider'] = torch.zeros(batch_shape + (6,), device=self.ppo_device)
        # === AAA W5: C+ metric style reward buffer ===
        # 为什么：style-specific PPO advantage 需要保留 env 侧独立 r_style，不能只依赖已混合的 rewards。
        #   默认 loss coef=0 时只记录不参与优化。
        self.experience_buffer.tensor_dict['style_reward'] = torch.zeros(batch_shape + (1,), device=self.ppo_device)
        # === AAA W5 P2: proxy supervised target ===
        # 为什么：learned proxy 需要 rollout 真实 step_width_norm 作监督标签；NaN 表示 C+ 未开启或无标签。
        self.experience_buffer.tensor_dict['step_width_norm'] = torch.full(batch_shape + (1,), float('nan'), device=self.ppo_device)
        # === AAA end ===
        # === AAA W4: res_penalty buffer（‖Δa‖²，w4 patch §8.3）===
        # 为什么：存每步 ‖Δa‖² 供 reward 合并时减 λ_res·‖Δa‖²（tanh 硬界双保险的 norm 惩罚，抉择2）。
        #   Δa 由 eval_actor 缓存（network 侧 _last_delta_a），play_steps 取算。shape (num_envs, horizon, 1) 对齐 rewards。
        self.experience_buffer.tensor_dict['res_penalty'] = torch.zeros(batch_shape + (1,), device=self.ppo_device)
        self._res_lambda = float(self.config.get('res_lambda', 0.001))  # λ_res，默认 0.001 起步（可 config 覆盖）
        # === AAA end ===
        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)  # Demo is the data from the dataset. Real samples

        self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ['amp_obs']
        self.tensor_list += ['slider']  # AAA W4: slider 随 amp_obs 进 batch_dict
        self.tensor_list += ['style_reward']  # AAA W5: r_style 随 rollout 进 batch_dict
        self.tensor_list += ['step_width_norm']  # AAA W5 P2: proxy 监督标签
        return

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            # === AAA W4: demo buffer 同步存 slider（w4 patch §3）===
            curr_samples, curr_slider = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples, 'slider': curr_slider})
            # === AAA end ===

        return

    def _update_amp_demos(self):
        # === AAA W4: demo buffer 同步存 slider ===
        new_amp_obs_demo, new_demo_slider = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo, 'slider': new_demo_slider})
        # === AAA end ===
        return

    def _norm_disc_reward(self):
        return self._disc_reward_mean_std is not None

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']

        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs, slider=None):
        # AAA W4: 透传 slider 给 conditional disc（reward 路径）。amp_obs 在此 preproc，slider 保持 raw。
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs, slider)

    def _calc_amp_rewards(self, amp_obs, slider=None):
        # AAA W4: slider 透传到 disc reward 计算（w4 patch §2.3）。
        disc_r = self._calc_disc_rewards(amp_obs, slider)
        output = {'disc_rewards': disc_r}
        return output

    def _calc_disc_rewards(self, amp_obs, slider=None):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs, slider)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))

            if (self._norm_disc_reward()):
                self._disc_reward_mean_std.train()
                norm_disc_r = self._disc_reward_mean_std(disc_r.flatten())
                disc_r = norm_disc_r.reshape(disc_r.shape)
                disc_r = 0.5 * disc_r + 0.25

            disc_r *= self._disc_reward_scale

        return disc_r

    def _store_replay_amp_obs(self, amp_obs, slider=None):
        # === AAA W4: replay buffer 同步存 slider，保证 replay 分支条件化正确（w4 patch §2.3）===
        # 为什么：replay 存的是历史 agent amp_obs，其对应 slider 也必须原样存盘，否则 disc 用错配 slider 训练。
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]
            if slider is not None:
                slider = slider[keep_mask]  # 同步过滤

        if (amp_obs.shape[0] > buf_size):
            rand_idx = torch.randperm(amp_obs.shape[0])
            rand_idx = rand_idx[:buf_size]
            amp_obs = amp_obs[rand_idx]
            if slider is not None:
                slider = slider[rand_idx]  # 同步过滤

        store_dict = {'amp_obs': amp_obs}
        if slider is not None:
            store_dict['slider'] = slider
        self._amp_replay_buffer.store(store_dict)
        # === AAA end ===
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        return
    
    def _assemble_train_info(self, train_info, frame):
        train_info_dict = super()._assemble_train_info(train_info, frame)
        
        if "disc_loss" in train_info:
            disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
            train_info_dict.update({
                "disc/loss": torch_ext.mean_list(train_info['disc_loss']).item(),
                "disc/agent_acc": torch_ext.mean_list(train_info['disc_agent_acc']).item(),
                "disc/demo_acc": torch_ext.mean_list(train_info['disc_demo_acc']).item(),
                "disc/agent_logit": torch_ext.mean_list(train_info['disc_agent_logit']).item(),
                "disc/demo_logit": torch_ext.mean_list(train_info['disc_demo_logit']).item(),
                "disc/grad_penalty": torch_ext.mean_list(train_info['disc_grad_penalty']).item(),
                "disc/logit_loss": torch_ext.mean_list(train_info['disc_logit_loss']).item(),
                "disc/reward_mean": disc_reward_mean.item(),
                "disc/reward_std": disc_reward_std.item(),
            })
        
        if "returns" in train_info:
            train_info_dict['rewards/returns'] = train_info['returns'].mean().item()
            
        if "mb_rewards" in train_info:
            train_info_dict['rewards/mb_rewards'] = train_info['mb_rewards'].mean().item()
        
        # if 'terminated_flags' in train_info:
        #     train_info_dict["success_rate"] =  1 - torch.mean((train_info['terminated_flags'] > 0).float()).item()
        
        if "reward_raw" in train_info:
            reward_raw=train_info['reward_raw'].cpu().numpy().tolist()
            train_info_dict["rewards/body_pos"] =  reward_raw[0]
            train_info_dict["rewards/body_rot"] =  reward_raw[1]
            train_info_dict["rewards/lin_vel"] =  reward_raw[2]
            train_info_dict["rewards/ang_vel"] =  reward_raw[3]
            train_info_dict["rewards/power"] =  reward_raw[4]
        
        if "sym_loss" in train_info:
            train_info_dict['loss/sym_loss'] = torch_ext.mean_list(train_info['sym_loss']).item()
        return train_info_dict

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            # === AAA W4: debug 也带 slider（info 由 env extras 提供，key='slider'）===
            dbg_slider = info.get('slider', None)
            if dbg_slider is not None:
                dbg_slider = dbg_slider[0:1]
            disc_pred = self._eval_disc(amp_obs, dbg_slider)
            amp_rewards = self._calc_amp_rewards(amp_obs, dbg_slider)
            # === AAA end ===
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            # print("disc_pred: ", disc_pred, disc_reward)
        return
