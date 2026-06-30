from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
import phc.learning.network_builder as network_builder
import torch
import torch.nn as nn
import numpy as np
import math
from phc.learning.style_residual import StyleResidual
from phc.learning.slider_encoder import SliderEncoder

DISC_LOGIT_INIT_SCALE = 1.0


class AMPBuilder(network_builder.A2CBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):

        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)

            amp_input_shape = kwargs.get('amp_input_shape')

            # === AAA: bounded style residual setup (W3) ===
            # base = 提取的单 primitive (actor_mlp + mu)，冻结；style_residual + slider_encoder 可训
            # 设计：SliderEncoder 放 network（非 env），optimizer 自动收 style 参数；
            #   W3 用固定 aaa_style_label buffer，无需 env/runner 改动；
            #   W4 起 runner 经 obs_dict['slider'] 传 6 维 raw label，network 内编码。
            # ⚠️ 必须在 _build_disc 之前设 style_enabled=True：_build_disc 据 style_enabled 把 disc 输入
            #   扩成 amp_obs_dim+32（w4 patch §2.1），顺序反了会导致 eval_disc cat z_style 后维度不匹配。
            self.style_enabled = True
            _aaa_obs_dim = self.actor_mlp[0].in_features   # 945 (no cnn)
            _aaa_act_dim = self.mu.out_features             # 69
            self.slider_encoder = SliderEncoder(slider_dim=6, latent_dim=32)
            self.style_residual = StyleResidual(
                obs_dim=_aaa_obs_dim, z_style_dim=32, action_dim=_aaa_act_dim)
            self.style_alpha = self.style_residual.style_alpha  # 暴露给 optimizer
            self.register_buffer('aaa_style_label', torch.zeros(1, 6))  # W3 固定 style 0
            # === AAA Step C: structured scalar head (default-off) ===
            # 为什么：Step B 已证明 D2_axis0 固定结构方向能产生可见 step_width 控制；Step C 不应回到
            #   无结构 69D residual，而是只让 policy 学一个 scalar bias，再写入该结构方向。
            # 做什么：新增 scalar head f(obs,z_style)->[-1,1]，方向固定为 L_Hip axis0 + / R_Hip axis0 -。
            #   默认 aaa_structured_step_enabled=False，不改变旧 eval_actor；参数进 optimizer 但无梯度。
            self.aaa_structured_step_enabled = bool(params.get('aaa_structured_step_enabled', False))
            self.aaa_structured_disable_raw = bool(params.get('aaa_structured_disable_raw_residual', False))
            self.aaa_structured_amp = float(params.get('aaa_structured_amp', 0.1))
            self.aaa_structured_scalar_head = nn.Sequential(
                nn.Linear(_aaa_obs_dim + 32, 128),
                nn.ELU(),
                nn.Linear(128, 1),
            )
            _aaa_dir = torch.zeros(_aaa_act_dim)
            if _aaa_act_dim > 12:
                _aaa_dir[0] = 1.0 / math.sqrt(2.0)
                _aaa_dir[12] = -1.0 / math.sqrt(2.0)
            self.register_buffer('aaa_structured_direction', _aaa_dir)
            for _p in self.actor_mlp.parameters(): _p.requires_grad_(False)
            for _p in self.mu.parameters():         _p.requires_grad_(False)
            # === AAA end ===

            self._build_disc(amp_input_shape)

            return

        def load(self, params):
            super().load(params)

            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']
            return

        def forward(self, obs_dict):
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs_dict)
            value_outputs = self.eval_critic(obs_dict)
            
            if self.has_rnn:
                mu, sigma, a_states = actor_outputs
                value, c_states = value_outputs
                states = a_states + c_states
                output = mu, sigma, value, states
            else:
                output = actor_outputs + (value_outputs, states)

            return output

        def eval_actor(self, obs_dict):
            # RNN is built with Batch-first enabled. 
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(-1, a_out.size(-1))

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    a_out_in = a_out
                    a_out = self.actor_mlp(a_out_in)
                    
                    if self.rnn_concat_input:
                        a_out = torch.cat([a_out, a_out_in], dim=1)

                batch_size = a_out.size()[0]
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1)

                if self.rnn_name == 'sru':
                    a_out = a_out.transpose(0, 1)

                ################# New RNN
                if len(states) == 2:
                    a_states = states[0].reshape(num_seqs, seq_length, -1)
                else:
                    a_states = states[:2].reshape(num_seqs, seq_length, -1)
                a_out, a_states = self.a_rnn(a_out, a_states[:, 0:1].transpose(0, 1).contiguous())
                
                ################ Old RNN
                # if len(states) == 2:	
                #     a_states = states[0]	
                # else:	
                #     a_states = states[:2]	
                # a_out, a_states = self.a_rnn(a_out, a_states)

                if self.rnn_name == 'sru':
                    a_out = a_out.transpose(0, 1)
                else:
                    if self.rnn_ln:
                        a_out = self.a_layer_norm(a_out)

                a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)

                if type(a_states) is not tuple:
                    a_states = (a_states,)

                if self.is_rnn_before_mlp:
                    a_out = self.actor_mlp(a_out)

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, a_states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, a_states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, a_states

            else:
                a_out = self.actor_mlp(a_out)
                
                # mlp_out = self.actor_mlp(a_out[:1])
                # (self.actor_mlp(a_out[:5])[0] - self.actor_mlp(a_out[:2])[0]).abs()

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, 

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, 

                if self.is_continuous:
                    
                    mu = self.mu_act(self.mu(a_out))
                    # === AAA: bounded style residual ===
                    # mu_style = mu_base + α·tanh(Δπ(s, z_style)); α init=0
                    # W3: obs_dict 无 'slider' 时用固定 aaa_style_label；W4 起 runner 传 raw slider
                    if self.style_enabled:
                        _aaa_slider = obs_dict.get('slider', None)
                        if self.aaa_structured_step_enabled and self.aaa_structured_disable_raw:
                            _aaa_delta = self.eval_structured_delta(obs, _aaa_slider)
                        else:
                            _aaa_delta = self.eval_style_delta(obs, _aaa_slider)  # Δa = α·tanh(Δπ)
                            if self.aaa_structured_step_enabled:
                                _aaa_delta = _aaa_delta + self.eval_structured_delta(obs, _aaa_slider)
                        mu = mu + _aaa_delta
                        # === AAA W4: 缓存 Δa 供 residual norm 惩罚（w4 patch §8.3）===
                        # 为什么：bounded 双保险的 norm 惩罚需精确 ‖Δa‖²（非 action proxy），W6 不用返工。
                        #   play_steps 取此缓存算 ‖Δa‖² 进 experience_buffer，reward 合并时减 λ_res·‖Δa‖²。
                        self._last_delta_a = _aaa_delta  # (B, 69)，play_steps 在 no_grad 下读取
                        # === AAA end ===
                    # === AAA end ===
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                    
                    return mu, sigma
                    # return torch.round(mu, decimals=3), sigma

            return

        def eval_style_delta(self, obs, slider=None, alpha_override=None):
            # === AAA W5: style delta helper for residual contrast loss ===
            # 为什么：C+ reward-shaping-only 失败的核心证据是同 obs 切 slider 时 Δa 仅 1.10% 变化。
            #   train_minibatch 需要直接比较同一 obs 下不同 slider 的 Δa，给 adapter 一条可微的
            #   "必须读 slider" 辅助信号；复用这里避免在 agent 侧复制 encoder/residual 细节。
            # 做什么：返回 effective_delta = α_eff * tanh(residual(obs, slider))，默认 α_eff=style_alpha；
            #   alpha_override 仅供 auxiliary loss 在 α 太小时给 residual 提供有效梯度，不改变 eval_actor 行为。
            if slider is None:
                slider = self.aaa_style_label.expand(obs.shape[0], -1)
            z_style = self.slider_encoder(slider)
            residual = self.style_residual(obs, z_style)
            alpha = self.style_alpha if alpha_override is None else alpha_override
            return alpha * torch.tanh(residual)
            # === AAA end ===

        def eval_structured_scalar(self, obs, slider=None):
            # === AAA Step C: scalar structured head ===
            # 为什么：只输出一个标量 gait bias，避免 69D residual 自由扰动。
            # 做什么：复用 slider_encoder，输入 obs+z_style，输出 tanh-bounded scalar。
            if slider is None:
                slider = self.aaa_style_label.expand(obs.shape[0], -1)
            z_style = self.slider_encoder(slider)
            return torch.tanh(self.aaa_structured_scalar_head(torch.cat([obs, z_style], dim=-1)))

        def eval_structured_delta(self, obs, slider=None, amp_override=None):
            # === AAA Step C: scalar -> fixed D2_axis0 action direction ===
            # 为什么：Step B fixed canary 的有效载体是 D2_axis0；训练时只学习 scalar 幅度。
            amp = self.aaa_structured_amp if amp_override is None else amp_override
            scalar = self.eval_structured_scalar(obs, slider)
            return amp * scalar * self.aaa_structured_direction.view(1, -1)
        
        def get_actor_paramters(self):
            return list(self.actor_mlp.parameters()) + list(self.actor_cnn.parameters()) + list(self.mu.parameters()) 

        def eval_critic(self, obs_dict):
            obs = obs_dict['obs']
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(-1, c_out.size(-1))
            seq_length = obs_dict.get('seq_length', 1)
            states = obs_dict.get('rnn_states', None)

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    c_out_in = c_out
                    c_out = self.critic_mlp(c_out_in)

                    if self.rnn_concat_input:
                        c_out = torch.cat([c_out, c_out_in], dim=1)

                batch_size = c_out.size()[0]
                num_seqs = batch_size // seq_length
                c_out = c_out.reshape(num_seqs, seq_length, -1)

                if self.rnn_name == 'sru':
                    c_out = c_out.transpose(0, 1)
                ################# New RNN
                if len(states) == 2:
                    c_states = states[1].reshape(num_seqs, seq_length, -1)
                else:
                    c_states = states[2:].reshape(num_seqs, seq_length, -1)
                c_out, c_states = self.c_rnn(c_out, c_states[:, 0:1].transpose(0, 1).contiguous()) # ZL: only pass the first state, others are ignored. ???            
                
                ################# Old RNN
                # if len(states) == 2:	
                #     c_states = states[1]	
                # else:	
                #     c_states = states[2:]	
                # c_out, c_states = self.c_rnn(c_out, c_states)
                
                
                if self.rnn_name == 'sru':
                    c_out = c_out.transpose(0, 1)
                else:
                    if self.rnn_ln:
                        c_out = self.c_layer_norm(c_out)
                c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                if type(c_states) is not tuple:
                    c_states = (c_states,)

                if self.is_rnn_before_mlp:
                    c_out = self.critic_mlp(c_out)
                value = self.value_act(self.value(c_out))
                return value, c_states

            else:
                c_out = self.critic_mlp(c_out)

                value = self.value_act(self.value(c_out))
                return value

        def eval_disc(self, amp_obs, slider=None):
            # === AAA W4: conditional disc（w4 patch §2.2）===
            # 为什么：disc 输入拼 z_style 才能学到"风格→动作"条件映射（否则条件信号被忽略，MultiAct 警告，w4 patch §6）。
            #   slider_encoder 与 policy style_residual 共享同一 encoder，保证风格信号一致（doc21 §4.2）。
            #   amp_obs 已由调用方 _preproc_amp_obs 归一化，slider 保持 raw [0,1]（已归一化，不再 preproc）。
            # 做什么：AAA 模式下 _build_disc 已把 disc 输入扩成 amp_obs_dim+32，故必须始终拼 z_style；
            #   slider 为 None 时回落固定 style 0（aaa_style_label，与 W3 一致），保证维度对齐、零回归。
            if getattr(self, 'style_enabled', False):
                if slider is None:
                    slider = self.aaa_style_label.expand(amp_obs.shape[0], -1)
                z_style = self.slider_encoder(slider)           # (B, 32)
                disc_in = torch.cat([amp_obs, z_style], dim=-1)
            else:
                disc_in = amp_obs
            # === AAA end ===
            disc_mlp_out = self._disc_mlp(disc_in)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights

        def _build_disc(self, input_shape):
            # === AAA W4: conditional disc 输入 = amp_obs ⊕ z_style(32)（w4 patch §2.1）===
            # 为什么：disc 第一层 Linear 输入维 = amp_obs_dim + 32，与 eval_disc 的 cat 对齐。
            if getattr(self, 'style_enabled', False):
                input_shape = (input_shape[0] + 32,)
            # === AAA end ===
            self._disc_mlp = nn.Sequential()

            mlp_args = {'input_size': input_shape[0], 'units': self._disc_units, 'activation': self._disc_activation, 'dense_func': torch.nn.Linear}
            self._disc_mlp = self._build_mlp(**mlp_args)

            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias)

            return

    def build(self, name, **kwargs):
        net = AMPBuilder.Network(self.params, **kwargs)
        return net
