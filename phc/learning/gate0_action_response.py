# === AAA Gate-0: action perturbation response test（设计稿 gate0_action_response_design.md）===
# 为什么：诊断D 证伪 reward 路径后，codex 最终裁决（plan §307-311）下一步 = gait-event horizon 的
#   action perturbation response test。在【不训练、不进梯度】前提下，量化 frozen PHC 的 lower-body
#   action 维持续偏移能否在步态周期尺度上产生符号稳定、量级超噪声底的 Δstep_width，据此分支到
#   masked local response proxy / structured residual / 换维度。
# 做什么：player 模式（仿 aaa_w4_style_sweep.py 驱动），bypass adapter（style_alpha=0 退化成 base policy），
#   在 base mu 上加 ±δ·direction 持续偏移，rollout 到下一次同侧脚 strike（gait event），配对测
#   Δstep_width = sw_pert − sw_base（baseline 同初态不加偏移跑相同步数）。多次 reset 采样不同 onset phase，
#   按 stance/swing 分桶聚合 sign accuracy / effect size / δ 线性性 / content cost。
#
# ⚠️ 仅诊断脚本，不进训练路径；本地只写不跑，云端 player 模式跑。遵循 cloud workflow。
# ⚠️ 本脚本为 Claude 单方按设计稿落地，设计稿 6 个开放点此处取默认值（direction 按 dof_names 动态匹配
#    解决开放点#1；阈值/档位/bucket/裁决阈值/repeat 见 env var 可配），待 codex 共议后可调参重跑。
import os
import os.path as osp
import sys
import json
import math

import numpy as np
import torch
import joblib


# ------------------------------------------------------------------------------
# direction 构建（开放点#1：按 dof_names 动态匹配，不硬编码维数）
# ------------------------------------------------------------------------------
def _find_dof_idx(dof_names, keys):
    """按子串匹配返回 dof 索引列表（keys 任一子串命中即收）。"""
    out = []
    for i, n in enumerate(dof_names):
        if any(k in n for k in keys):
            out.append(i)
    return out


def build_directions(dof_names, num_actions):
    """建候选 perturbation direction，返回 dict[name] = (action_idx_list, sign_list)。

    ⚠️ SMPL humanoid（实际 robot=smpl_humanoid）：_dof_names = body_names[1:] 是 23 个 joint name
      （L_Hip/R_Hip/Spine1/L_Knee/R_Knee/Spine2/L_Ankle/R_Ankle/...），action = 69 维 = 23 joint × 3
      axis-angle，joint-major flatten（humanoid_im.py:970 reshape(-1, len(_dof_names), 3)）。
      故 joint j 的 3 个 action 维 = [j*3, j*3+1, j*3+2]。
    H1（has_dof_subset, dof 级）：dof_names 长度 = num_actions，每维一个 dof（hip_roll=abduction）。
    本函数两种都支持：若 num_actions == len(dof_names)*3 → SMPL joint×3 模式；否则 dof 级模式。
    pelvis 不在 action 空间（root 6D 不可控）→ D2 用 hip 反对称作骨盆横移 proxy（设计稿开放点#1 修订）。
    """
    smpl_mode = (num_actions == len(dof_names) * 3)
    def joint_to_action_idx(joint_idx_list):
        if smpl_mode:
            out = []
            for j in joint_idx_list:
                out.extend([j * 3, j * 3 + 1, j * 3 + 2])
            return out
        return list(joint_idx_list)
    def find_joints(keys):
        return [i for i, n in enumerate(dof_names) if any(k in n for k in keys)]

    L_hip = find_joints(['L_Hip', 'left_hip_roll', 'l_hip_roll'])
    R_hip = find_joints(['R_Hip', 'right_hip_roll', 'r_hip_roll'])
    L_ankle = find_joints(['L_Ankle', 'left_ankle'])
    R_ankle = find_joints(['R_Ankle', 'right_ankle'])
    L_knee = find_joints(['L_Knee', 'left_knee'])
    R_knee = find_joints(['R_Knee', 'right_knee'])

    dirs = {}
    # D0: lower-body mask（无结构 baseline）—— 全部下肢 joint 3 轴同号
    lower_j = sorted(set(L_hip + R_hip + L_knee + R_knee + L_ankle + R_ankle))
    if lower_j:
        idx = joint_to_action_idx(lower_j)
        dirs['D0_lower_mask'] = (idx, [1.0] * len(idx))
    # D1: hip pair 对称（abduction isotropic 近似——SMPL axis-angle 哪轴是 abduction 需分轴探测，
    #    Gate-0 第一版用 3 轴同号作 isotropic proxy，若响应弱可改分轴 D1x/D1y/D1z）
    if L_hip and R_hip:
        idx = joint_to_action_idx(L_hip + R_hip)
        dirs['D1_hip_abd_pair'] = (idx, [1.0] * len(idx))
    # D2: pelvis sway proxy（hip 反对称：左+右- → 骨盆横移倾向）
    if L_hip and R_hip:
        idx_l = joint_to_action_idx(L_hip); idx_r = joint_to_action_idx(R_hip)
        dirs['D2_pelvis_sway'] = (idx_l + idx_r,
                                  [+1.0] * len(idx_l) + [-1.0] * len(idx_r))
    # D3: ankle lateral proxy（左右踝同向 → 触地横向位置）
    if L_ankle and R_ankle:
        idx = joint_to_action_idx(L_ankle + R_ankle)
        dirs['D3_ankle_lat'] = (idx, [1.0] * len(idx))
    return dirs


def build_offset_tensor(num_actions, num_envs, env_cond, directions, delta):
    """建 (num_envs, num_actions) offset tensor：baseline env=0，pert env=sign*δ·direction。

    env_cond: list of (env_idx, is_baseline, cond_key, sign) 描述每个 env 的角色。
    direction 向量归一化到 L2=1 后乘 δ，使不同 direction 的 δ 量级可比（设计稿 §4 δ 是 direction 向量 L2）。
    """
    offset = torch.zeros(num_envs, num_actions, device='cuda')
    for env_idx, is_baseline, cond_key, sign in env_cond:
        if is_baseline:
            continue
        dof_idx, dof_sign = directions[cond_key]
        vec = torch.zeros(num_actions, device='cuda')
        for j, di in enumerate(dof_idx):
            vec[di] = float(dof_sign[j])
        nrm = vec.norm()
        if nrm > 0:
            vec = vec / nrm  # 单位化
        offset[env_idx] = sign * delta * vec
    return offset


# ------------------------------------------------------------------------------
# player.get_action 注入 hook（bypass adapter via style_alpha=0，在 base mu 上加 offset）
# ------------------------------------------------------------------------------
def install_action_offset_hook(player, offset_tensor):
    """monkeypatch player.get_action：复刻原逻辑，在 mu 上加 offset_tensor（持续偏移，每帧）。

    为什么 hook 在 mu（rescale 前）：诊断B 的 Δa L2~0.025 是 mu 空间量纲，δ 标定与此对齐；
      hook 后让 player 原 rescale/clip 逻辑照常走，不破坏 PD target 映射。
    bypass adapter：调用方应先 force style_alpha=0，使 res_dict['mus'] = base mu（adapter 退化）。
    """
    orig_get_action = player.get_action

    def patched_get_action(obs_dict, is_determenistic=False):
        from rl_games.common.tr_helpers import unsqueeze_obs
        obs = obs_dict['obs']
        if player.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = player._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': player.states,
        }
        if getattr(player.model.a2c_network, 'style_enabled', False):
            if getattr(player, '_eval_slider_override', None) is not None:
                input_dict['slider'] = player._eval_slider_override
            else:
                input_dict['slider'] = player.env.task.style_labels
        with torch.no_grad():
            res_dict = player.model(input_dict)
        mu = res_dict['mus']
        # === AAA Gate-0: 在 base mu 上加持续 action offset（bypass adapter 已由 force_alpha=0 保证）===
        mu = mu + offset_tensor.to(mu.device)
        current_action = mu  # is_determenistic 路径
        if player.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
        if player.clip_actions:
            from phc.learning.amp_players import rescale_actions
            return rescale_actions(player.actions_low, player.actions_high,
                                   torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    player.get_action = patched_get_action
    return orig_get_action


# ------------------------------------------------------------------------------
# 度量
# ------------------------------------------------------------------------------
def _step_width_from_body_pos(body_pos, step_body_ids, step_axis):
    """body_pos: (num_envs, num_bodies, 3) → step_raw = |ankle_l[axis] - ankle_r[axis]| per env."""
    ankle = body_pos[:, step_body_ids, :]  # (N, 2, 3)
    return torch.abs(ankle[:, 0, step_axis] - ankle[:, 1, step_axis])  # (N,)


def _foot_contact(contact_forces, foot_body_ids):
    """contact_forces: (num_envs, num_bodies, 3) → per-foot force magnitude (num_envs, num_feet)."""
    f = contact_forces[:, foot_body_ids, :]  # (N, 2, 3)
    return torch.norm(f, dim=-1)  # (N, 2)


# ------------------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------------------
def run_gate0(runner, cfg):
    from phc.utils.flags import flags

    debug_path = os.environ.get('AAA_GATE0_DEBUG_PATH')
    def _debug(msg):
        if debug_path:
            with open(debug_path, 'a') as f:
                f.write(str(msg) + "\n"); f.flush()
    _debug("stage=enter_run_gate0")

    # --- 参数（env var 优先，cfg 兜底；设计稿默认值）---
    motion_ids = [int(x) for x in os.environ.get('AAA_GATE0_MOTION_IDS', '29,13').split(',')]
    dir_keys_env = os.environ.get('AAA_GATE0_DIRECTIONS', '')  # 空=全部候选
    dir_keys = [k.strip() for k in dir_keys_env.split(',') if k.strip()] if dir_keys_env else None
    deltas = [float(x) for x in os.environ.get('AAA_GATE0_DELTAS', '0.025,0.05,0.10').split(',')]  # 开放点#3
    signs = [+1.0, -1.0]
    repeats = int(os.environ.get('AAA_GATE0_REPEATS', '8'))  # 每 condition 采样数（开放点#1/4）
    contact_thr = float(os.environ.get('AAA_GATE0_CONTACT_THR', '1.0'))  # 开放点#2 foot contact force 阈值(N)
    max_event_steps = int(os.environ.get('AAA_GATE0_MAX_EVENT_STEPS', '80'))  # ~1.5 stride @50Hz 上限
    bucket_mode = os.environ.get('AAA_GATE0_BUCKET', 'stance_swing')  # 开放点#4: stance_swing | quartile | none
    sign_acc_thr = float(os.environ.get('AAA_GATE0_SIGN_ACC_THR', '0.75'))  # 开放点#5
    effect_thr_mult = float(os.environ.get('AAA_GATE0_EFFECT_THR_MULT', '3.0'))  # 开放点#5
    noise_floor = float(os.environ.get('AAA_GATE0_NOISE_FLOOR', '0.0017'))  # 诊断D motion13 range 作噪声底

    flags.im_eval = True  # body_pos 进 extras（同 sweep）
    load_path = runner.load_path
    print(f"[AAA Gate-0] restore from {load_path}")
    player = runner.create_player()
    _debug("stage=player_created")
    player.restore(load_path)
    player.is_determenistic = True
    player.env.is_tensor_obses = True

    # bypass adapter：style_alpha=0 → mu = base mu（设计稿 §0：测 plant 对 action 偏移响应，非 adapter 非线性）
    sa = player.model.a2c_network.style_alpha
    sa.data.fill_(0.0)
    print(f"[AAA Gate-0] forced style_alpha = 0 (bypass adapter, was {sa.item() if sa.numel()==1 else '?'})")

    env_task = player.env.task
    device = player.device
    num_envs = env_task.num_envs
    num_actions = env_task.num_actions
    if num_envs > 1:
        player.has_batch_dimension = True
    _debug(f"stage=env_ready num_envs={num_envs} num_actions={num_actions}")

    # dof_names & direction（开放点#1 动态匹配）
    dof_names = env_task._dof_names if hasattr(env_task, '_dof_names') else None
    if dof_names is None:
        # 兜底：从 robot cfg 读
        dof_names = cfg.get('env', {}).get('dof_names', [])
    directions = build_directions(list(dof_names), num_actions)
    if dir_keys is not None:
        directions = {k: v for k, v in directions.items() if k in dir_keys}
    if not directions:
        raise RuntimeError(f"[Gate-0] no direction built from dof_names={dof_names}")
    print(f"[AAA Gate-0] dof_names={list(dof_names)}")
    print(f"[AAA Gate-0] directions={ {k: v[0] for k, v in directions.items()} }")
    _debug(f"dof_names={list(dof_names)} directions={list(directions.keys())}")

    # step_width body & foot body（复用 env 已建好的 _aaa_cplus_step_width_body_ids）
    step_body_ids = env_task._aaa_cplus_step_width_body_ids
    step_axis = env_task._aaa_cplus_step_axis
    body_names = env_task._body_names
    foot_keys = ['left_ankle', 'right_ankle', 'L_Ankle', 'R_Ankle']
    foot_body_ids = [body_names.index(n) for n in body_names
                     if any(k in n for k in foot_keys)]
    if len(foot_body_ids) < 2:
        # 兜底用 step_width body（同 ankle）
        foot_body_ids = [int(step_body_ids[0]), int(step_body_ids[1])]
    foot_body_ids = foot_body_ids[:2]
    print(f"[AAA Gate-0] step_body_ids={step_body_ids.tolist()} axis={step_axis} foot_body_ids={foot_body_ids}")

    # --- condition 分配到 env pair（baseline + pert 同初态）---
    # 每 condition 用 2 env（0=baseline, 1=pert）。condition = (motion, dir_key, sign, delta)
    conds = []
    for mid in motion_ids:
        for dk in directions.keys():
            for sign in signs:
                for d in deltas:
                    conds.append((mid, dk, sign, d))
    n_cond = len(conds)
    n_envs_needed = 2 * n_cond
    if n_envs_needed > num_envs:
        print(f"[AAA Gate-0] ⚠️ conditions({n_cond})×2={n_envs_needed} > num_envs({num_envs}), 截断到 {num_envs//2} conditions")
        n_cond = num_envs // 2
        conds = conds[:n_cond]
        n_envs_needed = 2 * n_cond
    # env_cond[env_idx] = (env_idx, is_baseline, cond_idx)
    env_cond = []
    for ci in range(n_cond):
        env_cond.append((2 * ci, True, ci))     # baseline
        env_cond.append((2 * ci + 1, False, ci))  # pert
    # 同 condition 的 baseline 与 pert 必须 same motion → motion per env 由 condition 定
    cond_motion = {ci: conds[ci][0] for ci in range(n_cond)}
    # onset phase：baseline/pert 同 reset → 同 start_time（同初态）

    # --- 每次只测一个 delta/dir/sign 组合的 motion 集合？不——不同 condition motion 可能不同 ---
    # 简化：同一批 condition 若 motion 混合，reset 时各 env 按自己 condition 的 motion 设 _sampled_motion_ids。
    # 但同 pair baseline/pert 必须同 motion（已保证）。不同 pair 可不同 motion。

    # 收集样本
    samples = []  # list of dict per (cond, repeat)

    # offset tensor 需按 condition 建（每 condition 一个 delta/dir/sign），但一次 rollout 多 condition 并行
    # → offset tensor per env 由其 condition 决定。重复 repeats 次 reset。
    for rep in range(repeats):
        _debug(f"stage=repeat {rep}/{repeats}")
        # 设每个 env 的 motion（按 condition）
        for ei, is_base, ci in env_cond:
            env_task._sampled_motion_ids[ei] = cond_motion[ci]
        # reset（随机 start_time → 随机 onset phase；事后按 start_time 分桶）
        obs_dict = player.env_reset()
        if not isinstance(obs_dict, dict):
            obs_dict = {'obs': obs_dict}
        if 'obs' not in obs_dict:
            obs_dict = {'obs': obs_dict[list(obs_dict)[0]]}
        # 记录 onset phase（per env，用 motion_length 归一）
        onset_phase = {}
        with torch.no_grad():
            for ei, is_base, ci in env_cond:
                mid = cond_motion[ci]
                mlen = float(env_task._motion_lib.get_motion_lengths()[mid]) if hasattr(env_task._motion_lib, 'get_motion_lengths') else 1.0
                # motion_lib.get_motion_lengths 返回 tensor[num_unique]；按 mid 索引
                try:
                    mlen_t = env_task._motion_lib.get_motion_lengths()
                    mlen = float(mlen_t[mid])
                except Exception:
                    mlen = 1.0
                st = float(env_task._motion_start_times[ei])
                onset_phase[ei] = (st / mlen) % 1.0 if mlen > 0 else 0.0

        # onset 时 perturbed-side（用左脚索引 foot_body_ids[0]）contact 状态 → stance/swing 分桶
        # contact_forces 在 reset 后可用
        cf = env_task._contact_forces
        onset_contact = _foot_contact(cf, foot_body_ids)  # (N, 2) per env

        # 建 offset tensor：每 env 按其 condition 的 (dir, sign, delta)
        offset = torch.zeros(num_envs, num_actions, device=device)
        for ei, is_base, ci in env_cond:
            if is_base:
                continue
            _, dk, sign, d = conds[ci]
            dof_idx, dof_sign = directions[dk]
            vec = torch.zeros(num_actions, device=device)
            for j, di in enumerate(dof_idx):
                vec[di] = float(dof_sign[j])
            nrm = vec.norm()
            if nrm > 0:
                vec = vec / nrm
            offset[ei] = sign * d * vec

        # 安装 hook（每 rep 重装，offset 变了）
        install_action_offset_hook(player, offset)

        # rollout 到 gait event：pert env 检测 foot[0]（perturbed-side）strike；
        # baseline env(2*ci) 与 pert env(2*ci+1) 同 condition 同 motion 同一批 reset，但 PHC reset 是
        # per-env 独立采样 _motion_start_times → 两者 start_time 不同（不同 onset phase）。
        # ⚠️ v1 限制：非严格同初态，Δsw 含 phase 噪声；靠 repeats 平均 + stance/swing 分桶聚合缓解。
        #   严格同初态需 reset 后 clone pert env sim state 到 baseline env（isaacgym set_actor_state），
        #   工程重留 v2。Gate-0 v1 目标=测"有无符号响应"，sign accuracy 对 phase 噪声更鲁棒。
        #   每步存 baseline env step_width 轨迹，T_event 定后从轨迹取同帧 sw_base。
        # strike 定义：onset 时若 stance → 先等 liftoff（contact<thr）再等 strike（contact>thr）；
        #              onset 时若 swing → 直接等 strike。max_event_steps 上限。
        pert_envs = [ei for ei, is_base, ci in env_cond if not is_base]
        T_event = {ci: None for ci in range(n_cond)}
        onset_f0 = {ei: float(onset_contact[ei, 0]) for ei in pert_envs}
        # 状态机：0=等 liftoff（若 onset stance）, 1=等 strike
        state = {ei: (1 if onset_f0[ei] < contact_thr else 0) for ei in pert_envs}
        fallen = {ei: False for ei in pert_envs}
        sw_pert = {ci: None for ci in range(n_cond)}
        base_sw_traj = {ci: [] for ci in range(n_cond)}  # baseline env(=2*ci) 每步 step_width
        with torch.no_grad():
            for step in range(max_event_steps):
                action = player.get_action(obs_dict, is_determenistic=True)
                obs_dict, r, done, info = player.env_step(player.env, action)
                if not isinstance(obs_dict, dict):
                    obs_dict = {'obs': obs_dict}
                bp = env_task._rigid_body_pos
                cf = env_task._contact_forces
                sw_all = _step_width_from_body_pos(bp, step_body_ids, step_axis)  # (num_envs,)
                contact = _foot_contact(cf, foot_body_ids)  # (N, 2)
                # 存 baseline env step_width（env 2*ci，与 pert env 2*ci+1 同初态同 start_time）
                for ci in range(n_cond):
                    base_sw_traj[ci].append(float(sw_all[2 * ci]))
                # 检查 pert env 的 gait event
                for ei in pert_envs:
                    ci = (ei - 1) // 2  # pert env = 2*ci+1
                    if T_event[ci] is not None:
                        continue  # 已定
                    if done[ei] > 0.5:
                        fallen[ei] = True
                        T_event[ci] = step + 1  # 提前终止，记 fall
                        sw_pert[ci] = float(sw_all[ei])
                        continue
                    f0 = float(contact[ei, 0])
                    if state[ei] == 0:  # 等 liftoff
                        if f0 < contact_thr:
                            state[ei] = 1
                    else:  # 等 strike
                        if f0 >= contact_thr:
                            T_event[ci] = step + 1
                            sw_pert[ci] = float(sw_all[ei])
                # 所有 pert 都定 T_event 后停（baseline 轨迹已同步采到 T_event 帧）
                if all(T_event[ci] is not None for ci in range(n_cond)):
                    break

        # sw_base[ci] = baseline env 在 T_event[ci] 帧的 step_width（v1 非严格同初态，见上注）
        sw_base = {}
        for ci in range(n_cond):
            te = T_event[ci]
            if te is not None and 0 < te <= len(base_sw_traj[ci]):
                sw_base[ci] = base_sw_traj[ci][te - 1]
            else:
                sw_base[ci] = None

        # 收本 rep 样本
        for ci in range(n_cond):
            mid, dk, sign, d = conds[ci]
            ei_p = 2 * ci + 1
            te = T_event[ci]
            samples.append({
                'motion_id': mid, 'dir': dk, 'sign': sign, 'delta': d, 'repeat': rep,
                'onset_phase': onset_phase[ei_p],
                'onset_stance': bool(onset_f0[ei_p] >= contact_thr),
                'T_event': te,
                'fall': fallen[ei_p],
                'sw_pert': sw_pert[ci],
                'sw_base': sw_base[ci],
            })


    # --- 聚合 ---
    def bucket_of(s):
        if bucket_mode == 'none':
            return 'all'
        if bucket_mode == 'stance_swing':
            return 'stance' if s['onset_stance'] else 'swing'
        # quartile
        p = s['onset_phase']
        return f'q{int(p * 4)}'

    # 按 (motion, dir, bucket) 聚合 sign accuracy / effect size / δ 线性性
    summary = {}
    for mid in motion_ids:
        for dk in directions.keys():
            for bk in sorted({bucket_of(s) for s in samples}):
                grp = [s for s in samples if s['motion_id'] == mid and s['dir'] == dk and bucket_of(s) == bk and not s['fall']
                       and s['sw_pert'] is not None and s['sw_base'] is not None]
                if not grp:
                    continue
                # sign accuracy: +δ → Δsw>0, −δ → Δsw<0（假设 direction 正向 = step_width 增）
                correct = 0; total = 0
                dsws = []
                for s in grp:
                    dsw = s['sw_pert'] - s['sw_base']
                    dsws.append((s['delta'], s['sign'], dsw))
                    expect = 1 if s['sign'] > 0 else -1
                    if (dsw > 0 and expect > 0) or (dsw < 0 and expect < 0):
                        correct += 1
                    total += 1
                sign_acc = correct / total if total > 0 else 0.0
                # effect size: mean |Δsw(+δ)| / mean sw_base，与噪声底比
                pos = [abs(d) for _, sg, d in dsws if sg > 0]
                neg = [abs(d) for _, sg, d in dsws if sg < 0]
                mean_dsw = np.mean([abs(d) for _, _, d in dsws])
                mean_base = np.mean([s['sw_base'] for s in grp])
                effect = mean_dsw / mean_base if mean_base > 0 else 0.0
                effect_vs_noise = mean_dsw / noise_floor if noise_floor > 0 else float('inf')
                # δ 线性性：Δsw vs δ 的斜率（+sign 拟合）
                pos_d = [(dlt, dsw) for dlt, sg, dsw in dsws if sg > 0]
                slope = None
                if len(pos_d) >= 2:
                    xs = np.array([dlt for dlt, _ in pos_d]); ys = np.array([dsw for _, dsw in pos_d])
                    slope = float(np.polyfit(xs, ys, 1)[0]) if xs.std() > 0 else 0.0
                key = f"m{mid}_{dk}_{bk}"
                summary[key] = {
                    'n': total, 'sign_acc': round(sign_acc, 3),
                    'effect': round(effect, 4), 'effect_vs_noise': round(effect_vs_noise, 2),
                    'mean_dsw': round(float(mean_dsw), 5), 'mean_base': round(float(mean_base), 4),
                    'slope_dsw_per_delta': round(slope, 4) if slope is not None else None,
                    'sign_acc_thr': sign_acc_thr, 'effect_thr_mult': effect_thr_mult,
                }

    # 裁决分支（设计稿 §7）
    def verdict(s):
        if s is None:
            return 'NO_DATA'
        if s['sign_acc'] >= sign_acc_thr and s['effect_vs_noise'] >= effect_thr_mult:
            return 'ACTION_RESCUABLE'  # → masked local response proxy
        return 'WEAK_OR_NO_RESPONSE'   # 视多数 → structured / 换维度

    for k, s in summary.items():
        s['verdict'] = verdict(s)

    # --- 落盘 + 打印 ---
    out_dir = player.config['network_path']
    os.makedirs(out_dir, exist_ok=True)
    out_json = osp.join(out_dir, 'gate0_response.json')
    result = {
        'config': {
            'motion_ids': motion_ids, 'directions': list(directions.keys()),
            'deltas': deltas, 'repeats': repeats, 'contact_thr': contact_thr,
            'max_event_steps': max_event_steps, 'bucket_mode': bucket_mode,
            'noise_floor': noise_floor, 'dof_names': list(dof_names),
        },
        'samples': samples,
        'summary': summary,
    }
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[AAA Gate-0] dumped {out_json}")

    print("=" * 90)
    print(f"[AAA Gate-0] summary  (noise_floor={noise_floor}, sign_acc_thr={sign_acc_thr}, effect_thr×noise={effect_thr_mult})")
    print(f"{'key':<32}{'n':>4}{'sign_acc':>10}{'effect':>9}{'×noise':>8}{'slope':>9}  verdict")
    for k, s in summary.items():
        sl = s['slope_dsw_per_delta'] if s['slope_dsw_per_delta'] is not None else float('nan')
        print(f"{k:<32}{s['n']:>4}{s['sign_acc']:>10.3f}{s['effect']:>9.4f}{s['effect_vs_noise']:>8.2f}{sl:>9.4f}  {s['verdict']}")
    print("=" * 90)
    # 整体裁决
    rescuable = [k for k, s in summary.items() if s['verdict'] == 'ACTION_RESCUABLE']
    if rescuable:
        print(f"[AAA Gate-0] 整体：部分 direction 稳定响应 → 走 masked local response proxy: {rescuable}")
    elif summary:
        print("[AAA Gate-0] 整体：无稳定响应 → 考虑 structured residual 或换维度（elbow_bend）/ 升级 foot-placement 注入")
    else:
        print("[AAA Gate-0] 整体：无有效样本（检查 contact_thr / max_event_steps / dof_names）")
    print("[AAA Gate-0] done.")
    _debug("stage=gate0_done")
    return result
# === AAA end ===
