# === AAA W4: 第二步「3-5 风格可区分」验收 harness（w4 patch §8.8） ===
# 为什么：im_eval 的 run()/_post_step 按 138 motion 切片（get_motion_num_steps 逐 motion 切 pred_pos），
#   与"固定单 motion + 多 slider"不兼容，故不复用 im_eval run()，写轻量自定义驱动。
#   目的：固定内容（同一 motion），切 _eval_slider_override，肉眼/客观见风格变（energy high→大动作 /
#   step_width wide→宽步 / elbow_bend high→屈肘）。前置 player 注入已通（commit ee10c3a）。
# 做什么：runner 已 build_alg_runner+load+reset 就绪后调用 run_sweep(runner, cfg)。
#   ① 复用 im_eval gate（可选）：env.task.style_labels vs _eval_slider_override=zeros 对比。
#   ② controlled sweep：固定单 motion + slider ladder rollout，落盘 pred_pos + 关节幅度指标。
#
# ⚠️ 仅作验收脚本，不进训练路径；用 AAA W4 注释规范包裹。
import os
import os.path as osp
import sys

import numpy as np
import torch
import joblib


def _build_structured_direction(env_task, direction_name):
    # === AAA Step B: fixed structured direction for sweep ===
    # 为什么：Gate-0 v2 已证明 full D2 与 D2_axis0 有稳定 step_width response；Step B 需要在真实
    #   slider sweep 中验证持续结构化偏置是否被 PHC tracking 抵消。
    # 做什么：复用 Gate-0 的方向构造，返回 L2-normalized action direction，避免把 D2 语义重新硬编码。
    #   fallback：若 gate0 build_directions 不含该方向名（如 D2_axis0_pelvis_sway 是 network buffer 名，
    #   gate0 只暴露 D0/D1/D2_pelvis_sway/D3），直接用 network 的 aaa_structured_direction buffer
    #   （= D2_axis0 = L_Hip_x +/R_Hip_x -，与 C1/C3 训练时同一 buffer，保证 Step B↔C3 可比）。
    try:
        from learning.gate0_action_response import build_directions
    except Exception:
        from gate0_action_response import build_directions

    dof_names = list(env_task._dof_names)
    dirs = build_directions(dof_names, env_task.num_actions)
    if direction_name in dirs:
        idx, signs = dirs[direction_name]
        vec = torch.zeros(env_task.num_actions, device=env_task.device)
        for j, di in enumerate(idx):
            vec[di] = float(signs[j])
        nrm = vec.norm()
        if nrm > 0:
            vec = vec / nrm
        return vec, idx, signs
    # fallback: 用 network 的 aaa_structured_direction buffer（D2_axis0）
    net = getattr(env_task, '_aaa_net_ref', None)
    if net is None:
        # player 注入 hook 时 env_task 上可能没 net ref；尝试从 player 拿
        pass
    buf = None
    try:
        buf = env_task._aaa_structured_direction_buffer  # 由 run_sweep 在 hook 前注入
    except Exception:
        buf = None
    if buf is None:
        raise RuntimeError(f"structured direction {direction_name} not found in gate0 dirs "
                           f"(available={list(dirs.keys())}) and no network buffer fallback injected")
    vec = buf.to(env_task.device).clone()
    nrm = vec.norm()
    if nrm > 0:
        vec = vec / nrm
    # idx/signs 仅用于打印，从非零位置反推
    nz = torch.nonzero(vec, as_tuple=False).flatten().tolist()
    idx = nz
    signs = [float(vec[i]) for i in nz]
    return vec, idx, signs


def _foot_body_ids(env_task):
    body_names = env_task._body_names
    foot_keys = ['left_ankle', 'right_ankle', 'L_Ankle', 'R_Ankle']
    ids = [body_names.index(n) for n in body_names if any(k in n for k in foot_keys)]
    if len(ids) < 2 and hasattr(env_task, '_aaa_cplus_step_width_body_ids'):
        ids = [int(env_task._aaa_cplus_step_width_body_ids[0]), int(env_task._aaa_cplus_step_width_body_ids[1])]
    return ids[:2]


def install_structured_step_hook(player, env_task, slider_override, direction_vec, amp, origin, phase_gate, contact_thr):
    # === AAA Step B: fixed mapping mu offset ===
    # 为什么：不训练 69D residual，先验证 Gate-0 找到的 D2/D2_axis0 结构化方向能否在真实 sweep 中
    #   产生可见且 content 不崩的 step_width 控制。
    # 做什么：b = amp*(slider_step_width-origin)，每帧在 actor mu 上加 phase_gate*b*direction_vec。
    foot_ids = _foot_body_ids(env_task)

    def phase_scale():
        if phase_gate == 'none':
            return torch.ones(env_task.num_envs, 1, device=env_task.device)
        contact = torch.norm(env_task._contact_forces[:, foot_ids, :], dim=-1)
        if phase_gate == 'stance_left':
            active = contact[:, 0] >= contact_thr
        elif phase_gate == 'stance_any':
            active = (contact >= contact_thr).any(dim=-1)
        elif phase_gate == 'swing_left':
            active = contact[:, 0] < contact_thr
        else:
            raise RuntimeError(f"unknown phase_gate={phase_gate}")
        return active.float().unsqueeze(-1)

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
            'slider': slider_override,
        }
        with torch.no_grad():
            res_dict = player.model(input_dict)
        mu = res_dict['mus']
        bias = amp * (slider_override[:, 1:2] - origin)
        mu = mu + phase_scale() * bias * direction_vec.view(1, -1)
        current_action = mu
        if player.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
        if player.clip_actions:
            from phc.learning.amp_players import rescale_actions
            return rescale_actions(player.actions_low, player.actions_high, torch.clamp(current_action, -1.0, 1.0))
        return current_action

    player.get_action = patched_get_action


def _pick_motion(env_task, motion_key=None, motion_id=None):
    # === AAA W4: 选定固定内容 motion（验收要"同一 motion 切 slider"）===
    # motion_lib._motion_data_keys 是 np.array[str]，下标 = motion_id（0..num_unique-1）
    keys = env_task._motion_lib._motion_data_keys
    if motion_key is not None:
        mid = int(np.where(keys == motion_key)[0][0])
    elif motion_id is not None:
        mid = int(motion_id)
    else:
        mid = 0
    return mid, str(keys[mid])


def _build_slider_ladder(slider_dim, device):
    # === AAA W5: C+ step_width-only canary ladder ===
    # 为什么：W4 多维 ladder 在 slider↔motion 混淆下不可诊断；C+ 第一刀只验 step_width 单维，
    #   固定内容 motion 后切 target step_width=0/0.25/0.5/0.75/1，看 actual step_width 是否单调。
    # 做什么：仅设置 slider 第1维（step_width），其余维度为 0，占位但不作为 C+ reward 目标。
    vals = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
    ladder = torch.zeros(vals.shape[0], slider_dim, device=device)
    ladder[:, 1] = vals
    labels = [f'step_width={float(v):.2f}' for v in vals]
    return ladder, labels


def _joint_amplitude_metrics(pred_pos):
    # === AAA W4: 客观风格指标（pred_pos shape (T, num_bodies, 3)，SMPL_MUJOCO 24 body）===
    # body id（SMPL_MUJOCO_NAMES）：Pelvis=0, L_Ankle=3, R_Ankle=7,
    #   L_Shoulder=15, L_Elbow=16, R_Shoulder=21, R_Elbow=22。
    # 指标设计（粗粒度，证 slider 切换→方向性变化，不必精确物理量）：
    #   - root_z_std：Pelvis z 标准差 → 垂直起伏/能量 proxy
    #   - step_width：左右脚踝横向(x)距离均值 → 步宽 proxy
    #   - elbow_rom：双肘关节角（shoulder→elbow 向量与 elbow→wrist 向量夹角）变化幅度 → 屈肘 proxy
    #   - action_speed：root xy 位移速度 → 能量 proxy（互补 root_z_std）
    PELVIS, L_ANK, R_ANK = 0, 3, 7
    L_SHO, L_ELB, R_SHO, R_ELB = 15, 16, 21, 22
    L_WRI, R_WRI = 17, 23  # wrist: L=17, R=23

    root_z_std = float(pred_pos[:, PELVIS, 2].std())
    step_width = float(np.abs(pred_pos[:, L_ANK, 0] - pred_pos[:, R_ANK, 0]).mean())

    def joint_angle(shoulder, elbow, wrist):
        u = pred_pos[:, shoulder] - pred_pos[:, elbow]   # elbow→shoulder
        v = pred_pos[:, wrist] - pred_pos[:, elbow]      # elbow→wrist
        dot = (u * v).sum(-1)
        n = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-8
        ang = np.degrees(np.arccos(np.clip(dot / n, -1, 1)))
        return ang
    ang_l = joint_angle(L_SHO, L_ELB, L_WRI)
    ang_r = joint_angle(R_SHO, R_ELB, R_WRI)
    elbow_rom = float((ang_l.max() - ang_l.min() + ang_r.max() - ang_r.min()) / 2)

    # root xy 速度（位移/帧）
    root_xy = pred_pos[:, PELVIS, :2]
    speed = float(np.linalg.norm(np.diff(root_xy, axis=0), axis=-1).mean())

    return {
        'root_z_std': root_z_std,
        'step_width': step_width,
        'elbow_rom': elbow_rom,
        'root_speed': speed,
    }


def run_sweep(runner, cfg):
    # === AAA W4: controlled sweep 主入口（w4 patch §8.8 ②）===
    # 为什么：固定单 motion + slider ladder，self-consistent rollout，证 slider 切换→风格变。
    from learning.im_amp_players import IMAMPPlayerContinuous  # noqa
    from phc.utils.flags import flags

    debug_path = os.environ.get('AAA_SWEEP_DEBUG_PATH')
    def _debug(msg):
        if debug_path:
            with open(debug_path, 'a') as f:
                f.write(str(msg) + "\n")
                f.flush()

    _debug("stage=enter_run_sweep")

    # ⚠️ body_pos 仅在 flags.im_eval=True 时进 extras（humanoid_im.py:700），故 sweep 期间强制开。
    # cfg.im_eval 由 run_hydra.main 依 cfg 设 flags.im_eval；这里再兜底确保 True。
    flags.im_eval = True

    load_path = runner.load_path
    print(f"[AAA W4 sweep] restore from {load_path}")
    player = runner.create_player()
    _debug("stage=player_created")
    player.restore(load_path)
    _debug("stage=player_restored")
    player.is_determenistic = True
    player.env.is_tensor_obses = True  # IM 训练用 tensor obs

    # === AAA W4 probe: 强制 α（monkeypatch style_alpha）测"放大 α 是否产生 slider-dependent 风格"===
    # 为什么：disc-only 证实 disc 有杠杆（α 增长、r_disc 升），但可能只是 tracking correction 非 style。
    #   故强制把 α 拉到大幅值，sweep 看 slider 切换是否改变动作指标——若变→架构能产风格(方案A)，不变→residual 不编码风格(方案B/C)。
    force_alpha = cfg.get('aaa_w4_force_alpha', None)
    if force_alpha is not None:
        sa = player.model.a2c_network.style_alpha
        sa.data.fill_(float(force_alpha))
        print(f"[AAA W4 sweep] ⚠️ forced style_alpha = {float(force_alpha)} (was {sa.item() if sa.numel()==1 else '?'})")

    env_task = player.env.task
    device = player.device
    num_envs = env_task.num_envs
    slider_dim = 6
    print(f"[AAA W4 sweep] num_envs={num_envs} slider_dim={slider_dim}")
    _debug(f"stage=env_ready num_envs={num_envs}")

    # ⚠️ has_batch_dimension 默认 False（rl_games 在 run() 内才设），sweep 不走 run() 故手动设：
    #   num_envs>1 时 obs 带 batch 维，否则 get_action 的 unsqueeze_obs 把 (N,obs) 变 (1,N,obs) 致 _preproc_obs cat 崩。
    if num_envs > 1:
        player.has_batch_dimension = True

    # 选固定 motion（验收内容）。默认 motion_id=0，可由 cfg.aaa_w4_motion_key 覆盖。
    motion_key = cfg.get('aaa_w4_motion_key', None)
    motion_id = cfg.get('aaa_w4_motion_id', None)
    mid, mname = _pick_motion(env_task, motion_key=motion_key, motion_id=motion_id)
    print(f"[AAA W4 sweep] fixed motion id={mid} key={mname}")
    _debug(f"stage=motion_picked mid={mid} key={mname}")

    # slider ladder（5 种）。num_envs 可能 >>5，把 ladder broadcast 到前 5 个 env，
    # 其余 env 用 style0（不参与对比，仅占位保持 batch 维）。
    ladder, labels = _build_slider_ladder(slider_dim, device)
    slider_override = torch.zeros(num_envs, slider_dim, device=device)
    n_ladder = min(len(ladder), num_envs)
    slider_override[:n_ladder] = ladder[:n_ladder]
    player._eval_slider_override = slider_override  # player get_action 注入用（commit ee10c3a）
    _debug("stage=slider_override_set")

    structured_step = bool(cfg.get('aaa_structured_step_sweep', False))
    structured_info = None
    if structured_step:
        # === AAA Step B: bypass learned adapter for fixed structured canary ===
        # 为什么：Step B 验证结构化 action direction 本身，不混入已知 flat/不稳的 learned residual。
        # 做什么：style_alpha=0 退化 base actor，再由 install_structured_step_hook 加固定 D2_axis0 偏置。
        player.model.a2c_network.style_alpha.data.fill_(0.0)
        direction_name = cfg.get('aaa_structured_direction', 'D2_axis0_pelvis_sway')
        amp = float(cfg.get('aaa_structured_amp', 0.2))
        origin = float(cfg.get('aaa_structured_origin', 0.5))
        phase_gate = cfg.get('aaa_structured_phase_gate', 'stance_left')
        contact_thr = float(cfg.get('aaa_structured_contact_thr', 1.0))
        # 注入 network 的 aaa_structured_direction buffer（D2_axis0）作 fallback，供 _build_structured_direction 使用
        # 为什么：gate0 build_directions 不暴露 D2_axis0（只有 D2_pelvis_sway 全轴）；Step B 必须用与 C1/C3
        #   训练时同一的 D2_axis0 buffer（L_Hip_x +/R_Hip_x -）保证可比。
        if hasattr(player.model, 'a2c_network') and hasattr(player.model.a2c_network, 'aaa_structured_direction'):
            env_task._aaa_structured_direction_buffer = player.model.a2c_network.aaa_structured_direction.detach().cpu()
        direction_vec, direction_idx, direction_signs = _build_structured_direction(env_task, direction_name)
        install_structured_step_hook(player, env_task, slider_override, direction_vec, amp, origin, phase_gate, contact_thr)
        structured_info = {
            'direction': direction_name,
            'direction_idx': list(map(int, direction_idx)),
            'direction_signs': [float(x) for x in direction_signs],
            'amp': amp,
            'origin': origin,
            'phase_gate': phase_gate,
            'contact_thr': contact_thr,
        }
        print(f"[AAA structured sweep] enabled {structured_info}")

    # 固定所有 env 到同一 motion（IM 模式 _sampled_motion_ids 是 env→motion 映射，直接覆写）
    # ⚠️ 取模约定见 w4 patch §8.7：_style_table 索引要 %num_unique，但 _sampled_motion_ids 直接赋 motion_id
    #    即可（motion_lib 内部 %num_unique）。赋值后 reset 让 ref state 按该 motion 初始化。
    env_task._sampled_motion_ids[:] = mid
    _debug("stage=before_env_reset")
    obs_dict = player.env_reset()
    _debug("stage=after_env_reset")
    # ⚠️ env_reset 可能返回 dict {'obs':...} 或裸 tensor（取决于 wrapper），统一成 dict 供 get_action
    if not isinstance(obs_dict, dict):
        obs_dict = {'obs': obs_dict}
    if 'obs' not in obs_dict:
        obs_dict = {'obs': obs_dict[list(obs_dict)[0]]}

    # rollout：跑该 motion 全长（取 motion_lib 帧数），每步收 pred body_pos
    # ⚠️ get_motion_num_steps 接 motion_ids tensor，返回 per-motion 步数（motion_lib_base.py:431）
    num_steps = int(env_task._motion_lib.get_motion_num_steps(torch.tensor([mid], device=device))[0].item())
    max_steps = min(num_steps, int(cfg.get('aaa_w4_max_steps', 300)))
    print(f"[AAA W4 sweep] motion num_steps={num_steps}, rollout max_steps={max_steps}")
    _debug(f"stage=before_rollout num_steps={num_steps} max_steps={max_steps}")

    pred_traj = [[] for _ in range(n_ladder)]  # 每 ladder env 一条轨迹
    reward_traj = [[] for _ in range(n_ladder)]
    done_count = [0 for _ in range(n_ladder)]
    # === AAA P1: 正式 content 指标逐帧捕获（doc 24 §3 P1）===
    # 为什么：codex P1 要求 MPJPE/root tracking/root speed drift/SR/fall/residual norm；现有 sweep 只存 body_pos+reward。
    #   flags.im_eval=True 时 env extras 已含 mpjpe(逐 env 标量) + body_pos_gt(ref 全身) + _terminate_buf(fall 判定)，
    #   这里只补捕获，不改 env。gt 只存 pelvis(root tracking/root speed drift 用)，省内存。
    # 做什么：每帧每 ladder env 存 mpjpe 标量 + gt_pelvis(3,) + terminate 标志；structured bias norm 离线由 amp/origin/slider 算。
    mpjpe_traj = [[] for _ in range(n_ladder)]
    gt_pelvis_traj = [[] for _ in range(n_ladder)]
    terminate_traj = [[] for _ in range(n_ladder)]
    body_pos_source = 'missing'
    with torch.no_grad():
        for n in range(max_steps):
            action = player.get_action(obs_dict, is_determenistic=True)
            obs_dict, r, done, info = player.env_step(player.env, action)
            if not isinstance(obs_dict, dict):
                obs_dict = {'obs': obs_dict}
            # info 即 env extras；body_pos 是当前 sim body 世界坐标 (num_envs, num_bodies, 3)
            body_pos = info.get('body_pos', None)
            # P1 指标来源（flags.im_eval=True 时存在，见 humanoid_im.py:872-874）
            mpjpe_t = info.get('mpjpe', None)            # (num_envs,) tensor
            body_pos_gt = info.get('body_pos_gt', None)  # (num_envs, num_bodies, 3) ndarray
            if body_pos is None:
                # === AAA W5: sweep body_pos fallback ===
                # 为什么：部分 test/sweep 路径不会把 body_pos 放进 extras，旧脚本会在 np.stack 空轨迹时报错。
                # 做什么：直接从 env task 的 rigid body tensor 取当前位置，只影响离线 sweep 验收。
                body_pos = getattr(env_task, '_rigid_body_pos', None)
                if body_pos is not None:
                    body_pos_source = '_rigid_body_pos'
                if body_pos is None:
                    rb_state = getattr(env_task, '_rigid_body_state', None)
                    if rb_state is not None:
                        body_pos = rb_state[..., :3]
                        body_pos_source = '_rigid_body_state'
            else:
                body_pos_source = 'info.body_pos'
            if body_pos is not None:
                bp = body_pos if torch.is_tensor(body_pos) else torch.as_tensor(body_pos)
                # terminate_buf：>0 表示该 env 已 fall（_terminate_buf 在 compute_humanoid_im_reset 里置位）
                term_buf = getattr(env_task, '_terminate_buf', None)
                for i in range(n_ladder):
                    pred_traj[i].append(bp[i].detach().cpu().numpy())
                    reward_traj[i].append(float(r[i]))
                    if done[i] > 0.5:
                        done_count[i] += 1
                    # P1 指标逐帧存（mpjpe 标量 + gt pelvis + terminate 标志）
                    if mpjpe_t is not None:
                        mt = mpjpe_t[i] if torch.is_tensor(mpjpe_t) else mpjpe_t[i]
                        mpjpe_traj[i].append(float(mt.detach().cpu() if torch.is_tensor(mt) else mt))
                    if body_pos_gt is not None:
                        gt_pelvis_traj[i].append(np.asarray(body_pos_gt[i][0, :3]))  # pelvis only
                    if term_buf is not None:
                        terminate_traj[i].append(int(term_buf[i].item() > 0.5))
            # 不调 _post_step：它会按 138 motion 切片 + forward_motion_samples + 落 failed.pkl，
            # 与"固定单 motion sweep"冲突。sweep 只需 body_pos，已直接从 info 取。
            if done.sum() > 0:
                # 某 env 提前 terminate（fall），其余继续；保持 batch 维，已存轨迹不再追加空帧
                pass

    if debug_path:
        with open(debug_path, 'w') as f:
            f.write(f"motion_id={mid}\n")
            f.write(f"motion_key={mname}\n")
            f.write(f"max_steps={max_steps}\n")
            f.write(f"body_pos_source={body_pos_source}\n")
            f.write("traj_lengths=" + ",".join(str(len(t)) for t in pred_traj) + "\n")

    # 落盘 + 指标
    out_dir = player.config['network_path']
    os.makedirs(out_dir, exist_ok=True)
    sweep_out = osp.join(out_dir, 'w4_style_sweep.pkl')
    result = {
        'motion_id': mid, 'motion_key': mname,
        'slider_labels': labels[:n_ladder],
        'slider_ladder': ladder[:n_ladder].cpu().numpy(),
        'pred_traj': [np.stack(t) for t in pred_traj],  # list of (T, num_bodies, 3)
        'structured_step': structured_info,
        # === AAA P1: 正式 content 指标（doc 24 §3 P1）===
        'mpjpe_traj': [np.asarray(t) for t in mpjpe_traj],          # list of (T,)
        'gt_pelvis_traj': [np.stack(t) if len(t) else np.zeros((0, 3)) for t in gt_pelvis_traj],  # list of (T,3)
        'terminate_traj': [np.asarray(t) for t in terminate_traj],   # list of (T,) int {0,1}
        'reward_traj': [np.asarray(t) for t in reward_traj],         # list of (T,)
        'motion_num_steps': int(num_steps),
        'max_steps': int(max_steps),
    }
    joblib.dump(result, sweep_out, compress=True)
    print(f"[AAA W4 sweep] dumped {sweep_out}")

    # 客观指标表
    print("=" * 78)
    print(f"[AAA W4 sweep] metrics (motion={mname})  — 各 slider 列指标应有方向性差异")
    print(f"{'slider':<16}{'root_z_std':>12}{'step_width':>14}{'elbow_rom':>12}{'root_speed':>14}{'rew_mean':>12}{'done':>8}")
    for i in range(n_ladder):
        mp = _joint_amplitude_metrics(result['pred_traj'][i])
        rew_mean = float(np.mean(reward_traj[i])) if reward_traj[i] else float('nan')
        print(f"{labels[i]:<16}{mp['root_z_std']:>12.4f}{mp['step_width']:>14.4f}{mp['elbow_rom']:>12.2f}{mp['root_speed']:>14.4f}{rew_mean:>12.4f}{done_count[i]:>8d}")
    print("=" * 78)
    print("[AAA W4 sweep] done.")
    return result
# === AAA end ===
