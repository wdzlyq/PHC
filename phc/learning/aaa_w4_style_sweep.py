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
    # === AAA W4: slider ladder（每个 env 一种配置）===
    # env0=style0(zeros), env1=energy=1, env2=step_width=1, env3=elbow_bend=1, env4=全1
    # 其余槽位（vert_bob/cadence/trunk_sway）置 0（W4 起步 3 维，抉择7）
    ladder = torch.zeros(5, slider_dim, device=device)
    ladder[1, 0] = 1.0  # energy=1
    ladder[2, 1] = 1.0  # step_width=1
    ladder[3, 2] = 1.0  # elbow_bend=1
    ladder[4, :3] = 1.0  # 全 1
    labels = ['style0', 'energy=1', 'step_width=1', 'elbow_bend=1', 'all=1']
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

    # ⚠️ body_pos 仅在 flags.im_eval=True 时进 extras（humanoid_im.py:700），故 sweep 期间强制开。
    # cfg.im_eval 由 run_hydra.main 依 cfg 设 flags.im_eval；这里再兜底确保 True。
    flags.im_eval = True

    load_path = runner.load_path
    print(f"[AAA W4 sweep] restore from {load_path}")
    player = runner.create_player()
    player.restore(load_path)
    player.is_determenistic = True
    player.env.is_tensor_obses = True  # IM 训练用 tensor obs

    env_task = player.env.task
    device = player.device
    num_envs = env_task.num_envs
    slider_dim = 6
    print(f"[AAA W4 sweep] num_envs={num_envs} slider_dim={slider_dim}")

    # ⚠️ has_batch_dimension 默认 False（rl_games 在 run() 内才设），sweep 不走 run() 故手动设：
    #   num_envs>1 时 obs 带 batch 维，否则 get_action 的 unsqueeze_obs 把 (N,obs) 变 (1,N,obs) 致 _preproc_obs cat 崩。
    if num_envs > 1:
        player.has_batch_dimension = True

    # 选固定 motion（验收内容）。默认 motion_id=0，可由 cfg.aaa_w4_motion_key 覆盖。
    motion_key = cfg.get('aaa_w4_motion_key', None)
    motion_id = cfg.get('aaa_w4_motion_id', None)
    mid, mname = _pick_motion(env_task, motion_key=motion_key, motion_id=motion_id)
    print(f"[AAA W4 sweep] fixed motion id={mid} key={mname}")

    # slider ladder（5 种）。num_envs 可能 >>5，把 ladder broadcast 到前 5 个 env，
    # 其余 env 用 style0（不参与对比，仅占位保持 batch 维）。
    ladder, labels = _build_slider_ladder(slider_dim, device)
    slider_override = torch.zeros(num_envs, slider_dim, device=device)
    n_ladder = min(len(ladder), num_envs)
    slider_override[:n_ladder] = ladder[:n_ladder]
    player._eval_slider_override = slider_override  # player get_action 注入用（commit ee10c3a）

    # 固定所有 env 到同一 motion（IM 模式 _sampled_motion_ids 是 env→motion 映射，直接覆写）
    # ⚠️ 取模约定见 w4 patch §8.7：_style_table 索引要 %num_unique，但 _sampled_motion_ids 直接赋 motion_id
    #    即可（motion_lib 内部 %num_unique）。赋值后 reset 让 ref state 按该 motion 初始化。
    env_task._sampled_motion_ids[:] = mid
    obs_dict = player.env_reset()
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

    pred_traj = [[] for _ in range(n_ladder)]  # 每 ladder env 一条轨迹
    with torch.no_grad():
        for n in range(max_steps):
            action = player.get_action(obs_dict, is_determenistic=True)
            obs_dict, r, done, info = player.env_step(player.env, action)
            if not isinstance(obs_dict, dict):
                obs_dict = {'obs': obs_dict}
            # info 即 env extras；body_pos 是当前 sim body 世界坐标 (num_envs, num_bodies, 3)
            body_pos = info.get('body_pos', None)
            if body_pos is not None:
                bp = body_pos if torch.is_tensor(body_pos) else torch.as_tensor(body_pos)
                for i in range(n_ladder):
                    pred_traj[i].append(bp[i].detach().cpu().numpy())
            # 不调 _post_step：它会按 138 motion 切片 + forward_motion_samples + 落 failed.pkl，
            # 与"固定单 motion sweep"冲突。sweep 只需 body_pos，已直接从 info 取。
            if done.sum() > 0:
                # 某 env 提前 terminate（fall），其余继续；保持 batch 维，已存轨迹不再追加空帧
                pass

    # 落盘 + 指标
    out_dir = player.config['network_path']
    os.makedirs(out_dir, exist_ok=True)
    sweep_out = osp.join(out_dir, 'w4_style_sweep.pkl')
    result = {
        'motion_id': mid, 'motion_key': mname,
        'slider_labels': labels[:n_ladder],
        'slider_ladder': ladder[:n_ladder].cpu().numpy(),
        'pred_traj': [np.stack(t) for t in pred_traj],  # list of (T, num_bodies, 3)
    }
    joblib.dump(result, sweep_out, compress=True)
    print(f"[AAA W4 sweep] dumped {sweep_out}")

    # 客观指标表
    print("=" * 78)
    print(f"[AAA W4 sweep] metrics (motion={mname})  — 各 slider 列指标应有方向性差异")
    print(f"{'slider':<16}{'root_z_std':>12}{'step_width':>14}{'elbow_rom':>12}{'root_speed':>14}")
    for i in range(n_ladder):
        mp = _joint_amplitude_metrics(result['pred_traj'][i])
        print(f"{labels[i]:<16}{mp['root_z_std']:>12.4f}{mp['step_width']:>14.4f}{mp['elbow_rom']:>12.2f}{mp['root_speed']:>14.4f}")
    print("=" * 78)
    print("[AAA W4 sweep] done.")
    return result
# === AAA end ===
