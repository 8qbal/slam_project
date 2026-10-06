"""Low-level reward terms written as in the paper (references/main.pdf Sec. 5.1, Eqs. 2-6).

The rl_vslam_v0 policy was trained with Isaac Lab's Spot reward functions, which differ in form from the paper's
equations (e.g. exp(-|e|/std) with a speed ramp instead of exp(-|e|^2/sigma)). These follow the equations. Where the
paper leaves a constant open (sigma, h_min, the gait gating), the value from the original config is reused.

Signs: Eq. 1 adds w_j * p_j with w_j < 0. Eq. 5 (p_wall = -min depth) and Eq. 6 (p_slip = -sum |v_foot|^2) also carry
their own minus sign. Taken literally, with w_wall = -0.1 the wall term becomes +0.1 * min depth, i.e. it rewards
keeping distance from the nearest surface, which is the stated intent, so Eq. 5 is used as written. Taken literally,
Eq. 6 with w_slip = -0.5 would *reward* foot slip, so p_slip returns the positive sum and the negative weight makes
it a penalty.
"""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def base_linear_velocity_exp(
    env: ManagerBasedRLEnv, sigma: float, command_name: str = "base_velocity", asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """Eq. 2: exp(-|v_xy - v_xy^cmd|^2 / sigma)."""
    vel_xy = env.scene[asset_cfg.name].data.root_lin_vel_b.torch[:, :2]
    err_sq = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_xy), dim=1)
    return torch.exp(-err_sq / sigma)


def gait_phase_indicator(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    velocity_threshold: float,
    command_name: str = "base_velocity",
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Eq. 3: prod_k 1[foot_k in correct phase], for a trot.

    ``sensor_cfg.body_names`` must list the feet as (fl, hr, fr, hl) with ``preserve_order=True``. A foot is in the
    correct phase when its contact state equals its diagonal partner's and differs from the other diagonal pair's,
    so the product is 1 exactly when the contacts form a trot pattern. As in the original GaitReward, the term is
    only active when a velocity is commanded or the body moves faster than ``velocity_threshold``.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0
    fl, hr, fr, hl = contact.unbind(dim=1)
    in_phase = (fl == hr) & (fr == hl) & (fl != fr)

    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(env.scene[asset_cfg.name].data.root_lin_vel_b.torch[:, :2], dim=1)
    active = (cmd > 0.0) | (body_vel > velocity_threshold)
    return (in_phase & active).float()


def foot_clearance_hinge(env: ManagerBasedRLEnv, h_min: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Eq. 4: sum_k max(0, h_k - h_min), h_k = foot height."""
    foot_z = env.scene[asset_cfg.name].data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    return torch.sum(torch.clamp(foot_z - h_min, min=0.0), dim=1)


def wall_min_depth(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Eq. 5: -min(d_depth) over the depth image (cleaned and clamped to 0-5 m like the depth observation)."""
    depth = env.scene.sensors[sensor_cfg.name].data.output["distance_to_image_plane"].torch
    depth = torch.clamp(torch.nan_to_num(depth, nan=5.0, posinf=5.0, neginf=5.0), min=0.0, max=5.0)
    return -depth.flatten(start_dim=1).min(dim=1).values


def foot_contact_velocity_sq(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Eq. 6 (magnitude): sum_k |v_foot,k|^2 over the feet in contact."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    in_contact = torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0] > threshold
    foot_vel = env.scene[asset_cfg.name].data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :]
    return torch.sum(torch.sum(torch.square(foot_vel), dim=-1) * in_contact, dim=1)
