"""Reward terms of the low-level policy.

``visual_wall_penalty`` and ``open_space_seeker_reward`` are the depth terms of the low-level policy; the other
locomotion terms of both policies are Isaac Lab's Spot rewards. ``wall_min_depth`` is the paper's wall term
(references/main.pdf Sec. 5.1, Eq. 5).

Sign: Eq. 1 adds w_j * p_j with w_j < 0, and Eq. 5 (p_wall = -min depth) carries its own minus sign. Taken literally,
with w_wall = -0.1 the term becomes +0.1 * min depth, i.e. it rewards keeping distance from the nearest surface,
which is the stated intent, so Eq. 5 is used as written.
"""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .observations import _get_clean_depth

__all__ = [
    "visual_wall_penalty",
    "open_space_seeker_reward",
    "wall_min_depth",
    "diagonal_duty_penalty",
    "foot_duty_penalty",
    "short_swing_penalty",
]


# -- depth terms


def visual_wall_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, safe_distance: float) -> torch.Tensor:
    """Squared violation of ``safe_distance`` by the nearest of three front regions (mean of the closest 10 %)."""
    depth = _get_clean_depth(env, sensor_cfg)  # (N, H, W)

    left_front = depth[:, 20:50, 8:28].reshape(env.num_envs, -1)
    center_front = depth[:, 20:50, 22:42].reshape(env.num_envs, -1)
    right_front = depth[:, 20:50, 36:56].reshape(env.num_envs, -1)

    def robust_near_dist(region: torch.Tensor) -> torch.Tensor:
        k = max(1, int(region.shape[1] * 0.1))
        low_k, _ = torch.topk(region, k=k, dim=1, largest=False)
        return low_k.mean(dim=1)

    left_dist = robust_near_dist(left_front)
    center_dist = robust_near_dist(center_front)
    right_dist = robust_near_dist(right_front)

    near_dist = torch.minimum(torch.minimum(left_dist, center_dist), right_dist)

    violation = torch.clamp(safe_distance - near_dist, min=0.0)
    return violation**2


def open_space_seeker_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Forward speed times the normalized mean depth ahead: a small pull towards open space."""
    depth = _get_clean_depth(env, sensor_cfg)  # (N, H, W)

    front = depth[:, 20:50, 25:55]
    front_mean = front.mean(dim=(1, 2)) / 5.0

    fwd_vel = torch.clamp(env.scene["robot"].data.root_lin_vel_b.torch[:, 0], min=0.0)
    return fwd_vel * front_mean


# -- paper Eq. 5


def wall_min_depth(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Eq. 5: -min(d_depth) over the depth image (cleaned and clamped to 0-5 m like the depth observation)."""
    depth = env.scene.sensors[sensor_cfg.name].data.output["distance_to_image_plane"].torch
    depth = torch.clamp(torch.nan_to_num(depth, nan=5.0, posinf=5.0, neginf=5.0), min=0.0, max=5.0)
    return -depth.flatten(start_dim=1).min(dim=1).values


# -- gait symmetry


def diagonal_duty_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Difference in duty factor (share of the last step cycle on the ground) between the two trot diagonals.

    ``sensor_cfg.body_names`` must list the feet as (fl, fr, hl, hr). Isaac Lab's gait reward only asks one diagonal to
    be down while the other is up, so a limp (FL+HR down 75 % of the cycle, FR+HL 25 %) earns it in full; all
    paper-version policies learned that limp. 0 for an even trot, 0.5 for that limp. Only while a speed is commanded,
    like the gait reward.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    ids = [sensor.body_names.index(name) for name in sensor_cfg.body_names]
    air = sensor.data.last_air_time.torch[:, ids]
    contact = sensor.data.last_contact_time.torch[:, ids]
    duty = contact / (air + contact).clamp(min=1e-6)
    diff = ((duty[:, 0] + duty[:, 3]) - (duty[:, 1] + duty[:, 2])).abs() / 2.0
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    return torch.where(cmd > 0.0, diff, 0.0)


def foot_duty_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, target: float = 0.5) -> torch.Tensor:
    """Mean distance of each foot's duty factor (share of its last step cycle on the ground) from ``target``.

    A trot has every foot down half of the cycle. The WallsSoft limp (FL / HR 73 %, FR / HL 31 %) gives ~0.2; the 8 Hz
    four-foot shuffle of the from-scratch Sym run has ~0.5 on every foot (``short_swing_penalty`` catches that).
    Only while a speed is commanded.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    air = sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    contact = sensor.data.last_contact_time.torch[:, sensor_cfg.body_ids]
    duty = contact / (air + contact).clamp(min=1e-6)
    dev = (duty - target).abs().mean(dim=1)
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    return torch.where(cmd > 0.0, dev, 0.0)


def short_swing_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, min_air_time: float = 0.2) -> torch.Tensor:
    """At each touchdown, how much shorter than ``min_air_time`` the swing was (summed over feet).

    Without it, equal 0.04 s air / 0.08 s contact times on every foot satisfy the gait reward (the shuffle above).
    Only while a speed is commanded.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    touchdown = sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    air = sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    short = (min_air_time - air).clamp(min=0.0) * touchdown.float()
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    return torch.where(cmd > 0.0, short.sum(dim=1), 0.0)
