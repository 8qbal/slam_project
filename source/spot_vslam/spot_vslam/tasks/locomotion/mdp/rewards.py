"""Reward terms of the low-level policy.

``visual_wall_penalty`` and ``open_space_seeker_reward`` are the depth terms of the rl_vslam_v0 policy; the other
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
]


# -- rl_vslam_v0 depth terms


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
