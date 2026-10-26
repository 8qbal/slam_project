"""Depth-camera observations of the low-level policy (paper Table 2)."""

import torch
import torch.nn.functional as F

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

__all__ = ["camera_depth_obs", "camera_depth_stats", "noisy_camera_depth_obs"]


def _get_clean_depth(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Depth image in metres with NaN/inf replaced and clamped to 0-5 m, shape (N, H, W)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output["distance_to_image_plane"].torch.clone()
    depth = torch.nan_to_num(depth, nan=5.0, posinf=5.0, neginf=5.0)
    depth = torch.clamp(depth, min=0.0, max=5.0)
    depth = depth.squeeze(-1)  # (N, H, W)
    return depth


def camera_depth_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Depth image average-pooled 4x4 and flattened, normalized to [-1, 1].

    60x80 -> 15x20 = 300 dims (paper).
    """
    return _pooled_depth(_get_clean_depth(env, sensor_cfg))


def _pooled_depth(depth: torch.Tensor) -> torch.Tensor:
    pooled_depth = F.avg_pool2d(depth.unsqueeze(1), kernel_size=4, stride=4)  # (N, 1, H/4, W/4)
    normalized_depth = (pooled_depth.reshape(depth.shape[0], -1) / 5.0) * 2.0 - 1.0
    return normalized_depth


def camera_depth_stats(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """5 depth summaries: left / center / right mean, front minimum, left-right balance.

    The regions are written for a 60x80 image.
    """
    return _depth_stats(_get_clean_depth(env, sensor_cfg))


def _depth_stats(depth: torch.Tensor) -> torch.Tensor:
    left = depth[:, 20:50, 5:25].mean(dim=(1, 2))
    center = depth[:, 20:50, 25:55].mean(dim=(1, 2))
    right = depth[:, 20:50, 55:75].mean(dim=(1, 2))

    front_patch = depth[:, 20:50, 25:55].reshape(depth.shape[0], -1)
    front_min = front_patch.min(dim=1).values
    lr_balance = left - right

    stats = torch.stack([left, center, right, front_min, lr_balance], dim=1)

    # the four depths to [-1, 1] over 0-5 m; the left-right difference scaled and clipped
    stats[:, :4] = (stats[:, :4] / 5.0) * 2.0 - 1.0
    stats[:, 4] = torch.clamp(stats[:, 4] / 5.0, min=-1.0, max=1.0)
    return stats


def noisy_camera_depth_obs(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    range_coeff: float = 0.01,
    dropout_max: float = 0.05,
    edge_threshold: float = 0.3,
    edge_dropout: float = 0.5,
) -> torch.Tensor:
    """``camera_depth_obs`` and ``camera_depth_stats`` (305 dims, same layout) from one depth image with real-camera
    errors, as stereo / RGB-D cameras show them (``Spot-Paper-WallsSoftNoise-v0``):

    - range noise: Gaussian with std ``range_coeff * z**2`` (1 cm at 1 m, 25 cm at 5 m),
    - missing pixels: each env drops a random share in [0, ``dropout_max``] of its pixels every step,
    - edges: pixels next to a depth jump above ``edge_threshold`` m are dropped with probability ``edge_dropout``.

    Dropped pixels read 5 m, as NaN (no return) does in ``_get_clean_depth``. The noise only applies while the policy
    group's ``enable_corruption`` is on, so evaluation scripts that turn it off see the clean image.
    """
    depth = _get_clean_depth(env, sensor_cfg)  # (N, H, W)
    if env.cfg.observations.policy.enable_corruption:
        dropout = torch.rand(depth.shape[0], 1, 1, device=depth.device) * dropout_max
        drop = torch.rand_like(depth) < dropout
        # edges from the clean image, else far-range noise reads as edges everywhere
        jump = torch.zeros_like(depth, dtype=torch.bool)
        dx = (depth[:, :, 1:] - depth[:, :, :-1]).abs() > edge_threshold
        dy = (depth[:, 1:, :] - depth[:, :-1, :]).abs() > edge_threshold
        jump[:, :, 1:] |= dx
        jump[:, :, :-1] |= dx
        jump[:, 1:, :] |= dy
        jump[:, :-1, :] |= dy
        drop |= jump & (torch.rand_like(depth) < edge_dropout)
        depth = depth + torch.randn_like(depth) * range_coeff * depth**2
        depth = torch.where(drop, torch.full_like(depth, 5.0), depth).clamp(0.0, 5.0)
    return torch.cat([_pooled_depth(depth), _depth_stats(depth)], dim=1)
