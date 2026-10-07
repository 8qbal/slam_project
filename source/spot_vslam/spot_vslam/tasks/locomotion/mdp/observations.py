"""Depth-camera observations of the low-level policy (paper Table 2)."""

import torch
import torch.nn.functional as F

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

__all__ = ["camera_depth_obs", "camera_depth_stats"]


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

    60x80 -> 15x20 = 300 dims (paper); 48x64 -> 12x16 = 192 dims (rl_vslam_v0).
    """
    depth = _get_clean_depth(env, sensor_cfg)  # (N, H, W)
    depth = depth.unsqueeze(1)  # (N, 1, H, W)
    pooled_depth = F.avg_pool2d(depth, kernel_size=4, stride=4)
    normalized_depth = (pooled_depth.reshape(env.num_envs, -1) / 5.0) * 2.0 - 1.0
    return normalized_depth


def camera_depth_stats(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """5 depth summaries: left / center / right mean, front minimum, left-right balance.

    The regions are written for a 60x80 image (on 48x64 they are silently cropped, as in rl_vslam_v0).
    """
    depth = _get_clean_depth(env, sensor_cfg)  # (N, H, W)

    left = depth[:, 20:50, 5:25].mean(dim=(1, 2))
    center = depth[:, 20:50, 25:55].mean(dim=(1, 2))
    right = depth[:, 20:50, 55:75].mean(dim=(1, 2))

    front_patch = depth[:, 20:50, 25:55].reshape(env.num_envs, -1)
    front_min = front_patch.min(dim=1).values
    lr_balance = left - right

    stats = torch.stack([left, center, right, front_min, lr_balance], dim=1)

    # the four depths to [-1, 1] over 0-5 m; the left-right difference scaled and clipped
    stats[:, :4] = (stats[:, :4] / 5.0) * 2.0 - 1.0
    stats[:, 4] = torch.clamp(stats[:, 4] / 5.0, min=-1.0, max=1.0)
    return stats
