"""High-level observations (paper Table 5): pose, goal and base velocity, with the pose from GT or from SLAM.

``pose_source="gt"``: ground-truth pose in the start frame (training condition A). ``pose_source="slam"``: the
ORB-SLAM3 pose that ``Ros2Manager`` puts in ``env.orb_slam_res`` (also relative to the start pose).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import wrap_to_pi

from .commands import GoalCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "base_lin_vel_xy",
    "base_yaw_rate",
    "goal_heading_error",
    "goal_position",
    "robot_pose",
    "slam_quality",
]


def robot_pose(env: ManagerBasedRLEnv, command_name: str = "goal", pose_source: str = "gt") -> torch.Tensor:
    """(x, y, yaw) in the start frame, shape (N, 3)."""
    if pose_source == "gt":
        cmd: GoalCommand = env.command_manager.get_term(command_name)
        return cmd.gt_pose_start_frame()
    if pose_source == "slam":
        # cuVSLAM (CuVslamManager) if the task runs it, else ORB-SLAM3 (Ros2Manager)
        res = getattr(env, "slam_res", None) or getattr(env, "orb_slam_res", None)
        if res is None:
            return torch.zeros(env.num_envs, 3, device=env.device)
        return res["pose_xyyaw"]
    raise ValueError(f"Unknown pose_source '{pose_source}' (gt | slam)")


def goal_position(env: ManagerBasedRLEnv, command_name: str = "goal", pose_source: str = "gt") -> torch.Tensor:
    """Goal minus robot position, both in the start frame, shape (N, 2)."""
    cmd: GoalCommand = env.command_manager.get_term(command_name)
    return cmd.command - robot_pose(env, command_name, pose_source)[:, :2]


def goal_heading_error(env: ManagerBasedRLEnv, command_name: str = "goal", pose_source: str = "gt") -> torch.Tensor:
    """Angle from the robot's heading to the goal direction, in [-pi, pi], shape (N, 1)."""
    pose = robot_pose(env, command_name, pose_source)
    delta = env.command_manager.get_term(command_name).command - pose[:, :2]
    return wrap_to_pi(torch.atan2(delta[:, 1], delta[:, 0]) - pose[:, 2]).unsqueeze(-1)


def base_lin_vel_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_lin_vel_b.torch[:, :2]


def base_yaw_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_ang_vel_b.torch[:, 2:3]


FEATURES_SCALE = 500.0
"""Tracked features per frame are 190-440 in the Map A / B runs (docs/PAPER_OUTLINE.md)."""
GAP_SCALE = 5.0
"""Metres between the odometry and the loop-closure pose that count as the largest value."""
FEATURES_EMA = 0.25
"""Weight of the newest frame in the feature average (about the last 10 high-level steps)."""


def slam_quality(env: ManagerBasedRLEnv) -> torch.Tensor:
    """cuVSLAM quality signals, shape (N, 3), all in [0, 1]: tracked features, the same averaged over about 10 steps,
    and the gap between the odometry and the loop-closure pose (0 while there is no loop-closure pose).

    Measured on Map B drives (scripts/tools/analyze_slam_quality.py): the gap tells when the pose is wrong (AUROC 0.88),
    the averaged feature count tells when it is about to drift (0.91); ``status`` is always 1 and is not used."""
    res = getattr(env, "slam_res", None)
    if res is None or "num_features" not in res:
        return torch.zeros(env.num_envs, 3, device=env.device)
    features = (res["num_features"].flatten() / FEATURES_SCALE).clamp(0.0, 1.0)
    ema = getattr(env, "_slam_features_ema", None)
    if ema is None or ema.shape != features.shape:
        ema = features.clone()
    fresh = env.episode_length_buf <= 1  # first step of an episode: start the average from the new tracker
    ema = torch.where(fresh, features, (1.0 - FEATURES_EMA) * ema + FEATURES_EMA * features)
    env._slam_features_ema = ema
    gap = torch.nan_to_num((res["odom_xyyaw"][:, :2] - res["slam_xyyaw"][:, :2]).norm(dim=1), nan=0.0)
    return torch.stack([features, ema, (gap / GAP_SCALE).clamp(0.0, 1.0)], dim=1)
