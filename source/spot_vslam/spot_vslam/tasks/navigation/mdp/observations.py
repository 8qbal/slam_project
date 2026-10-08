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

__all__ = ["base_lin_vel_xy", "base_yaw_rate", "goal_heading_error", "goal_position", "robot_pose"]


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
