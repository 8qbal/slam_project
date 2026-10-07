"""SLAM diagnostics read from ``OrbSlamSubscriberManager`` (zeros until it is loaded)."""

import torch

__all__ = ["base_yaw_deg", "get_slam_camera_error", "get_slam_local_pc_count", "get_slam_tracking_state"]


def get_slam_tracking_state(env) -> torch.Tensor:
    """ORB-SLAM3 tracking state per env, shape (num_envs, 1). Zeros until the SLAM subscriber is loaded."""
    manager = getattr(env, "slam_subscriber_manager", None)
    if manager is None:
        return torch.zeros((env.num_envs, 1), device=env.device)
    return manager.state_buffer.float().unsqueeze(-1)


def get_slam_local_pc_count(env) -> torch.Tensor:
    """Number of points in the ORB-SLAM3 local point cloud per env, shape (num_envs, 1)."""
    manager = getattr(env, "slam_subscriber_manager", None)
    if manager is None:
        return torch.zeros((env.num_envs, 1), device=env.device)
    return manager.local_pc_count.float().unsqueeze(-1)


def get_slam_camera_error(env):
    """Distance between the GT camera position and the ORB-SLAM3 position, shape (num_envs, 1)."""
    gt_cam_pos = env.scene.sensors["camera"].data.pos_w.torch
    if hasattr(env, "slam_subscriber_manager") and env.slam_subscriber_manager is not None:
        slam_pos = env.slam_subscriber_manager.pose_buffer[:, 0:3]
    else:
        slam_pos = torch.zeros_like(gt_cam_pos)
    dist_error = torch.norm(gt_cam_pos - slam_pos, dim=-1, keepdim=True)
    return dist_error


def base_yaw_deg(env) -> torch.Tensor:
    """Robot base yaw in degrees, shape (num_envs, 1). Isaac Lab 3.0 quaternions are (x, y, z, w)."""
    q = env.scene["robot"].data.root_quat_w.torch
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.rad2deg(yaw).unsqueeze(-1)
