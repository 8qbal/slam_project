"""ORB-SLAM3 outputs of every env on the GPU (pose, tracking state, local point cloud size).

A reader of ``Ros2Manager``'s subscriptions, not a second subscriber: ``Ros2Manager`` owns the ROS 2 node, the topic
names and the spinning, so one set of ``orbslam/*`` subscriptions serves ``env.orb_slam_res`` and these buffers.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class OrbSlamSubscriberCfg:
    """Opt-in marker (``cfg.slam_subscriber``): the env also fills the buffers of :class:`OrbSlamSubscriberManager`.

    Topics come from ``Ros2ManagerCfg`` (``spot_vslam.slam.ros2_manager``).
    """


class OrbSlamSubscriberManager:
    """GPU buffers for each env's ORB-SLAM3 outputs, copied from ``Ros2Manager``'s callback caches.

    Needs ``env.ros2_manager`` (it is updated first in the env step, so the caches are fresh); ``update()`` copies them
    to the GPU in one batch per step.
    """

    def __init__(self, cfg: OrbSlamSubscriberCfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self.env = env
        self.num_envs = self.env.num_envs
        self.device = self.env.device

        self.ros2_manager = getattr(self.env, "ros2_manager", None)
        if self.ros2_manager is None:
            raise RuntimeError(
                "OrbSlamSubscriberManager needs a Ros2Manager: set cfg.ros2 as well as cfg.slam_subscriber."
            )
        self.node = self.ros2_manager.node
        self.ros2_manager.enable_local_pc()

        # GPU buffers. Pose is [x, y, z, qx, qy, qz, qw].
        self.pose_buffer = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        self.pose_buffer[:, 6] = 1.0
        self.state_buffer = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        self.local_pc_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

    def update(self, dt: float):
        """Copy the ``Ros2Manager`` caches to the GPU buffers (called every env step)."""
        pose, state, pc_count = self.ros2_manager.orb_snapshot()
        self.pose_buffer[:] = torch.from_numpy(pose).to(self.device)
        self.state_buffer[:] = torch.from_numpy(state).to(self.device)
        self.local_pc_count[:] = torch.from_numpy(pc_count).to(self.device)

    def reset(self, env_ids: Sequence[int]):
        """Clear the buffers of the reset envs (``Ros2Manager.reset``, called first, clears their source caches)."""
        ids = torch.as_tensor(
            np.asarray(env_ids.cpu() if isinstance(env_ids, torch.Tensor) else env_ids), device=self.device
        )
        self.pose_buffer[ids] = 0.0
        self.pose_buffer[ids, 6] = 1.0
        self.state_buffer[ids] = 0
        self.local_pc_count[ids] = 0
        return {}

    def close(self):
        # the subscriptions belong to Ros2Manager, which destroys the node
        pass
