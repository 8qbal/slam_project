"""Subscribes to every env's ORB-SLAM3 outputs (pose, tracking state, local point cloud)."""

import functools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32

from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class OrbSlamSubscriberCfg:
    """Config for the ORB-SLAM3 subscriber manager. Topics are ``<prefix><suffix>``, e.g. /spot0/orbslam/robot_pose."""

    topic_prefix_template: str = "/spot{i}/orbslam"
    pose_topic_suffix: str = "/robot_pose"
    state_topic_suffix: str = "/tracking_state"
    local_pc_topic_suffix: str = "/local_pc"


class OrbSlamSubscriberManager:
    """Subscribes to each env's ORB-SLAM3 topics.

    The ROS callbacks write to CPU (NumPy) caches; ``update()`` copies them to GPU tensors in one batch per step.
    """

    def __init__(self, cfg: OrbSlamSubscriberCfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self.env = env
        self.num_envs = self.env.num_envs
        self.device = self.env.device

        # share Ros2Manager's node when there is one
        try:
            self.node = self.env.ros2_manager.node
        except AttributeError:
            if not rclpy.ok():
                rclpy.init()
            self.node = rclpy.create_node("isaaclab_slam_subscriber_node")

        # GPU buffers. Pose is [x, y, z, qx, qy, qz, qw].
        self.pose_buffer = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        self.state_buffer = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        self.local_pc_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

        # CPU caches written by the callbacks
        self._pose_cache = np.zeros((self.num_envs, 7), dtype=np.float32)
        self._pose_cache[:, 6] = 1.0
        self._state_cache = np.zeros((self.num_envs,), dtype=np.int32)
        self._pc_count_cache = np.zeros((self.num_envs,), dtype=np.int32)

        self.subs: list[rclpy.subscription.Subscription] = []
        # best effort: only the latest message matters
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.node.get_logger().info(f"Creating ORB-SLAM3 subscribers (env count: {self.num_envs})...")

        for i in range(self.num_envs):
            topic_prefix = self.cfg.topic_prefix_template.format(i=i)
            self.subs.append(
                self.node.create_subscription(
                    PoseStamped,
                    topic_prefix + self.cfg.pose_topic_suffix,
                    functools.partial(self._pose_callback, env_id=i),
                    qos_profile,
                )
            )
            self.subs.append(
                self.node.create_subscription(
                    Int32,
                    topic_prefix + self.cfg.state_topic_suffix,
                    functools.partial(self._state_callback, env_id=i),
                    qos_profile,
                )
            )
            self.subs.append(
                self.node.create_subscription(
                    PointCloud2,
                    topic_prefix + self.cfg.local_pc_topic_suffix,
                    functools.partial(self._local_pc_callback, env_id=i),
                    qos_profile,
                )
            )

    def _pose_callback(self, msg: PoseStamped, env_id: int):
        p = msg.pose.position
        q = msg.pose.orientation
        self._pose_cache[env_id] = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    def _state_callback(self, msg: Int32, env_id: int):
        self._state_cache[env_id] = int(msg.data)

    def _local_pc_callback(self, msg: PointCloud2, env_id: int):
        self._pc_count_cache[env_id] = len(msg.data) // msg.point_step if msg.point_step > 0 else 0

    def update(self, dt: float):
        """Copy the CPU caches to the GPU buffers (called every env step)."""
        self.pose_buffer[:] = torch.from_numpy(self._pose_cache).to(self.device)
        self.state_buffer[:] = torch.from_numpy(self._state_cache).to(self.device)
        self.local_pc_count[:] = torch.from_numpy(self._pc_count_cache).to(self.device)

    def reset(self, env_ids: Sequence[int]):
        """Clear the buffers and caches of the reset envs (else the next update() writes the old values back)."""
        self.pose_buffer[env_ids] = 0.0
        self.pose_buffer[env_ids, 6] = 1.0
        self.state_buffer[env_ids] = 0
        self.local_pc_count[env_ids] = 0

        ids_np = env_ids.cpu().numpy() if isinstance(env_ids, torch.Tensor) else np.array(env_ids)
        self._pose_cache[ids_np] = 0.0
        self._pose_cache[ids_np, 6] = 1.0
        self._state_cache[ids_np] = 0
        self._pc_count_cache[ids_np] = 0
        return {}

    def close(self):
        self.node.get_logger().info("Shutting down ORB-SLAM Subscriber...")
        for sub in self.subs:
            self.node.destroy_subscription(sub)
