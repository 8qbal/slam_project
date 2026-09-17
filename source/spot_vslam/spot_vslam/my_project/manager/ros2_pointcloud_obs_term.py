# ros2_pointcloud_obs_term.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import numpy as np
import torch

class ROS2ObservationTerm:
    """訂閱 ROS2 topic 並轉成 observation buffer，支援多台 Spot"""
    def __init__(self, node: Node, num_spots: int, device="cpu"):
        self.node = node
        self.num_spots = num_spots
        self.device = device
        self.buffers = [{"local_pc": None, "tracking_state": None} for _ in range(num_spots)]
        self.subs = []

        for i in range(num_spots):
            spot_prefix = f"/spot{i+1}"
            local_pc_topic = f"{spot_prefix}/orbslam/local_pc"
            state_topic = f"{spot_prefix}/orbslam/tracking_state"

            self.subs.append(self.node.create_subscription(
                PointCloud2,
                local_pc_topic,
                lambda msg, idx=i: self._callback(msg, idx, "local_pc"),
                10
            ))
            self.subs.append(self.node.create_subscription(
                String,
                state_topic,
                lambda msg, idx=i: self._callback(msg, idx, "tracking_state"),
                10
            ))

    def _callback(self, msg, env_id, key):
        self.buffers[env_id][key] = msg  # 這裡可再加上轉換成 numpy

    def compute(self):
        obs = {"local_pc": [], "tracking_state": []}
        for buf in self.buffers:
            local_pc_tensor = torch.zeros(128, device=self.device) if buf["local_pc"] is None else torch.from_numpy(np.zeros(128)).to(self.device)
            state_tensor = torch.zeros(16, device=self.device) if buf["tracking_state"] is None else torch.from_numpy(np.zeros(16)).to(self.device)
            obs["local_pc"].append(local_pc_tensor)
            obs["tracking_state"].append(state_tensor)
        obs["local_pc"] = torch.stack(obs["local_pc"], dim=0)
        obs["tracking_state"] = torch.stack(obs["tracking_state"], dim=0)
        return obs

    def update(self):
        rclpy.spin_once(self.node, timeout_sec=0.001)
