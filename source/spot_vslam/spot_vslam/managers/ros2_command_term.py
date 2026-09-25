# ros2_command_term.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import torch
from isaaclab.managers import CommandTerm

class ROS2CommandTerm(CommandTerm):
    def __init__(self, cfg, env, node: Node):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 3), device=self.device)  # x, y, theta
        self.node = node
        self.sub = self.node.create_subscription(
            PoseStamped,
            '/slam/pose',
            self._ros2_callback,
            10
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _ros2_callback(self, msg: PoseStamped):
        self._command[:, 0] = msg.pose.position.x
        self._command[:, 1] = msg.pose.position.y
        self._command[:, 2] = 0.0  # 這裡可以換成 yaw

    def _update_command(self):
        rclpy.spin_once(self.node, timeout_sec=0.001)
