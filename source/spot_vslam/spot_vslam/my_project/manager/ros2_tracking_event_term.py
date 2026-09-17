# ros2_tracking_event_term.py
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from isaaclab.managers import EventTermBase, EventTermCfg

class ROSTrackingEventTerm(EventTermBase):
    def __init__(self, cfg: EventTermCfg, env, node: Node):
        super().__init__(cfg, env)
        self.node = node
        self.state = "OK"
        self.sub = self.node.create_subscription(String, cfg.topic, self.cb, 10)

    def cb(self, msg: String):
        self.state = msg.data

    def reset_on_event(self, env_ids):
        if self.state == "LOST":
            return torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

    def update(self):
        rclpy.spin_once(self.node, timeout_sec=0.001)

class ROSTrackingEventTermCfg(EventTermCfg):
    class_type = ROSTrackingEventTerm
    topic: str = "/spot1/orbslam/tracking_state"
