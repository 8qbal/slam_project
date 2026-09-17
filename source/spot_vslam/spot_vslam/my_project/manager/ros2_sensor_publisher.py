# ros2_sensor_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
from isaaclab.managers import ObservationTerm

class ROS2MultiSpotPub(ObservationTerm):
    def __init__(self, cfg, env, node: Node):
        super().__init__(cfg, env)
        self.bridge = CvBridge()
        self.num_spots = getattr(cfg, "num_spots", 1)
        self.node = node

        self.rgb_pubs = {}
        self.depth_pubs = {}
        self.pc_pubs = {}
        self.state_pubs = {}

        for i in range(1, self.num_spots + 1):
            self.rgb_pubs[i] = self.node.create_publisher(Image, f'/spot{i}/camera/image_raw', 10)
            self.depth_pubs[i] = self.node.create_publisher(Image, f'/spot{i}/camera/depth', 10)
            self.pc_pubs[i] = self.node.create_publisher(PointCloud2, f'/spot{i}/orbslam/local_pc', 10)
            self.state_pubs[i] = self.node.create_publisher(String, f'/spot{i}/orbslam/tracking_state', 10)

    def compute(self) -> torch.Tensor:
        obs = self._env.observation_manager.compute()

        for i in range(1, self.num_spots + 1):
            rgb_tensor = obs[f"spot{i}_rgb"].cpu().numpy()
            depth_tensor = obs[f"spot{i}_depth"].cpu().numpy()
            pc_tensor = obs.get(f"spot{i}_pc", None)
            state_str = obs.get(f"spot{i}_state", "UNKNOWN")

            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_tensor, encoding="rgb8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_tensor, encoding="32FC1")
            self.rgb_pubs[i].publish(rgb_msg)
            self.depth_pubs[i].publish(depth_msg)

            if pc_tensor is not None:
                self.pc_pubs[i].publish(pc_tensor)

            state_msg = String()
            state_msg.data = state_str
            self.state_pubs[i].publish(state_msg)

        return torch.tensor([])

    def update(self):
        rclpy.spin_once(self.node, timeout_sec=0.001)
