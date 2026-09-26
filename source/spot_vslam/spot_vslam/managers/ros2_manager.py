import math
import time
import threading
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32, Empty
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import Camera


def numpy_to_imgmsg(array: np.ndarray, encoding: str) -> Image:
    """Build a ``sensor_msgs/Image`` from a HxW or HxWxC array.

    Replaces ``cv_bridge.CvBridge.cv2_to_imgmsg``: the ROS 2 Jazzy ``cv_bridge`` binary is built
    against NumPy 1.x and segfaults under the NumPy 2.x that Isaac Sim / Isaac Lab 3.0 require.
    """
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    array = np.ascontiguousarray(array)
    msg = Image()
    msg.height, msg.width = array.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = int(array.dtype.byteorder == ">")
    msg.step = array.strides[0]
    msg.data = array.tobytes()
    return msg


@dataclass
class Ros2ManagerCfg:
    camera_name: str
    topic_prefix: str = "/spot"
    frame_id: str = "camera_link"
    base_frame_id: str = "base_link"
    odom_frame_id: str = "odom"


class Ros2Manager:
    def __init__(self, cfg: Ros2ManagerCfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self.env = env
        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.orb_initialized = False

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("isaaclab_ros2_manager")

        # 發布節流：目標 30 FPS
        self.base_start_time = time.time()
        self.frame_count = 0
        self.target_fps = 30
        self.fixed_step = 1.0 / self.target_fps
        self.last_pub_wall_time = 0.0

        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self.node)

        self.node.get_logger().info(f"ROS 2 Manager Ready. Target: {self.target_fps} FPS (Reliable).")

        try:
            self.camera: Camera = self.env.scene.sensors[self.cfg.camera_name]
            self.camera_info = CameraInfo()
        except KeyError:
            self.node.get_logger().error(f"找不到相機感測器 '{self.cfg.camera_name}'")
            raise

        self.intrinsic_matrices = None
        self.rgb_pubs = {}
        self.depth_pubs = {}
        self.rgb_info_pubs = {}
        self.depth_info_pubs = {}

        # ORB subscriber
        self.pose_subs = {}
        self.state_subs = {}
        self.orb_reset_pubs = {}
        self._orb_lock = threading.Lock()

        # GPU tensors 給 env 直接讀
        self.orb_pose_xyyaw = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.orb_status = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

        # CPU 緩存，callback 寫這裡
        self._orb_pose_xyyaw_cpu = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._orb_status_cpu = np.zeros((self.num_envs, 1), dtype=np.float32)

        # TF frame 前綴，例如 spot_0, spot_1
        clean_prefix_base = self.cfg.topic_prefix.strip("/")
        self.frame_prefixes = [f"{clean_prefix_base}_{i}" for i in range(self.num_envs)]

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        reset_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 發布靜態 TF
        self._publish_static_tf()

        # 相機 publishers + ORB reset publishers
        for i in range(self.num_envs):
            prefix = f"{self.cfg.topic_prefix}{i}"
            self.rgb_pubs[i] = self.node.create_publisher(Image, f"{prefix}/camera/image_raw", qos_profile)
            self.rgb_info_pubs[i] = self.node.create_publisher(CameraInfo, f"{prefix}/camera/camera_info", qos_profile)
            self.depth_pubs[i] = self.node.create_publisher(Image, f"{prefix}/camera/depth", qos_profile)
            self.depth_info_pubs[i] = self.node.create_publisher(CameraInfo, f"{prefix}/camera/depth/camera_info", qos_profile)

            # 新增：發給 ORB-SLAM3 的 reset topic
            self.orb_reset_pubs[i] = self.node.create_publisher(
                Empty,
                f"{prefix}/orbslam/reset",
                reset_qos_profile,
            )

        # ORB-SLAM3 subscribers
        self._create_orb_subscribers(qos_profile)

        # 先放一份空資料進 env，避免外部第一次讀不到
        self.env.orb_slam_res = {
            "pose_xyyaw": self.orb_pose_xyyaw,
            "status": self.orb_status,
        }

    def _publish_static_tf(self):
        """發布 base_link -> camera_link 的靜態轉換"""
        static_transforms = []
        stamp = self.node.get_clock().now().to_msg()

        cam_offset = self.camera.cfg.offset

        for i in range(self.num_envs):
            prefix = self.frame_prefixes[i]

            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = f"{prefix}/{self.cfg.base_frame_id}"
            t.child_frame_id = f"{prefix}/{self.cfg.frame_id}"

            t.transform.translation.x = float(cam_offset.pos[0])
            t.transform.translation.y = float(cam_offset.pos[1])
            t.transform.translation.z = float(cam_offset.pos[2])

            # Isaac Lab quaternion order: (x, y, z, w)
            t.transform.rotation.x = float(cam_offset.rot[0])
            t.transform.rotation.y = float(cam_offset.rot[1])
            t.transform.rotation.z = float(cam_offset.rot[2])
            t.transform.rotation.w = float(cam_offset.rot[3])

            static_transforms.append(t)

        self.tf_static_broadcaster.sendTransform(static_transforms)

    def _create_orb_subscribers(self, qos_profile: QoSProfile):
        """
        為每個 env 訂閱：
          /spot{i}/orbslam/robot_pose
          /spot{i}/orbslam/tracking_state
        """
        for i in range(self.num_envs):
            prefix = f"{self.cfg.topic_prefix}{i}"
            pose_topic = f"{prefix}/orbslam/robot_pose"
            state_topic = f"{prefix}/orbslam/tracking_state"

            self.pose_subs[i] = self.node.create_subscription(
                PoseStamped,
                pose_topic,
                lambda msg, env_id=i: self._pose_callback(msg, env_id),
                qos_profile,
            )

            self.state_subs[i] = self.node.create_subscription(
                Int32,
                state_topic,
                lambda msg, env_id=i: self._state_callback(msg, env_id),
                qos_profile,
            )

            self.node.get_logger().info(
                f"Subscribed ORB topics for env {i}: {pose_topic}, {state_topic}"
            )

    def _quat_xyzw_to_yaw(self, x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _pose_callback(self, msg: PoseStamped, env_id: int):
        """
        讀 /spot{i}/orbslam/robot_pose
        存成 [x, y, yaw]
        """
        px = float(msg.pose.position.x)
        py = float(msg.pose.position.y)

        qx = float(msg.pose.orientation.x)
        qy = float(msg.pose.orientation.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)

        yaw = self._quat_xyzw_to_yaw(qx, qy, qz, qw)

        with self._orb_lock:
            self._orb_pose_xyyaw_cpu[env_id, 0] = px
            self._orb_pose_xyyaw_cpu[env_id, 1] = py
            self._orb_pose_xyyaw_cpu[env_id, 2] = yaw

    def _state_callback(self, msg: Int32, env_id: int):
        """
        C++ mapping:
          0 = INIT
          1 = TRACKING
          2 = LOST

        Python 這裡轉成：
          1.0 = tracking ok
          0.0 = not ok
        """
        status = 1.0 if int(msg.data) == 1 else 0.0

        with self._orb_lock:
            self._orb_status_cpu[env_id, 0] = status

    def _sync_orb_results_to_env(self):
        """
        把 callback 收到的 CPU 緩存同步到 GPU tensor，
        並掛到 env.orb_slam_res 給外部直接讀。
        """
        with self._orb_lock:
            pose_cpu = self._orb_pose_xyyaw_cpu.copy()
            status_cpu = self._orb_status_cpu.copy()

        self.orb_pose_xyyaw = torch.tensor(pose_cpu, device=self.device, dtype=torch.float32)
        self.orb_status = torch.tensor(status_cpu, device=self.device, dtype=torch.float32)

        self.env.orb_slam_res = {
            "pose_xyyaw": self.orb_pose_xyyaw,
            "status": self.orb_status,
        }

    def publish_orb_reset(self, env_ids=None):
        """
        發送 /spot{i}/orbslam/reset 給 ORB-SLAM3 node。
        """
        msg = Empty()

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().numpy().tolist()
        elif not isinstance(env_ids, list):
            env_ids = list(env_ids)

        for i in env_ids:
            if i in self.orb_reset_pubs:
                self.orb_reset_pubs[i].publish(msg)
                self.node.get_logger().warn(f"Published ORB reset for env {i}")

    def update(self, dt: float):

        if not self.orb_initialized:
            self.publish_orb_reset()
            self.orb_initialized = True
        # 1. 限速發布
        current_wall_time = time.time()
        if (current_wall_time - self.last_pub_wall_time) < self.fixed_step:
            # 即使這一輪不發影像，也要 spin 一下收 ORB topic
            rclpy.spin_once(self.node, timeout_sec=0.0)
            self._sync_orb_results_to_env()
            return

        self.last_pub_wall_time = current_wall_time
        self.frame_count += 1

        # 2. 計算固定時間戳，讓 VSLAM 端時間連續
        perfect_timestamp = self.base_start_time + (self.frame_count * self.fixed_step)
        ros_time_sec = int(perfect_timestamp)
        ros_time_nanosec = int((perfect_timestamp - ros_time_sec) * 1e9)
        current_time = rclpy.time.Time(seconds=ros_time_sec, nanoseconds=ros_time_nanosec)
        stamp = current_time.to_msg()

        try:
            rgb_tensors = self.camera.data.output["rgb"].torch
            depth_tensors = None
            available_keys = self.camera.data.output.keys()

            if "distance_to_image_plane" in available_keys:
                depth_tensors = self.camera.data.output["distance_to_image_plane"].torch
            elif "depth" in available_keys:
                depth_tensors = self.camera.data.output["depth"].torch

            if rgb_tensors is None:
                rclpy.spin_once(self.node, timeout_sec=0.0)
                self._sync_orb_results_to_env()
                return

        except Exception:
            rclpy.spin_once(self.node, timeout_sec=0.0)
            self._sync_orb_results_to_env()
            return

        clean_frame_ids = []
        base_frame_ids = []
        odom_frame_ids = []

        for i in range(self.num_envs):
            prefix = self.frame_prefixes[i]
            clean_frame_ids.append(f"{prefix}/{self.cfg.frame_id}")
            base_frame_ids.append(f"{prefix}/{self.cfg.base_frame_id}")
            odom_frame_ids.append(f"{prefix}/{self.cfg.odom_frame_id}")

        self._publish_odom_tf(odom_frame_ids, base_frame_ids, stamp)
        self.publish_data(rgb_tensors, depth_tensors, clean_frame_ids, stamp)

        # 收 ROS topic
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self._sync_orb_results_to_env()

    def _publish_odom_tf(self, odom_frames, base_frames, stamp):
        robot = self.env.scene["robot"]
        root_pos = robot.data.root_pos_w.torch
        root_quat = robot.data.root_quat_w.torch

        transforms = []
        for i in range(self.num_envs):
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = odom_frames[i]
            t.child_frame_id = base_frames[i]

            t.transform.translation.x = float(root_pos[i, 0])
            t.transform.translation.y = float(root_pos[i, 1])
            t.transform.translation.z = float(root_pos[i, 2])

            # Isaac Lab quat order: (x, y, z, w)
            t.transform.rotation.x = float(root_quat[i, 0])
            t.transform.rotation.y = float(root_quat[i, 1])
            t.transform.rotation.z = float(root_quat[i, 2])
            t.transform.rotation.w = float(root_quat[i, 3])

            transforms.append(t)

        self.tf_broadcaster.sendTransform(transforms)

    def publish_data(self, rgb_tensors, depth_tensors, frame_ids, stamp):
        if self.intrinsic_matrices is None:
            self._compute_intrinsics()

        for i in range(self.num_envs):
            # RGB
            rgb_np = rgb_tensors[i].cpu().numpy()
            if rgb_np.shape[-1] == 4:
                rgb_np = rgb_np[..., :3]

            rgb_msg = numpy_to_imgmsg(rgb_np.astype(np.uint8), encoding="rgb8")
            rgb_msg.header.stamp = stamp
            rgb_msg.header.frame_id = frame_ids[i]
            self.rgb_pubs[i].publish(rgb_msg)

            # CameraInfo
            K = self.intrinsic_matrices[i]
            k_list = K.flatten().tolist()

            cam_info = CameraInfo()
            cam_info.header = rgb_msg.header
            cam_info.width = self.camera.cfg.width
            cam_info.height = self.camera.cfg.height
            cam_info.distortion_model = "plumb_bob"
            cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            cam_info.k = k_list
            cam_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            cam_info.p = [
                float(K[0, 0]), float(K[0, 1]), float(K[0, 2]), 0.0,
                float(K[1, 0]), float(K[1, 1]), float(K[1, 2]), 0.0,
                float(K[2, 0]), float(K[2, 1]), float(K[2, 2]), 0.0,
            ]

            self.rgb_info_pubs[i].publish(cam_info)

            # Depth
            if depth_tensors is not None:
                depth_np = depth_tensors[i].cpu().numpy().astype(np.float32)
                depth_msg = numpy_to_imgmsg(depth_np, encoding="32FC1")
                depth_msg.header.stamp = stamp
                depth_msg.header.frame_id = frame_ids[i]
                self.depth_pubs[i].publish(depth_msg)
                self.depth_info_pubs[i].publish(cam_info)

    def _compute_intrinsics(self):
        raw_intrinsics = self.camera.data.intrinsic_matrices.torch
        if hasattr(raw_intrinsics, "cpu"):
            self.intrinsic_matrices = raw_intrinsics.cpu().numpy()
        else:
            self.intrinsic_matrices = raw_intrinsics.numpy()

    def reset(self, env_ids: Sequence[int]):
        """
        env reset 時：
        1. 發 ORB reset 給 C++ node
        2. 清本地 pose/status cache
        3. 同步到 env.orb_slam_res
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if isinstance(env_ids, torch.Tensor):
            env_ids_cpu = env_ids.detach().cpu().numpy().tolist()
        else:
            env_ids_cpu = list(env_ids)

        # 先通知 ORB-SLAM3 reset
        # self.publish_orb_reset(env_ids_cpu)

        # 稍微 spin 一下，讓 reset message 有機會送出去
        for _ in range(3):
            rclpy.spin_once(self.node, timeout_sec=0.0)

        # 清本地 cache
        with self._orb_lock:
            for i in env_ids_cpu:
                self._orb_pose_xyyaw_cpu[i, :] = 0.0
                self._orb_status_cpu[i, 0] = 0.0

        self._sync_orb_results_to_env()
        return {}

    def close(self):
        self.node.destroy_node()