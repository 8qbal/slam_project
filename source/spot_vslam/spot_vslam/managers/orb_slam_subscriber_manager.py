# # import rclpy
# # from rclpy.node import Node
# # from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# # from std_msgs.msg import String
# # from geometry_msgs.msg import PoseStamped  # 假設 ORB-SLAM3 發布 PoseStamped；若不同，調整
# # from sensor_msgs.msg import PointCloud2
# # import torch
# # from dataclasses import dataclass, field
# # from typing import Sequence, List
# # import functools

# # from isaaclab.envs import ManagerBasedRLEnv

# # @dataclass
# # class OrbSlamSubscriberCfg:
# #     """Config for the ORB-SLAM3 subscriber manager (新增 pose_topic_suffix 以修正錯誤)."""
# #     topic_prefix_template: str = "/spot{i}/orbslam" 
# #     state_topic_suffix: str = "/tracking_state"
# #     local_pc_topic_suffix: str = "/local_pc"

# # class OrbSlamSubscriberManager:
# #     """ 
# #     為每個環境 (Spot) 訂閱對應的 ORB-SLAM3 輸出 (State, LocalPC, Pose)。
# #     (新增 Pose 訂閱以支援測試；但測試腳本使用 Isaac Lab ground truth 控制，只監控 SLAM pose)
# #     """
# #     def __init__(self, cfg: OrbSlamSubscriberCfg, env: ManagerBasedRLEnv):
# #         self.cfg = cfg
# #         self.env = env
# #         self.num_envs = self.env.num_envs
# #         self.device = self.env.device
# #         try:
# #             self.node = self.env.ros2_manager.node
# #         except AttributeError:
# #             if not rclpy.ok(): rclpy.init()
# #             self.node = rclpy.create_node("isaaclab_slam_subscriber_node")

# #         # --- Buffers ---
# #         self.state_buffer = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
# #         self.local_pc_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

# #         # --- Subscribers ---
# #         self.state_subs: List[rclpy.subscription.Subscription] = []
# #         self.local_pc_subs: List[rclpy.subscription.Subscription] = []

# #         qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

# #         self.node.get_logger().info(f"為 {self.num_envs} 個環境建立 ORB-SLAM3 訂閱者 (Pose, State, LocalPC)...")
# #         for i in range(self.num_envs):
# #             topic_prefix = self.cfg.topic_prefix_template.format(i=i)
# #             state_topic = topic_prefix + self.cfg.state_topic_suffix
# #             local_pc_topic = topic_prefix + self.cfg.local_pc_topic_suffix
            
# #             state_callback_partial = functools.partial(self._state_callback, env_id=i)
# #             local_pc_callback_partial = functools.partial(self._local_pc_callback, env_id=i)

# #             state_sub = self.node.create_subscription(String, state_topic, state_callback_partial, qos_profile)
# #             local_pc_sub = self.node.create_subscription(PointCloud2, local_pc_topic, local_pc_callback_partial, qos_profile)
            
# #             self.state_subs.append(state_sub)
# #             self.local_pc_subs.append(local_pc_sub)
# #             self.node.get_logger().debug(f"  Env {i}: Subscribed to {state_topic}, {local_pc_topic}")


# #     def _state_callback(self, msg: String, env_id: int):
# #         state_val = 0
# #         if msg.data == "TRACKING": state_val = 1
# #         elif msg.data == "LOST": state_val = 2
# #         self.state_buffer[env_id] = state_val

# #     def _local_pc_callback(self, msg: PointCloud2, env_id: int):
# #         num_points = 0
# #         if msg.point_step > 0:
# #             num_points = len(msg.data) // msg.point_step
# #         self.local_pc_count[env_id] = num_points
        
# #     def update(self, dt: float):
# #         pass

# #     def reset(self, env_ids: Sequence[int]):
# #         self.state_buffer[env_ids] = 0
# #         self.local_pc_count[env_ids] = 0
# #         return {}

# #     def close(self):
# #         self.node.get_logger().info("Shutting down ORB-SLAM Subscriber...")
# #         for sub in self.state_subs: self.node.destroy_subscription(sub)
# #         for sub in self.local_pc_subs: self.node.destroy_subscription(sub)
# #         self.state_subs.clear()
# #         self.local_pc_subs.clear()

# # /manager/orb_slam_subscriber_manager.py
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# from std_msgs.msg import String
# from std_msgs.msg import Int32  # 確保匯入 Int32
# from geometry_msgs.msg import PoseStamped 
# from sensor_msgs.msg import PointCloud2
# import torch
# from dataclasses import dataclass, field
# from typing import Sequence, List
# import functools

# from isaaclab.envs import ManagerBasedRLEnv

# @dataclass
# class OrbSlamSubscriberCfg:
#     """Config for the ORB-SLAM3 subscriber manager (包含 Pose, State, LocalPC)."""
#     topic_prefix_template: str = "/spot{i}/orbslam" 
    
#     pose_topic_suffix: str = "/robot_pose" 
#     state_topic_suffix: str = "/tracking_state"
#     local_pc_topic_suffix: str = "/local_pc"

# class OrbSlamSubscriberManager:
#     """ 
#     為每個環境 (Spot) 訂閱對應的 ORB-SLAM3 輸出 (Pose, State, LocalPC)。
#     """
#     def __init__(self, cfg: OrbSlamSubscriberCfg, env: ManagerBasedRLEnv):
#         self.cfg = cfg
#         self.env = env
#         self.num_envs = self.env.num_envs
#         self.device = self.env.device
#         try:
#             self.node = self.env.ros2_manager.node
#         except AttributeError:
#             if not rclpy.ok(): rclpy.init()
#             self.node = rclpy.create_node("isaaclab_slam_subscriber_node")

#         # --- Buffers ---
#         # Pose Buffer: [x, y, z, qx, qy, qz, qw]
#         self.pose_buffer = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        
#         # State Buffer: 儲存狀態整數 (0=INIT, 1=TRACKING, 2=LOST)
#         self.state_buffer = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        
#         self.local_pc_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

#         # --- Subscribers ---
#         self.pose_subs: List[rclpy.subscription.Subscription] = []
#         self.state_subs: List[rclpy.subscription.Subscription] = []
#         self.local_pc_subs: List[rclpy.subscription.Subscription] = []

#         qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

#         self.node.get_logger().info(f"為 {self.num_envs} 個環境建立 ORB-SLAM3 訂閱者 (Pose, State, LocalPC)...")
#         for i in range(self.num_envs):
#             topic_prefix = self.cfg.topic_prefix_template.format(i=i)
            
#             # 1. 定義 Topic 名稱
#             pose_topic = topic_prefix + self.cfg.pose_topic_suffix
#             state_topic = topic_prefix + self.cfg.state_topic_suffix
#             local_pc_topic = topic_prefix + self.cfg.local_pc_topic_suffix
            
#             # 2. 定義 Callbacks
#             pose_callback_partial = functools.partial(self._pose_callback, env_id=i)
#             state_callback_partial = functools.partial(self._state_callback, env_id=i)
#             local_pc_callback_partial = functools.partial(self._local_pc_callback, env_id=i)

#             # 3. 建立 Subscribers
#             # 注意：state_topic 現在使用 Int32 型別
#             pose_sub = self.node.create_subscription(PoseStamped, pose_topic, pose_callback_partial, qos_profile)
#             state_sub = self.node.create_subscription(Int32, state_topic, state_callback_partial, qos_profile)
#             local_pc_sub = self.node.create_subscription(PointCloud2, local_pc_topic, local_pc_callback_partial, qos_profile)
            
#             # 4. 儲存
#             self.pose_subs.append(pose_sub)
#             self.state_subs.append(state_sub)
#             self.local_pc_subs.append(local_pc_sub)
            
#             self.node.get_logger().debug(f"  Env {i}: Subscribed to {pose_topic}, {state_topic}, {local_pc_topic}")

#     def _pose_callback(self, msg: PoseStamped, env_id: int):
#         """處理 Pose 訊息並寫入 buffer"""
#         p = msg.pose.position
#         q = msg.pose.orientation
#         # 寫入 [x, y, z, qx, qy, qz, qw]
#         pose_tensor = torch.tensor([p.x, p.y, p.z, q.x, q.y, q.z, q.w], device=self.device, dtype=torch.float32)
#         self.pose_buffer[env_id] = pose_tensor 

#     # --- ▼▼▼ 修改後的 State Callback ▼▼▼ ---
#     def _state_callback(self, msg: Int32, env_id: int):
#         """
#         處理 Tracking State (Int32)
#         C++ 端已定義映射:
#           0 -> INIT (SYSTEM_NOT_READY, NO_IMAGES_YET, NOT_INITIALIZED)
#           1 -> TRACKING (OK)
#           2 -> LOST (LOST)
#         """
#         # 直接儲存整數值，無需字串比對
#         self.state_buffer[env_id] = msg.data
#     # --- ▲▲▲ 修改結束 ▲▲▲ ---

#     def _local_pc_callback(self, msg: PointCloud2, env_id: int):
#         num_points = 0
#         if msg.point_step > 0:
#             num_points = len(msg.data) // msg.point_step
#         self.local_pc_count[env_id] = num_points
        
#     def update(self, dt: float):
#         # 這裡可以加入一些邏輯，例如檢查 Pose 是否太久沒更新 (Watchdog)
#         pass

#     def reset(self, env_ids: Sequence[int]):
#         self.pose_buffer[env_ids] = 0.0 
#         self.state_buffer[env_ids] = 0 # 重置為 INIT
#         self.local_pc_count[env_ids] = 0
#         return {}

#     def close(self):
#         self.node.get_logger().info("Shutting down ORB-SLAM Subscriber...")
        
#         for sub in self.pose_subs: self.node.destroy_subscription(sub)
#         self.pose_subs.clear()
        
#         for sub in self.state_subs: self.node.destroy_subscription(sub)
#         self.state_subs.clear()
        
#         for sub in self.local_pc_subs: self.node.destroy_subscription(sub)
#         self.local_pc_subs.clear()
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import PointCloud2
import torch
import numpy as np  # [新增] 用於 CPU 快取
from dataclasses import dataclass
from typing import Sequence, List
import functools

from isaaclab.envs import ManagerBasedRLEnv

@dataclass
class OrbSlamSubscriberCfg:
    """Config for the ORB-SLAM3 subscriber manager."""
    # 這裡的格式要跟您 ROS2 節點發布的名稱一致
    # 假設 topic 為: /spot0/orbslam/robot_pose
    topic_prefix_template: str = "/spot{i}/orbslam" 
    
    pose_topic_suffix: str = "/robot_pose" 
    state_topic_suffix: str = "/tracking_state"
    local_pc_topic_suffix: str = "/local_pc"

class OrbSlamSubscriberManager:
    """ 
    為每個環境 (Spot) 訂閱對應的 ORB-SLAM3 輸出。
    使用 CPU Buffer 接收 ROS 訊息，並在 update() 時批量同步到 GPU。
    """
    def __init__(self, cfg: OrbSlamSubscriberCfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self.env = env
        self.num_envs = self.env.num_envs
        self.device = self.env.device
        
        # 嘗試共用 Ros2Manager 的 Node，避免多重節點
        try:
            self.node = self.env.ros2_manager.node
        except AttributeError:
            if not rclpy.ok(): rclpy.init()
            self.node = rclpy.create_node("isaaclab_slam_subscriber_node")

        # --- GPU Buffers (給 RL 使用) ---
        # [x, y, z, qx, qy, qz, qw]
        self.pose_buffer = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        self.state_buffer = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        self.local_pc_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

        # --- [新增] CPU Cache (給 ROS Callback 使用) ---
        # 避免 Callback 直接操作 GPU，提升效能
        self._pose_cache = np.zeros((self.num_envs, 7), dtype=np.float32)
        # 初始化 Quaternion w=1.0 (避免零旋轉)
        self._pose_cache[:, 6] = 1.0 
        self._state_cache = np.zeros((self.num_envs,), dtype=np.int32)
        self._pc_count_cache = np.zeros((self.num_envs,), dtype=np.int32)

        # --- Subscribers ---
        self.subs: List[rclpy.subscription.Subscription] = []
        
        # 使用 Best Effort (即時性優先，掉了就算了)
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.node.get_logger().info(f"建立 ORB-SLAM3 訂閱者 (Env count: {self.num_envs})...")
        
        for i in range(self.num_envs):
            # 格式化 Topic 名稱，例如 /spot0/orbslam
            topic_prefix = self.cfg.topic_prefix_template.format(i=i)
            
            pose_topic = topic_prefix + self.cfg.pose_topic_suffix
            state_topic = topic_prefix + self.cfg.state_topic_suffix
            local_pc_topic = topic_prefix + self.cfg.local_pc_topic_suffix
            
            # 使用 functools 綁定 env_id
            self.subs.append(self.node.create_subscription(
                PoseStamped, pose_topic, 
                functools.partial(self._pose_callback, env_id=i), qos_profile))
            
            self.subs.append(self.node.create_subscription(
                Int32, state_topic, 
                functools.partial(self._state_callback, env_id=i), qos_profile))
            
            self.subs.append(self.node.create_subscription(
                PointCloud2, local_pc_topic, 
                functools.partial(self._local_pc_callback, env_id=i), qos_profile))

    def _pose_callback(self, msg: PoseStamped, env_id: int):
        """Pose Callback (運行於 CPU)"""
        p = msg.pose.position
        q = msg.pose.orientation
        # 寫入 NumPy Array (極快)
        self._pose_cache[env_id, 0] = p.x
        self._pose_cache[env_id, 1] = p.y
        self._pose_cache[env_id, 2] = p.z
        self._pose_cache[env_id, 3] = q.x
        self._pose_cache[env_id, 4] = q.y
        self._pose_cache[env_id, 5] = q.z
        self._pose_cache[env_id, 6] = q.w

    def _state_callback(self, msg: Int32, env_id: int):
        """State Callback (運行於 CPU)"""
        self._state_cache[env_id] = int(msg.data)

    def _local_pc_callback(self, msg: PointCloud2, env_id: int):
        """PointCloud Callback (運行於 CPU)"""
        num_points = 0
        if msg.point_step > 0:
            num_points = len(msg.data) // msg.point_step
        self._pc_count_cache[env_id] = num_points
        
    def update(self, dt: float):
        """
        在 Simulation Step 時被呼叫。
        負責將 CPU Cache 的資料搬運到 GPU Tensor。
        """
        # [優化] 批量搬運，減少 CPU-GPU 溝通次數
        self.pose_buffer[:] = torch.from_numpy(self._pose_cache).to(self.device)
        self.state_buffer[:] = torch.from_numpy(self._state_cache).to(self.device)
        self.local_pc_count[:] = torch.from_numpy(self._pc_count_cache).to(self.device)

    def reset(self, env_ids: Sequence[int]):
        """重置環境時清除 Buffer"""
        # 清除 GPU Buffer
        self.pose_buffer[env_ids] = 0.0
        self.pose_buffer[env_ids, 6] = 1.0 # Quaternion w=1
        self.state_buffer[env_ids] = 0
        self.local_pc_count[env_ids] = 0
        
        # 同步清除 CPU Cache (重要！否則下一幀 update 會把舊資料寫回去)
        if isinstance(env_ids, torch.Tensor):
            ids_np = env_ids.cpu().numpy()
        else:
            ids_np = np.array(env_ids) # 轉成 numpy index
        self._pose_cache[ids_np] = 0.0
        self._pose_cache[ids_np, 6] = 1.0
        self._state_cache[ids_np] = 0
        self._pc_count_cache[ids_np] = 0
        
        return {}

    def close(self):
        self.node.get_logger().info("Shutting down ORB-SLAM Subscriber...")
        for sub in self.subs:
            self.node.destroy_subscription(sub)
        self.subs.clear()