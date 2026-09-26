from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d


@dataclass
class DenseMapManagerCfg:
    camera_name: str = "tilted_camera"

    voxel_length: float = 0.02
    sdf_trunc: float = 0.05
    depth_scale: float = 1.0
    depth_max: float = 5.0

    export_dir: str = "./dense_map_output"
    export_every_n_frames: int = 50
    use_color: bool = True

    # 相機內參（對應 tilted_camera: 320x240）
    width: int = 320
    height: int = 240
    fx: float = 145.4545
    fy: float = 145.4545
    cx: float = 160.0
    cy: float = 120.0

    # camera 相對 robot body 的外參（對應 CameraCfg.OffsetCfg）
    cam_offset_pos: tuple[float, float, float] = (0.4, 0.0, 0.0)
    cam_offset_quat_xyzw: tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5)

    # pose 來源 (現在支援自由切換)
    pose_source: str = "gt"   # "gt" or "orb"
    debug_pose: bool = True

    # 即時顯示
    enable_live_vis: bool = True
    vis_update_every_n_frames: int = 10
    vis_as_mesh: bool = False
    window_name: str = "TSDF Dense Map"
    width_vis: int = 640
    height_vis: int = 480


class DenseMapManager:
    def __init__(self, cfg: DenseMapManagerCfg, env):
        self.cfg = cfg
        self.env = env
        self.env_unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

        os.makedirs(self.cfg.export_dir, exist_ok=True)

        color_type = (
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            if self.cfg.use_color
            else o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        )

        self.tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.cfg.voxel_length,
            sdf_trunc=self.cfg.sdf_trunc,
            color_type=color_type,
        )

        self.frame_count = 0
        self.intrinsic: Optional[o3d.camera.PinholeCameraIntrinsic] = None

        self.vis = None
        self.vis_geom = None
        self.vis_initialized = False

        if self.cfg.enable_live_vis:
            self._init_visualizer()

    def _init_visualizer(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=self.cfg.window_name,
            width=self.cfg.width_vis,
            height=self.cfg.height_vis,
            visible=True,
        )
        render_opt = self.vis.get_render_option()
        render_opt.point_size = 2.0
        render_opt.background_color = np.array([0.05, 0.05, 0.05])

    def _build_intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            self.cfg.width,
            self.cfg.height,
            self.cfg.fx,
            self.cfg.fy,
            self.cfg.cx,
            self.cfg.cy,
        )

    @staticmethod
    def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)


    def _get_gt_camera_pose_world(self) -> np.ndarray:
        """
        [GT 模式] 使用機器人完美的真實 3D 姿態與相機外參進行建圖
        """
        env = self.env_unwrapped
        robot_pos = env.scene["robot"].data.root_pos_w.torch[0].detach().cpu().numpy()
        robot_quat_xyzw = env.scene["robot"].data.root_quat_w.torch[0].detach().cpu().numpy()

        R_wr = self._quat_xyzw_to_rot(robot_quat_xyzw)
        T_wr = np.eye(4, dtype=np.float64)
        T_wr[:3, :3] = R_wr
        T_wr[:3, 3] = robot_pos

        cam_offset_pos = np.array(self.cfg.cam_offset_pos, dtype=np.float64)
        cam_offset_quat_xyzw = np.array(self.cfg.cam_offset_quat_xyzw, dtype=np.float64)
        R_rc = self._quat_xyzw_to_rot(cam_offset_quat_xyzw)

        T_rc = np.eye(4, dtype=np.float64)
        T_rc[:3, :3] = R_rc
        T_rc[:3, 3] = cam_offset_pos

        return T_wr @ T_rc

    def _get_orb_camera_pose_world(self) -> Optional[np.ndarray]:
        """
        [純 ORB 模式] 嚴格對照組：絕對不使用任何 GT 數據。
        套用座標鏡像校正、相機外參，但高度 (Z) 強制設為固定常數。
        """
        env = self.env_unwrapped
        if not hasattr(env, "orb_slam_res"):
            return None

        res = env.orb_slam_res
        pose_xyyaw = res.get("pose_xyyaw", None)
        status = res.get("status", None)

        if pose_xyyaw is None: return None
        if status is not None and float(status[0].detach().cpu().numpy()[0]) < 0.5:
            return None

        # 1. 取得 ORB 原始 2D 軌跡 (純粹的 SLAM 輸出)
        raw_x, raw_y, raw_yaw = pose_xyyaw[0].detach().cpu().numpy()

        # 2. 套用我們推導出的完美座標軸校正 (XY鏡像與Yaw反轉)
        x_align = -raw_y
        y_align = -raw_x
        yaw_align = -raw_yaw

        # 3. 【嚴格 ORB 模式】絕對不偷看 GT 的 Z 軸！
        # 由於 2D SLAM 沒有高度資訊，我們將 Z 軸強制固定為 Spot 大致的靜止站立高度 (約 0.6 公尺)。
        # 這樣可以保證建圖完全不依賴真實物理世界的起伏。
        z_height = 0.6

        # 4. 建立機器人本體的 3D 轉換矩陣
        c, s = np.cos(yaw_align), np.sin(yaw_align)
        T_wr = np.eye(4, dtype=np.float64)
        T_wr[:3, :3] = [
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ]
        T_wr[:3, 3] = [x_align, y_align, z_height]

        # 5. 加上相機的外參 (前置距離與朝下的角度)
        cam_offset_pos = np.array(self.cfg.cam_offset_pos, dtype=np.float64)
        cam_offset_quat_xyzw = np.array(self.cfg.cam_offset_quat_xyzw, dtype=np.float64)
        R_rc = self._quat_xyzw_to_rot(cam_offset_quat_xyzw)
        
        T_rc = np.eye(4, dtype=np.float64)
        T_rc[:3, :3] = R_rc
        T_rc[:3, 3] = cam_offset_pos

        # 6. 算出最終相機的 World Pose
        return T_wr @ T_rc

    def _get_camera_pose_world(self) -> np.ndarray:
        # 解除封印：現在支援在終端機指令自由切換 pose_source
        if self.cfg.pose_source == "gt":
            return self._get_gt_camera_pose_world()
        elif self.cfg.pose_source == "orb":
            return self._get_orb_camera_pose_world()
        else:
            raise ValueError(f"Unknown pose_source: {self.cfg.pose_source}")

    def _extract_rgbd(self):
        sensor = self.env_unwrapped.scene.sensors[self.cfg.camera_name]
        rgb = sensor.data.output["rgb"].torch[0].detach().cpu().numpy()
        depth = sensor.data.output["distance_to_image_plane"].torch[0].detach().cpu().numpy()

        if depth.ndim == 3:
            depth = depth[..., 0]

        if depth.shape == (self.cfg.width, self.cfg.height):
            depth = depth.T

        depth = np.nan_to_num(depth, nan=self.cfg.depth_max, posinf=self.cfg.depth_max, neginf=0.0)
        depth = np.clip(depth, 0.0, self.cfg.depth_max).astype(np.float32)

        rgb = np.asarray(rgb)
        if rgb.shape[:2] == (self.cfg.width, self.cfg.height):
            rgb = np.transpose(rgb, (1, 0, 2))

        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255.0).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        return rgb, depth

    def update(self):
        rgb, depth = self._extract_rgbd()

        if self.intrinsic is None:
            self.intrinsic = self._build_intrinsic()

        color_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_o3d,
            depth=depth_o3d,
            depth_scale=self.cfg.depth_scale,
            depth_trunc=self.cfg.depth_max,
            convert_rgb_to_intensity=False,
        )

        T_wc = self._get_camera_pose_world()
        if T_wc is None:
            return
        T_cw = np.linalg.inv(T_wc)

        self.tsdf.integrate(rgbd, self.intrinsic, T_cw)
        self.frame_count += 1

        if self.cfg.enable_live_vis and self.frame_count % self.cfg.vis_update_every_n_frames == 0:
            self.update_visualization()

        if self.frame_count % self.cfg.export_every_n_frames == 0:
            self.export()

    def update_visualization(self):
        if self.cfg.vis_as_mesh:
            geom = self.tsdf.extract_triangle_mesh()
            if len(geom.vertices) == 0:
                self._poll()
                return
            geom.compute_vertex_normals()
        else:
            geom = self.tsdf.extract_point_cloud()
            if len(geom.points) == 0:
                self._poll()
                return

        if not self.vis_initialized:
            self.vis_geom = geom
            self.vis.add_geometry(self.vis_geom)
            ctr = self.vis.get_view_control()
            ctr.set_zoom(0.6)
            self.vis_initialized = True
        else:
            self.vis.remove_geometry(self.vis_geom, reset_bounding_box=False)
            self.vis_geom = geom
            self.vis.add_geometry(self.vis_geom, reset_bounding_box=False)

        self._poll()

    def _poll(self):
        if self.vis is not None:
            self.vis.poll_events()
            self.vis.update_renderer()

    def export(self):
        mesh = self.tsdf.extract_triangle_mesh()
        if len(mesh.vertices) > 0:
            mesh.compute_vertex_normals()
            ply_path = os.path.join(self.cfg.export_dir, f"tsdf_mesh_{self.frame_count:06d}.ply")
            o3d.io.write_triangle_mesh(ply_path, mesh)

        pcd = self.tsdf.extract_point_cloud()
        if len(pcd.points) > 0:
            pcd_path = os.path.join(self.cfg.export_dir, f"tsdf_pcd_{self.frame_count:06d}.ply")
            o3d.io.write_point_cloud(pcd_path, pcd)

    def close(self):
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None