"""TSDF dense mapping (Open3D) from the RGB-D camera. Builds a map only; nothing from it feeds the policies."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class DenseMapManagerCfg:
    camera_name: str = "tilted_camera"

    voxel_length: float = 0.02
    sdf_trunc: float = 0.05
    depth_scale: float = 1.0
    depth_max: float = 5.0

    export_dir: str = "./output/dense_map"
    export_every_n_frames: int = 50
    use_color: bool = True

    # intrinsics of tilted_camera (320x240)
    width: int = 320
    height: int = 240
    fx: float = 145.4545
    fy: float = 145.4545
    cx: float = 160.0
    cy: float = 120.0

    # camera pose in the robot body frame (CameraCfg.OffsetCfg)
    cam_offset_pos: tuple[float, float, float] = (0.4, 0.0, 0.0)
    cam_offset_quat_xyzw: tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5)

    # camera pose source
    pose_source: str = "gt"  # "gt" or "orb"
    debug_pose: bool = True

    # live view
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
        self.intrinsic: o3d.camera.PinholeCameraIntrinsic | None = None

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
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _get_gt_camera_pose_world(self) -> np.ndarray:
        """Camera pose from the ground-truth robot pose and the camera mount."""
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

    def _get_orb_camera_pose_world(self) -> np.ndarray | None:
        """Camera pose from the ORB-SLAM3 pose only (no ground truth); None while tracking is lost.

        The planar SLAM pose is mapped to the world axes (x/y swapped and mirrored, yaw negated), and the height is
        fixed at Spot's standing height (0.6 m) because the SLAM pose has no z.
        """
        env = self.env_unwrapped
        if not hasattr(env, "orb_slam_res"):
            return None

        res = env.orb_slam_res
        pose_xyyaw = res.get("pose_xyyaw", None)
        status = res.get("status", None)

        if pose_xyyaw is None:
            return None
        if status is not None and float(status[0].detach().cpu().numpy()[0]) < 0.5:
            return None

        raw_x, raw_y, raw_yaw = pose_xyyaw[0].detach().cpu().numpy()

        x_align = -raw_y
        y_align = -raw_x
        yaw_align = -raw_yaw

        z_height = 0.6

        c, s = np.cos(yaw_align), np.sin(yaw_align)
        T_wr = np.eye(4, dtype=np.float64)
        T_wr[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T_wr[:3, 3] = [x_align, y_align, z_height]

        cam_offset_pos = np.array(self.cfg.cam_offset_pos, dtype=np.float64)
        cam_offset_quat_xyzw = np.array(self.cfg.cam_offset_quat_xyzw, dtype=np.float64)
        R_rc = self._quat_xyzw_to_rot(cam_offset_quat_xyzw)

        T_rc = np.eye(4, dtype=np.float64)
        T_rc[:3, :3] = R_rc
        T_rc[:3, 3] = cam_offset_pos

        return T_wr @ T_rc

    def _get_camera_pose_world(self) -> np.ndarray:
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
        # export whatever was integrated since the last periodic export (short runs never reach export_every_n_frames)
        if self.frame_count > 0 and self.frame_count % self.cfg.export_every_n_frames != 0:
            self.export()
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None
