"""In-process cuVSLAM (PyCuVSLAM) on the SLAM cameras, one tracker per env (docs/cuvslam.md).

Stereo (left ``tilted_camera`` + ``right_camera``, cuVSLAM's main mode) or RGB-D (``tilted_camera`` alone, the ORB-SLAM3
input). No ROS 2: images go from the camera tensors straight to ``Tracker.track()``. Results are written to
``env.slam_res`` with the keys of ``env.orb_slam_res`` plus quality signals, so ``robot_pose(pose_source="slam")`` can
read either SLAM system.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    # imported where used, so configs that import CuVslamManagerCfg don't need PyCuVSLAM (it is only in the uv venv)
    import cuvslam

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera


@dataclass
class CuVslamManagerCfg:
    """Config for :class:`CuVslamManager`."""

    camera_name: str = "tilted_camera"
    """Scene name of the left / RGB-D camera (``rgb``, plus ``distance_to_image_plane`` in RGB-D mode)."""

    right_camera_name: str | None = None
    """Scene name of the right camera of a rectified stereo pair: set = stereo, None = RGB-D. RGB-D odometry jumped
    metres on single frames in the warehouse, stereo is cuVSLAM's main mode and the ZED's (docs/cuvslam.md)."""

    loop_closure: bool = True
    """Run cuVSLAM's SLAM (loop closure) on top of odometry, like ORB-SLAM3. False = odometry only."""

    max_depth_m: float = 10.0
    """RGB-D only: depth beyond this (and NaN / inf) is passed as 0 = no depth. Matches the camera clipping range."""


class CuVslamManager:
    """One cuVSLAM tracker per env, fed with every new frame of the SLAM camera.

    ``env.slam_res`` after each update (all tensors on the env device):

    * ``pose_xyyaw`` (N, 3): robot body x, y, yaw in its start frame (the body at the last reset): the loop-closure
      pose when SLAM gives one, else the odometry pose
    * ``odom_xyyaw`` (N, 3): the odometry pose alone (no loop closure)
    * ``slam_xyyaw`` (N, 3): the loop-closure pose alone (NaN until SLAM gives one, or with ``loop_closure=False``)
    * ``status`` (N, 1): 1 = the tracker returned a pose for the last frame, 0 = lost / no frame yet
    * ``num_features`` (N, 1): tracked 2D features in the last frame
    * ``track_time_ms`` (N, 1): wall time of the last ``track()`` call
    * ``stamp_s`` (N,): sensor time of the image the pose belongs to (seconds since the env's last reset)
    """

    DEPTH_SCALE = 1000.0
    """Depth is passed as uint16 millimetres (the binding only takes uint16 depth)."""

    def __init__(self, cfg: CuVslamManagerCfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.camera: Camera = env.scene.sensors[cfg.camera_name]
        self.stereo = cfg.right_camera_name is not None
        self.right_camera: Camera | None = env.scene.sensors[cfg.right_camera_name] if self.stereo else None
        if self.stereo and self.right_camera.cfg.update_period != self.camera.cfg.update_period:
            raise ValueError(
                f"stereo cameras need the same update period, got {self.camera.cfg.update_period} (left) and"
                f" {self.right_camera.cfg.update_period} (right)"
            )
        self.rig = self._make_rig()
        self.trackers: list[cuvslam.Tracker | None] = [None] * self.num_envs

        self.pose_xyyaw = torch.zeros(self.num_envs, 3, device=self.device)
        self.odom_xyyaw = torch.zeros(self.num_envs, 3, device=self.device)
        self.slam_xyyaw = torch.full((self.num_envs, 3), float("nan"), device=self.device)
        self.status = torch.zeros(self.num_envs, 1, device=self.device)
        self.num_features = torch.zeros(self.num_envs, 1, device=self.device)
        self.track_time_ms = torch.zeros(self.num_envs, 1, device=self.device)
        self.stamp_s = torch.full((self.num_envs,), -1.0, device=self.device)
        self._publish()

    # -- setup

    def _make_rig(self) -> cuvslam.Rig:
        """Pinhole camera(s) from the camera cfg; rig frame = robot body, so the pose is the body pose."""
        cameras = [self._make_camera(self.camera)]
        if self.stereo:
            cameras.append(self._make_camera(self.right_camera))
        import cuvslam

        return cuvslam.Rig(cameras=cameras)

    @staticmethod
    def _make_camera(sensor: Camera) -> cuvslam.Camera:
        import cuvslam

        cfg = sensor.cfg
        spawn = cfg.spawn
        fx = fy = spawn.focal_length / spawn.horizontal_aperture * cfg.width  # square pixels
        offset = cfg.offset
        if offset.convention != "ros":
            raise ValueError(
                f"cuVSLAM needs the camera offset in the 'ros' (optical) convention, got {offset.convention}"
            )
        return cuvslam.Camera(
            size=[cfg.width, cfg.height],
            principal=[cfg.width / 2.0, cfg.height / 2.0],
            focal=[fx, fy],
            # Isaac Lab 3.0 quaternions are xyzw, as cuVSLAM's
            rig_from_camera=cuvslam.Pose(rotation=list(offset.rot), translation=list(offset.pos)),
        )

    def _make_tracker(self) -> cuvslam.Tracker:
        import cuvslam

        if self.stereo:
            odom_cfg = cuvslam.Tracker.OdometryConfig(
                odometry_mode=cuvslam.Tracker.OdometryMode.Multicamera,
                rectified_stereo_camera=True,
                enable_observations_export=True,
                enable_landmarks_export=False,
            )
        else:
            odom_cfg = cuvslam.Tracker.OdometryConfig(
                odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
                rgbd_settings=cuvslam.Tracker.OdometryRGBDSettings(
                    depth_scale_factor=self.DEPTH_SCALE, depth_camera_id=0
                ),
                enable_observations_export=True,
                enable_landmarks_export=False,
            )
        slam_cfg = cuvslam.Tracker.SlamConfig() if self.cfg.loop_closure else None
        return cuvslam.Tracker(self.rig, odom_cfg, slam_cfg)

    # -- per step

    def update(self, dt: float):
        output = self.camera.data.output  # reading the data brings the sensor up to date
        stamps = wp.to_torch(self.camera._timestamp_last_update)
        new = torch.nonzero(stamps > self.stamp_s).flatten().tolist()
        if not new:
            return
        rgb = output["rgb"].torch[..., :3]
        if self.stereo:
            # same update period as the left camera, so both render the same sim instant
            right = self.right_camera.data.output["rgb"].torch[..., :3]
        else:
            depth = output["distance_to_image_plane"].torch.squeeze(-1)
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = torch.where(depth > self.cfg.max_depth_m, torch.zeros_like(depth), depth)
            depth_mm = (depth * self.DEPTH_SCALE).round().clamp(0, 65535).to(torch.int32).to(torch.uint16)

        for i in new:
            if self.trackers[i] is None:
                self.trackers[i] = self._make_tracker()
            stamp_ns = int(round(float(stamps[i]) * 1e9))
            t0 = time.perf_counter()
            if self.stereo:
                odom, slam_pose = self.trackers[i].track(stamp_ns, [rgb[i].contiguous(), right[i].contiguous()])
            else:
                odom, slam_pose = self.trackers[i].track(
                    stamp_ns, [rgb[i].contiguous()], depths=[depth_mm[i].contiguous()]
                )
            self.track_time_ms[i, 0] = (time.perf_counter() - t0) * 1e3
            self.stamp_s[i] = stamps[i]

            ok = odom.world_from_rig is not None
            self.status[i, 0] = 1.0 if ok else 0.0
            if ok:
                self.odom_xyyaw[i] = torch.tensor(self._xyyaw(odom.world_from_rig.pose), device=self.device)
            if slam_pose is not None:
                self.slam_xyyaw[i] = torch.tensor(self._xyyaw(slam_pose), device=self.device)
            pose = slam_pose if slam_pose is not None else (odom.world_from_rig.pose if ok else None)
            if pose is not None:
                self.pose_xyyaw[i] = torch.tensor(self._xyyaw(pose), device=self.device)
            self.num_features[i, 0] = len(self.trackers[i].get_last_observations(0)) if ok else 0
        self._publish()

    @staticmethod
    def _xyyaw(pose: cuvslam.Pose) -> tuple[float, float, float]:
        qx, qy, qz, qw = (float(v) for v in pose.rotation)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return float(pose.translation[0]), float(pose.translation[1]), yaw

    def _publish(self):
        self.env.slam_res = {
            "pose_xyyaw": self.pose_xyyaw,
            "odom_xyyaw": self.odom_xyyaw,
            "slam_xyyaw": self.slam_xyyaw,
            "status": self.status,
            "num_features": self.num_features,
            "track_time_ms": self.track_time_ms,
            "stamp_s": self.stamp_s,
        }

    # -- episode reset

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict:
        """Drop the trackers of these envs: the next frame starts a new map at the robot's reset pose."""
        if env_ids is None:
            env_ids = range(self.num_envs)
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        for i in env_ids:
            self.trackers[i] = None
            self.pose_xyyaw[i] = 0.0
            self.odom_xyyaw[i] = 0.0
            self.slam_xyyaw[i] = float("nan")
            self.status[i] = 0.0
            self.num_features[i] = 0.0
            # sensor time restarts at 0 on reset, so any new frame counts as new
            self.stamp_s[i] = -1.0
        self._publish()
        return {}

    def close(self):
        self.trackers = [None] * self.num_envs
