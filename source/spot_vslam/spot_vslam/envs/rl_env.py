"""Manager-based RL environment with ROS 2 / ORB-SLAM3 hooks.

This is a thin subclass of Isaac Lab's :class:`isaaclab.envs.ManagerBasedRLEnv`. It keeps the upstream
step/reset logic and only adds two optional managers, created from the env cfg:

* ``cfg.ros2`` -> :class:`Ros2Manager` (publishes camera images / TF to ROS 2)
* ``cfg.slam_subscriber`` -> :class:`OrbSlamSubscriberManager` (receives ORB-SLAM3 poses)

Both are updated every env step right after the command manager and before interval events and
observations, so observation terms can read the latest SLAM pose.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv as BaseManagerBasedRLEnv
from isaaclab.envs import ManagerBasedRLEnvCfg

from spot_vslam.managers.orb_slam_subscriber_manager import OrbSlamSubscriberManager
from spot_vslam.managers.ros2_manager import Ros2Manager


class ManagerBasedRLEnv(BaseManagerBasedRLEnv):
    """:class:`isaaclab.envs.ManagerBasedRLEnv` plus the ROS 2 and ORB-SLAM3 managers."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        # must exist before the base class calls load_managers()
        self.ros2_manager: Ros2Manager | None = None
        self.slam_subscriber_manager: OrbSlamSubscriberManager | None = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def load_managers(self):
        super().load_managers()

        if getattr(self.cfg, "ros2", None) is not None:
            self.ros2_manager = Ros2Manager(self.cfg.ros2, self)
            print("[INFO] Ros2 Manager: loaded.")
        else:
            print("[WARN] 'ros2' config is None or missing. Ros2Manager skipped.")

        if getattr(self.cfg, "slam_subscriber", None) is not None:
            self.slam_subscriber_manager = OrbSlamSubscriberManager(self.cfg.slam_subscriber, self)
            print("[INFO] SLAM Subscriber Manager: loaded.")
        else:
            print("[WARN] 'slam_subscriber' config is None or missing. Manager skipped.")

        # The base step() calls command_manager.compute() exactly once per env step, after resets and
        # before interval events / observations. Hook the ROS 2 / SLAM update there instead of forking step().
        compute_commands = self.command_manager.compute

        def compute_commands_and_update_slam(dt: float):
            compute_commands(dt=dt)
            self._update_slam_managers(dt)

        self.command_manager.compute = compute_commands_and_update_slam

    def _update_slam_managers(self, dt: float):
        if self.ros2_manager:
            self.ros2_manager.update(dt=dt)
        if self.slam_subscriber_manager:
            self.slam_subscriber_manager.update(dt=dt)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor):
        super()._reset_idx(env_ids)

        if self.ros2_manager:
            info = self.ros2_manager.reset(env_ids)
            if info:
                self.extras["log"].update(info)
        if self.slam_subscriber_manager:
            info = self.slam_subscriber_manager.reset(env_ids)
            if info:
                self.extras["log"].update(info)

    def close(self):
        if not self._is_closed:
            # close the subscriber first: it may share the ROS 2 node owned by Ros2Manager
            if self.slam_subscriber_manager:
                self.slam_subscriber_manager.close()
                self.slam_subscriber_manager = None
            if self.ros2_manager:
                self.ros2_manager.close()
                self.ros2_manager = None
        super().close()
