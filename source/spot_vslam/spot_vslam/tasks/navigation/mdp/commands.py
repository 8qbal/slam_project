"""Goal command: a paper map's fixed goal, expressed in the frame of the robot's pose at the episode start.

ORB-SLAM3 (``orbslam3_rgbd_node``) gives poses relative to the first tracked camera pose, i.e. the start pose. The
goal and the ground-truth pose are put in the same start frame, so GT-pose and SLAM-pose observations match.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import wrap_to_pi

from spot_vslam.utils import yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GoalCommand(CommandTerm):
    """``command`` is the goal (x, y) in the start frame. Never resampled within an episode."""

    cfg: GoalCommandCfg

    def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        n = self.num_envs
        # every env has its own copy of the map at its env origin
        self.goal_w = torch.tensor(cfg.goal, device=self.device).repeat(n, 1) + env.scene.env_origins[:, :2]
        self.start_xy = torch.zeros(n, 2, device=self.device)
        self.start_yaw = torch.zeros(n, device=self.device)
        self.goal_s = torch.zeros(n, 2, device=self.device)
        self.prev_distance = torch.zeros(n, device=self.device)
        """Ground-truth distance to the goal at the end of the previous step (for the progress reward)."""
        self.metrics["distance"] = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.goal_s

    def gt_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.robot.data.root_pos_w.torch[:, :2], yaw_from_quat_xyzw(self.robot.data.root_quat_w.torch)

    def gt_pose_start_frame(self) -> torch.Tensor:
        """Ground-truth (x, y, yaw) in the start frame, shape (N, 3)."""
        xy_w, yaw_w = self.gt_pose_w()
        return torch.cat([self.world_to_start(xy_w), wrap_to_pi(yaw_w - self.start_yaw).unsqueeze(-1)], dim=-1)

    def world_to_start(self, xy_w: torch.Tensor) -> torch.Tensor:
        d = xy_w - self.start_xy
        c, s = torch.cos(self.start_yaw), torch.sin(self.start_yaw)
        return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=-1)

    def distance(self) -> torch.Tensor:
        """Current ground-truth distance to the goal."""
        return torch.linalg.norm(self.goal_w - self.gt_pose_w()[0], dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # called on reset, after the reset events placed the robot: its pose now is the start pose
        xy_w, yaw_w = self.gt_pose_w()
        self.start_xy[env_ids] = xy_w[env_ids]
        self.start_yaw[env_ids] = yaw_w[env_ids]
        self.goal_s[env_ids] = self.world_to_start(self.goal_w)[env_ids]
        self.prev_distance[env_ids] = self.distance()[env_ids]

    def _update_command(self):
        pass

    def _update_metrics(self):
        # runs after the rewards of the step, so the progress reward sees the previous step's distance
        self.prev_distance[:] = self.distance()
        self.metrics["distance"][:] = self.prev_distance


@configclass
class GoalCommandCfg(CommandTermCfg):
    class_type: type = GoalCommand

    asset_name: str = "robot"
    goal: tuple[float, float] = MISSING
    """Goal in map coordinates (relative to the env origin)."""
    goal_radius: float = MISSING
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
