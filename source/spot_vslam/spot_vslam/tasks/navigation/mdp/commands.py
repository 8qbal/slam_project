"""Goal command: a paper map's fixed goal, expressed in the frame of the robot's pose at the episode start.

With ``cfg.waypoints`` the policy is steered to the waypoints one after the other (the goal it sees is the next
waypoint) and only the last one, ``cfg.goal``, ends the episode. The policy has no map, so in a maze this does the
planning for it. The switch uses the ground-truth pose (a real planner would use the SLAM pose).

ORB-SLAM3 (``orbslam3_rgbd_node``) gives poses relative to the first tracked camera pose, i.e. the start pose. The
goal and the ground-truth pose are put in the same start frame, so GT-pose and SLAM-pose observations match.
"""

from __future__ import annotations

import math
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
        """The goal the policy sees: the next waypoint, or the final goal."""
        self.final_goal_w = self.goal_w.clone()
        self.targets_w = None
        if cfg.waypoints:  # (N, K + 1, 2): waypoints, then the final goal
            way = torch.tensor(cfg.waypoints, device=self.device, dtype=torch.float32)
            self.targets_w = torch.cat([way, torch.tensor(cfg.goal, device=self.device)[None]]).repeat(n, 1, 1)
            self.targets_w += env.scene.env_origins[:, None, :2]
            self.final_goal_w = self.targets_w[:, -1].clone()
            self.goal_w = self.targets_w[:, 0].clone()
        self.target_idx = torch.zeros(n, dtype=torch.long, device=self.device)
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
        """Current ground-truth distance to the (final) goal."""
        return torch.linalg.norm(self.final_goal_w - self.gt_pose_w()[0], dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # called on reset, after the reset events placed the robot: its pose now is the start pose
        xy_w, yaw_w = self.gt_pose_w()
        if self.targets_w is not None:
            self.target_idx[env_ids] = 0
            self.goal_w[env_ids] = self.targets_w[env_ids, 0]
        self.start_xy[env_ids] = xy_w[env_ids]
        self.start_yaw[env_ids] = yaw_w[env_ids]
        self.goal_s[env_ids] = self.world_to_start(self.goal_w)[env_ids]
        self.prev_distance[env_ids] = self.distance()[env_ids]

    def _update_command(self):
        if self.targets_w is None:
            return
        # next waypoint once the robot is within waypoint_radius of the current one (the last one is the goal)
        near = torch.linalg.norm(self.goal_w - self.gt_pose_w()[0], dim=-1) < self.cfg.waypoint_radius
        advance = near & (self.target_idx < self.targets_w.shape[1] - 1)
        ids = advance.nonzero().squeeze(-1)
        if len(ids):
            self.target_idx[ids] += 1
            self.goal_w[ids] = self.targets_w[ids, self.target_idx[ids]]
            self.goal_s[ids] = self.world_to_start(self.goal_w)[ids]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # A viewport overlay (debug draw), not USD marker prims: the cameras (the policy's depth, SLAM) must not see it
        if debug_vis and not hasattr(self, "_draw"):
            from isaaclab.sim.utils import enable_extension

            enable_extension("isaacsim.util.debug_draw")
            from isaacsim.util.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
        self._debug_vis = debug_vis
        if not debug_vis and hasattr(self, "_draw"):
            self._draw.clear_lines()
            self._draw.clear_points()

    def _debug_vis_callback(self, event):
        """Green circle (goal radius) and a tall pole at the goal, yellow dots on the waypoints."""
        green, yellow = (0.1, 0.9, 0.1, 1.0), (1.0, 0.8, 0.0, 1.0)
        angles = torch.linspace(0.0, 2.0 * math.pi, 33)
        ring = torch.stack([angles.cos(), angles.sin()], dim=-1) * self.cfg.goal_radius
        starts, ends, colors, sizes = [], [], [], []
        for g in self.final_goal_w.cpu():
            pts = [(float(g[0] + r[0]), float(g[1] + r[1]), 0.05) for r in ring]
            starts += pts[:-1] + [(float(g[0]), float(g[1]), 0.0)]
            ends += pts[1:] + [(float(g[0]), float(g[1]), 4.0)]
        colors = [green] * len(starts)
        sizes = [4.0] * (len(starts) - 1) + [8.0]
        self._draw.clear_lines()
        self._draw.draw_lines(starts, ends, colors, sizes)
        self._draw.clear_points()
        if self.targets_w is not None:
            way = self.targets_w[:, :-1].reshape(-1, 2).cpu()
            self._draw.draw_points(
                [(float(w[0]), float(w[1]), 0.3) for w in way], [yellow] * len(way), [16.0] * len(way)
            )

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
    waypoints: tuple[tuple[float, float], ...] = ()
    """Waypoints before the goal (map coordinates), visited in turn. Empty: straight to the goal."""
    waypoint_radius: float = 1.0
    """A waypoint counts as reached within this distance (m)."""
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
