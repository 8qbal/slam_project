"""High-level reward. The paper gives none; this is a plain goal-reaching reward (same terms as the SB3 version in
``spot_vslam.legacy.high_level.paper_vec_env``). Distances use the ground-truth pose."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .commands import GoalCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["body_contact", "goal_progress", "goal_reached", "not_moving", "path_progress", "terminated_by"]


def goal_progress(env: ManagerBasedRLEnv, command_name: str = "goal") -> torch.Tensor:
    """Metres moved towards the goal during this step."""
    cmd: GoalCommand = env.command_manager.get_term(command_name)
    return cmd.prev_distance - cmd.distance()


def goal_reached(env: ManagerBasedRLEnv, command_name: str = "goal") -> torch.Tensor:
    """True when the robot is within the goal radius (also used as the success termination)."""
    cmd: GoalCommand = env.command_manager.get_term(command_name)
    return cmd.distance() < cmd.cfg.goal_radius


def terminated_by(env: ManagerBasedRLEnv, term_names: Sequence[str]) -> torch.Tensor:
    """True when any of the given termination terms fired this step (e.g. a fall, not a success or a timeout)."""
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in term_names:
        done |= env.termination_manager.get_term(name)
    return done


def path_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Metres by which the path distance to the goal beats its best value so far this episode (GUIDE's progress
    reward): backing out of a dead end counts, pressing against a wall near the goal does not. Needs
    ``reset_in_maze(path_field=True)``."""
    pf = env.path_field
    all_ids = torch.arange(env.num_envs, device=env.device)
    d = pf.lookup(all_ids, env.scene["robot"].data.root_pos_w.torch[:, :2])
    d = torch.where(torch.isinf(d), pf.best, d)  # blocked cells (next to a wall): no change
    gain = (pf.best - d).clamp(min=0.0)
    pf.best = torch.minimum(pf.best, d)
    return gain


def not_moving(
    env: ManagerBasedRLEnv, command_name: str = "goal", speed: float = 0.1, yaw_rate: float = 0.15
) -> torch.Tensor:
    """1 while the robot neither walks nor turns and is not at the goal (stalled against a wall)."""
    cmd: GoalCommand = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    still = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=-1) < speed
    still &= robot.data.root_ang_vel_b.torch[:, 2].abs() < yaw_rate
    return (still & (cmd.distance() > cmd.cfg.goal_radius)).float()


def body_contact(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    """1 while the body or a leg touches something (net contact force above ``threshold`` N)."""
    sensor = env.scene["contact_forces"]
    if not hasattr(env, "_contact_body_ids"):
        env._contact_body_ids, _ = sensor.find_bodies(["body", ".*leg"])
    hist = sensor.data.net_forces_w_history[:, :, env._contact_body_ids]
    return (hist.norm(dim=-1).amax(dim=(1, 2)) > threshold).float()
