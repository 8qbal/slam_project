"""High-level reward. The paper gives none; this is a plain goal-reaching reward (same terms as the SB3 version in
``spot_vslam.legacy.high_level.paper_vec_env``). Distances use the ground-truth pose."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .commands import GoalCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["goal_progress", "goal_reached", "terminated_by"]


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
