"""Velocity command with turn-in-place envs."""

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass

__all__ = ["TurnInPlaceVelocityCommand", "TurnInPlaceVelocityCommandCfg"]


class TurnInPlaceVelocityCommand(UniformVelocityCommand):
    """``UniformVelocityCommand`` where a share of the resampled envs gets vx = vy = 0 and |wz| in
    ``turn_wz_range``. With vx from U(0, 1) and the standing envs at wz = 0, turning in place is otherwise rarely
    commanded, and every low level turned in place at under half the command (docs/low_level_reward.md)."""

    cfg: "TurnInPlaceVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        turn = ids[torch.rand(len(ids), device=self.device) < self.cfg.rel_turn_in_place_envs]
        if len(turn) == 0:
            return
        lo, hi = self.cfg.turn_wz_range
        wz = torch.empty(len(turn), device=self.device).uniform_(lo, hi)
        sign = torch.where(torch.rand(len(turn), device=self.device) < 0.5, -1.0, 1.0)
        self.vel_command_b[turn, 0] = 0.0
        self.vel_command_b[turn, 1] = 0.0
        self.vel_command_b[turn, 2] = sign * wz
        self.is_standing_env[turn] = False


@configclass
class TurnInPlaceVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = TurnInPlaceVelocityCommand
    rel_turn_in_place_envs: float = 0.2
    """Share of resampled envs that turn in place."""
    turn_wz_range: tuple[float, float] = (0.1, 0.4)
    """|wz| of those envs (rad/s)."""
