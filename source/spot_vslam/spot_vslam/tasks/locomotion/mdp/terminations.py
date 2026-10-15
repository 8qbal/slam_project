"""Termination terms of the low-level policy."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

__all__ = ["sim_blowup"]


def sim_blowup(
    env: ManagerBasedRLEnv,
    max_lin_vel: float = 10.0,
    max_ang_vel: float = 50.0,
    max_joint_vel: float = 100.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Envs whose physics blew up: a non-finite or absurd body / joint velocity (Spot walks at < 2 m/s).

    One such env gave rewards of -1e10 and then NaN weights in a WallsSoftSym run (ep ~8750); ending it resets it.
    """
    data = env.scene[asset_cfg.name].data
    lin = data.root_lin_vel_w.torch
    ang = data.root_ang_vel_w.torch
    qd = data.joint_vel.torch
    bad = ~(torch.isfinite(lin).all(1) & torch.isfinite(ang).all(1) & torch.isfinite(qd).all(1))
    bad |= torch.linalg.norm(lin, dim=1) > max_lin_vel
    bad |= torch.linalg.norm(ang, dim=1) > max_ang_vel
    bad |= qd.abs().amax(dim=1) > max_joint_vel
    return bad
