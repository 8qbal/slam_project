"""Path distance to the goal on the random training tiles (mazes, warehouses), for the high-level progress reward.

The straight-line progress reward pays a robot for pressing against the wall that lies closest to the goal. GUIDE
(Wang et al. 2026) pays progress along the maze graph instead: only a new best (shortest) path distance is rewarded, so
backing out of a dead end counts as progress.

``maze_terrain`` and ``warehouse_terrain`` record the rectangles of every tile they build (``record_tile``).
Isaac Lab builds the tiles in the order of ``unravel_index(i, (rows, cols))``, so the last ``rows * cols`` records are
the tiles of the terrain importer, row-major. ``PathField`` rasterises them into occupancy grids (inflated by the robot
radius) and, at every reset, runs a wavefront from the goal over the grid of that env's tile: a distance field.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["PathField", "record_tile"]

TILE_RECTS: list[tuple[tuple[float, float], list[tuple[float, float, float, float]]]] = []
"""Per built tile: (size, obstacle rectangles (x0, y0, x1, y1) in the tile corner frame)."""


def record_tile(size: tuple[float, float], rects) -> None:
    TILE_RECTS.append(((float(size[0]), float(size[1])), [tuple(float(v) for v in r) for r in rects]))


class PathField:
    """Occupancy grids of all tiles and one distance field per env (``field``, metres, inf = blocked)."""

    def __init__(self, env: ManagerBasedEnv, rows: int, cols: int, robot_radius: float = 0.35, res: float = 0.25):
        tiles = TILE_RECTS[-rows * cols :]
        if len(tiles) != rows * cols:
            raise RuntimeError(f"PathField: {len(tiles)} recorded tiles, the terrain has {rows * cols}")
        dev = env.device
        sx, sy = tiles[0][0]
        self.res = res
        self.w, self.h = int(math.ceil(sx / res)), int(math.ceil(sy / res))
        xs, ys = (np.arange(self.w) + 0.5) * res, (np.arange(self.h) + 0.5) * res
        occ = np.zeros((rows * cols, self.h, self.w), dtype=bool)
        for k, (_, rects) in enumerate(tiles):
            for x0, y0, x1, y1 in rects:
                mx = (xs > x0 - robot_radius) & (xs < x1 + robot_radius)
                my = (ys > y0 - robot_radius) & (ys < y1 + robot_radius)
                occ[k][np.ix_(my, mx)] = True
        self.occ = torch.from_numpy(occ).to(dev).reshape(rows, cols, self.h, self.w)
        origins = env.scene.terrain.terrain_origins[..., :2]  # tile centres (the terrain functions return the centre)
        self.corner = origins - torch.tensor([sx / 2, sy / 2], device=dev)  # (rows, cols, 2)
        n = env.num_envs
        self.field = torch.full((n, self.h, self.w), math.inf, device=dev)
        self.env_corner = torch.zeros(n, 2, device=dev)
        self.env_tile = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.best = torch.zeros(n, device=dev)
        self.dev = dev

    def cell(self, corner: torch.Tensor, xy_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = ((xy_w - corner) / self.res).floor().long()
        return idx[..., 1].clamp(0, self.h - 1), idx[..., 0].clamp(0, self.w - 1)  # row (y), column (x)

    def reset(self, env_ids, row, col, goal_w: torch.Tensor, start_w: torch.Tensor, max_iter: int = 3000):
        """Distance field of the envs ``env_ids``: tile (row, col), goal and start in the world frame."""
        n = len(env_ids)
        occ = self.occ[row, col]  # (n, H, W)
        corner = self.corner[row, col]
        gj, gi = self.cell(corner, goal_w)
        field = torch.full((n, self.h, self.w), math.inf, device=self.dev)
        field[torch.arange(n, device=self.dev), gj, gi] = 0.0
        d1, d2 = self.res, self.res * math.sqrt(2.0)
        for it in range(max_iter):
            p = F.pad(field, (1, 1, 1, 1), value=math.inf)
            n4 = torch.minimum(
                torch.minimum(p[:, :-2, 1:-1], p[:, 2:, 1:-1]), torch.minimum(p[:, 1:-1, :-2], p[:, 1:-1, 2:])
            )
            n8 = torch.minimum(
                torch.minimum(p[:, :-2, :-2], p[:, :-2, 2:]), torch.minimum(p[:, 2:, :-2], p[:, 2:, 2:])
            )
            new = torch.minimum(field, torch.minimum(n4 + d1, n8 + d2))
            new = torch.where(occ, torch.full_like(new, math.inf), new)
            if it % 25 == 24 and torch.equal(new, field):
                break
            field = new
        self.field[env_ids] = field
        self.env_corner[env_ids] = corner
        self.env_tile[env_ids, 0], self.env_tile[env_ids, 1] = row, col
        d = self.lookup(env_ids, start_w)
        self.best[env_ids] = torch.where(torch.isinf(d), (goal_w - start_w).norm(dim=-1), d)

    def lookup(self, env_ids, xy_w: torch.Tensor) -> torch.Tensor:
        """Path distance to the goal at ``xy_w`` (the smallest value in the 3 x 3 cells around it: the robot may stand
        in an inflated cell next to a wall). inf if all of them are blocked."""
        j, i = self.cell(self.env_corner[env_ids], xy_w)
        out = torch.full((len(xy_w),), math.inf, device=self.dev)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                v = self.field[env_ids, (j + dj).clamp(0, self.h - 1), (i + di).clamp(0, self.w - 1)]
                out = torch.minimum(out, v)
        return out

    def facing_wall_yaw(self, row, col, start_w: torch.Tensor, lo: float = 0.5, hi: float = 2.5, angles: int = 24):
        """A start heading that looks at a wall ``lo``..``hi`` m ahead (distance to the inflated wall); also returns
        whether one exists."""
        n = len(start_w)
        a = torch.arange(angles, device=self.dev) * (2 * math.pi / angles) - math.pi
        steps = (torch.arange(1, 17, device=self.dev) * 0.25)  # up to 4 m
        direction = torch.stack([a.cos(), a.sin()], dim=-1)  # (A, 2)
        pts = start_w[:, None, None, :] + steps[None, None, :, None] * direction[None, :, None, :]  # (n, A, S, 2)
        j, i = self.cell(self.corner[row, col][:, None, None, :], pts)
        hit = self.occ[row, col][torch.arange(n, device=self.dev)[:, None, None], j, i]  # (n, A, S)
        dist = (hit.int().argmax(dim=-1) + 1) * 0.25
        ok = hit.any(dim=-1) & (dist >= lo) & (dist <= hi)
        pick = (torch.rand(n, angles, device=self.dev) * ok).argmax(dim=-1)
        return a[pick], ok.any(dim=-1)
