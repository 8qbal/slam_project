"""Own sub-terrain: long, tall walls like the shelves and walls of the paper maps (docs/low_level_reward.md)."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass


def walls_terrain(difficulty: float, cfg: WallsTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Flat ground with randomly placed and turned walls; the middle (``platform_radius``) stays free for the spawn.

    Harder tiles get more walls (``num_walls`` start -> end). Every wall lies fully inside the tile, so it never blocks
    the spawn of a neighbouring tile.
    """
    sx, sy = cfg.size
    num = int(round(cfg.num_walls[0] + difficulty * (cfg.num_walls[1] - cfg.num_walls[0])))
    ground = trimesh.creation.box((sx, sy, 0.1), trimesh.transformations.translation_matrix((sx / 2, sy / 2, -0.05)))
    meshes = [ground]
    rng = np.random.default_rng()
    center = np.array([sx / 2, sy / 2])
    for _ in range(100 * num):
        if len(meshes) > num:
            break
        length, thick, height = (rng.uniform(*r) for r in (cfg.length_range, cfg.thickness_range, cfg.height_range))
        yaw = rng.uniform(0.0, np.pi)
        half = 0.5 * np.array([length, thick])
        # half extent along x / y of the turned wall, so it stays inside the tile
        ext = np.abs(np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])) @ half
        if np.any(2 * ext + 2 * cfg.border_width >= (sx, sy)):
            continue
        pos = rng.uniform(ext + cfg.border_width, np.array([sx, sy]) - ext - cfg.border_width)
        # distance from the tile middle to the wall (as a rectangle) must exceed the platform radius
        local = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ (center - pos)
        if np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)) < cfg.platform_radius:
            continue
        tf = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
        tf[:3, 3] = (pos[0], pos[1], height / 2)
        meshes.append(trimesh.creation.box((length, thick, height), tf))
    return meshes, np.array([sx / 2, sy / 2, 0.0])


@configclass
class WallsTerrainCfg(SubTerrainBaseCfg):
    """Flat tile with long walls (shelf / wall pieces) at random positions and angles."""

    function = walls_terrain

    num_walls: tuple[int, int] = (2, 6)
    """Walls on the easiest / hardest tile."""
    length_range: tuple[float, float] = (1.5, 5.0)
    thickness_range: tuple[float, float] = (0.2, 1.0)
    height_range: tuple[float, float] = (1.5, 3.0)
    platform_radius: float = 1.5
    """Free radius around the tile middle, where the robot spawns."""
    border_width: float = 0.25
