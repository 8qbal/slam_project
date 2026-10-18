"""Random mazes for high-level training, so the policy cannot learn the route of one fixed map.

``maze_terrain`` is a terrain tile with a random grid maze (loops included, all cells connected). ``reset_in_maze``
picks a random tile and a random start / goal inside it at every reset. The paper's training maze (``flat_maze.usd``)
is lost, and training on a map we also evaluate on only measures memorisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["HallTerrainCfg", "MazeTerrainCfg", "hall_terrain", "maze_terrain", "reset_in_maze"]

PATCHES = "init_pos"
"""Key of the flat patches (free spots at least ``patch_radius`` from any wall) used for starts and goals."""


def maze_terrain(difficulty: float, cfg: MazeTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Flat tile with a ``cells`` x ``cells`` maze: a random spanning tree of walls removed, then each other wall
    removed with probability ``loop_prob`` (so there are loops and wide spaces, not only one winding corridor)."""
    sx, sy = cfg.size
    n = cfg.cells
    cx, cy = sx / n, sy / n
    rng = np.random.default_rng()

    # walls between horizontally adjacent cells (east of (i, j)) and vertically adjacent ones (north of (i, j))
    wall_e = np.ones((n, n), dtype=bool)
    wall_n = np.ones((n, n), dtype=bool)
    visited = np.zeros((n, n), dtype=bool)
    stack = [(int(rng.integers(n)), int(rng.integers(n)))]
    visited[stack[0]] = True
    while stack:  # randomised depth-first search: carve a passage to each unvisited neighbour
        i, j = stack[-1]
        free = [
            (di, dj)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= i + di < n and 0 <= j + dj < n and not visited[i + di, j + dj]
        ]
        if not free:
            stack.pop()
            continue
        di, dj = free[rng.integers(len(free))]
        if di:
            wall_e[min(i, i + di), j] = False
        else:
            wall_n[i, min(j, j + dj)] = False
        visited[i + di, j + dj] = True
        stack.append((i + di, j + dj))
    wall_e[:-1] &= rng.random((n - 1, n)) >= cfg.loop_prob  # inner walls only, each may open
    wall_n[:, :-1] &= rng.random((n, n - 1)) >= cfg.loop_prob

    t, h = cfg.wall_thickness, cfg.wall_height

    def box(size, center):
        return trimesh.creation.box(size, trimesh.transformations.translation_matrix(center))

    meshes = [box((sx, sy, 0.1), (sx / 2, sy / 2, -0.05))]
    for i in range(n):
        for j in range(n):
            if i < n - 1 and wall_e[i, j]:  # wall on the east side of cell (i, j)
                meshes.append(box((t, cy + t, h), ((i + 1) * cx, (j + 0.5) * cy, h / 2)))
            if j < n - 1 and wall_n[i, j]:  # wall on the north side
                meshes.append(box((cx + t, t, h), ((i + 0.5) * cx, (j + 1) * cy, h / 2)))
    # solid outer walls
    meshes += [
        box((t, sy, h), (t / 2, sy / 2, h / 2)),
        box((t, sy, h), (sx - t / 2, sy / 2, h / 2)),
        box((sx, t, h), (sx / 2, t / 2, h / 2)),
        box((sx, t, h), (sx / 2, sy - t / 2, h / 2)),
    ]
    return meshes, np.array([sx / 2, sy / 2, 0.0])


@configclass
class MazeTerrainCfg(SubTerrainBaseCfg):
    """One tile = one random maze. Give it ``flat_patch_sampling={PATCHES: ...}`` for starts and goals."""

    function = maze_terrain

    cells: int = 7
    """Cells per side; corridor width is ``size / cells - wall_thickness``."""
    loop_prob: float = 0.3
    """Chance that a wall outside the spanning tree is removed too."""
    wall_thickness: float = 0.3
    wall_height: float = 2.5
    """High enough to fill the depth image's upper rows, like the walls and shelves of the paper maps."""


def hall_terrain(difficulty: float, cfg: HallTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """A hall like the paper maps (docs/DESIGN_DECISIONS.md): rows of long walls across the hall, each with one or two
    doorways, plus free-standing shelves and boxes in the bands between the rows.

    Map A and B are 20 x 30 m with five wall rows 6 m apart, each with a 1.74 m doorway (Map B adds 1.4 m wide shelves
    that narrow the passages). Every shelf keeps ``clearance`` from walls and from each other, so the free space stays
    connected for the robot.
    """
    sx, sy = cfg.size
    rng = np.random.default_rng()
    t, h = cfg.wall_thickness, cfg.wall_height

    def box(size, center):
        return trimesh.creation.box(size, trimesh.transformations.translation_matrix(center))

    meshes = [box((sx, sy, 0.1), (sx / 2, sy / 2, -0.05))]
    rects = []  # (x0, y0, x1, y1) of every wall and shelf, for the clearance test

    def add(x0, y0, x1, y1, height):
        meshes.append(box((x1 - x0, y1 - y0, height), ((x0 + x1) / 2, (y0 + y1) / 2, height / 2)))
        rects.append((x0, y0, x1, y1))

    add(0.0, 0.0, t, sy, h)
    add(sx - t, 0.0, sx, sy, h)
    add(0.0, 0.0, sx, t, h)
    add(0.0, sy - t, sx, sy, h)

    n_rows = int(rng.integers(cfg.num_rows[0], cfg.num_rows[1] + 1))
    ys = np.linspace(0.0, sy, n_rows + 2)[1:-1] + rng.uniform(-0.8, 0.8, n_rows)
    for y in ys:
        n_doors = int(rng.choice([1, 1, 2]))
        width = rng.uniform(*cfg.door_width)
        centers = []
        for _ in range(100):
            c = rng.uniform(width, sx - width)
            if all(abs(c - o) > width + 2.0 for o in centers):
                centers.append(c)
            if len(centers) == n_doors:
                break
        edges = [0.0] + [e for c in sorted(centers) for e in (c - width / 2, c + width / 2)] + [sx]
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a > 0.05:
                add(a, y - t, b, y + t, h)

    n_shelves = int(rng.integers(cfg.num_shelves[0], cfg.num_shelves[1] + 1))
    placed, c = 0, cfg.clearance
    for _ in range(200 * max(n_shelves, 1)):
        if placed == n_shelves:
            break
        width, length = rng.uniform(*cfg.shelf_width), rng.uniform(*cfg.shelf_length)
        if rng.random() < 0.5:
            width, length = length, width
        x0, y0 = rng.uniform(0.0, sx - width), rng.uniform(0.0, sy - length)
        if any(x0 - c < r[2] and x0 + width + c > r[0] and y0 - c < r[3] and y0 + length + c > r[1] for r in rects):
            continue
        add(x0, y0, x0 + width, y0 + length, rng.uniform(*cfg.shelf_height))
        placed += 1
    return meshes, np.array([sx / 2, sy / 2, 0.0])


@configclass
class HallTerrainCfg(SubTerrainBaseCfg):
    """One tile = one hall with wall rows, doorways and shelves (see ``hall_terrain``)."""

    function = hall_terrain

    num_rows: tuple[int, int] = (3, 5)
    """Wall rows across the hall."""
    door_width: tuple[float, float] = (1.3, 2.4)
    """Doorway width; the paper maps have 1.74 m."""
    num_shelves: tuple[int, int] = (0, 4)
    shelf_width: tuple[float, float] = (1.0, 1.6)
    shelf_length: tuple[float, float] = (3.0, 10.0)
    shelf_height: tuple[float, float] = (2.0, 6.0)
    clearance: float = 1.0
    """Free distance around every shelf; keeps the free space connected for the robot."""
    wall_thickness: float = 0.3
    wall_height: float = 6.0


def reset_in_maze(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    goal_distance: tuple[float, float],
    velocity_range: dict[str, tuple[float, float]],
    command_name: str = "goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Move each env to a random maze tile, put the robot on a random free spot there (any heading) and the goal on
    another free spot ``goal_distance`` (min, max) metres away in a straight line (the farthest one if none fits).

    The goal goes into the goal command's ``goal_w``; the command term turns it into the start frame afterwards.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    patches = terrain.flat_patches[PATCHES]  # (rows, cols, P, 3), world frame
    rows, cols, num, _ = patches.shape
    n = len(env_ids)
    dev = env.device

    row = torch.randint(0, rows, (n,), device=dev)
    col = torch.randint(0, cols, (n,), device=dev)
    terrain.terrain_levels[env_ids] = row
    terrain.terrain_types[env_ids] = col
    terrain.env_origins[env_ids] = terrain.terrain_origins[row, col]

    tile = patches[row, col]  # (n, P, 3)
    start_id = torch.randint(0, num, (n,), device=dev)
    start = tile[torch.arange(n, device=dev), start_id]
    dist = torch.linalg.norm(tile[..., :2] - start[:, None, :2], dim=-1)  # (n, P)
    ok = (dist >= goal_distance[0]) & (dist <= goal_distance[1])
    ok[torch.arange(n, device=dev), start_id] = False
    ok = torch.where(ok.any(dim=1, keepdim=True), ok, dist == dist.max(dim=1, keepdim=True).values)
    goal_id = torch.multinomial(ok.float(), 1).squeeze(1)
    goal = tile[torch.arange(n, device=dev), goal_id]
    env.command_manager.get_term(command_name).goal_w[env_ids] = goal[:, :2]

    pos = start.clone()
    pos[:, 2] += asset.data.default_root_pose.torch[env_ids, 2]  # spawn height; x / y come from the patch
    yaw = sample_uniform(-3.14, 3.14, (n,), dev)
    zeros = torch.zeros_like(yaw)
    quat = quat_from_euler_xyz(zeros, zeros, yaw)
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in keys], device=dev)
    vel = asset.data.default_root_vel.torch[env_ids] + sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), dev)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
