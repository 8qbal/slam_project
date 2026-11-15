"""Warehouse tiles: parallel rack rows with aisles between them, for evaluating (and later training) aisle navigation.

``warehouse_layout`` is the plan of one tile (pure numpy, tile centred on the origin): rows of solid racks along y, the
aisles between them, a front and a back cross aisle, and sometimes a middle cross aisle that cuts the rows in two.
``warehouse_task`` picks the start (in the front cross aisle) and the goal (deep inside one aisle) for a layout seed.
``warehouse_terrain`` builds the meshes: every rack is a solid block (a robot cannot pass under or through it) with
upright posts, shelf beams and goods boxes sticking out of its faces, so the walls are not plain slabs (cuVSLAM needs
geometry and texture to track, and the depth image has something to look at).

Same seed, same layout: the three policies of the pose-source comparison are evaluated on identical layouts, starts and
goals. ``seed=None`` gives a random layout (training).
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass

from .pathfield import record_tile

__all__ = ["WarehouseTerrainCfg", "warehouse_layout", "warehouse_task", "warehouse_terrain"]

RACK_DEPTH = 0.9
"""Rack row thickness (posts and beams); the solid core is a little thinner."""
RACK_HEIGHT = 2.4
WALL_HEIGHT = 2.5
WALL_THICKNESS = 0.3
FRONT_AISLE = 3.0
"""Width of the cross aisle at the south end (the robot starts there) and of the one at the north end."""
MID_AISLE = 2.4
BOX_PROTRUSION = 0.25
"""Goods boxes stick out this far at most, so an aisle of 2.2 m keeps 1.7 m clear."""


def warehouse_layout(seed: int | None, size: tuple[float, float] = (20.0, 24.0)) -> dict:
    """Plan of one tile, centred on the origin (x -sx/2..sx/2, y -sy/2..sy/2). All lengths in metres.

    Returns ``racks`` (list of (x0, y0, x1, y1) rectangles, one per rack segment), ``aisle_x`` (x of the aisle centres
    between neighbouring rows, left to right), ``y_rack`` (south and north end of the racks), ``mid`` (y of the middle
    cross aisle or None) and ``aisle_width``."""
    sx, sy = size
    rng = np.random.default_rng(seed)
    n_rows = int(rng.integers(3, 6))  # 3-5 rack rows: 2-4 aisles between them
    a_max = (sx - 0.6 - n_rows * RACK_DEPTH) / (n_rows + 1)
    aisle = float(rng.uniform(2.2, min(2.8, a_max)))
    block = n_rows * RACK_DEPTH + (n_rows - 1) * aisle  # first rack face to last rack face
    x_start = -block / 2 + float(rng.uniform(-0.5, 0.5)) * max(sx - 0.6 - block - 2 * aisle, 0.0) / 2
    y_south, y_north = -sy / 2 + FRONT_AISLE, sy / 2 - FRONT_AISLE
    mid = float(rng.uniform(-2.0, 2.0)) if rng.random() < 0.5 else None
    racks, aisle_x = [], []
    for k in range(n_rows):
        x0 = x_start + k * (RACK_DEPTH + aisle)
        if mid is None:
            racks.append((x0, y_south, x0 + RACK_DEPTH, y_north))
        else:
            racks.append((x0, y_south, x0 + RACK_DEPTH, mid - MID_AISLE / 2))
            racks.append((x0, mid + MID_AISLE / 2, x0 + RACK_DEPTH, y_north))
        if k < n_rows - 1:
            aisle_x.append(x0 + RACK_DEPTH + aisle / 2)
    return {"racks": racks, "aisle_x": aisle_x, "y_rack": (y_south, y_north), "mid": mid, "aisle_width": aisle,
            "n_rows": n_rows}


def warehouse_task(seed: int, lateral: int = 1, size: tuple[float, float] = (20.0, 24.0)):
    """Start and goal of layout ``seed``: ((x, y, yaw), (gx, gy)), tile centred on the origin.

    The goal is in the middle of one interior aisle, 2 m before its north end. The start is in the front cross aisle,
    in front of the aisle ``lateral`` places to the side (0: straight in front of the goal aisle, so the robot only has
    to walk down it; 1 or more: it must first move along the cross aisle and turn into the right aisle). The robot
    starts facing the goal."""
    layout = warehouse_layout(seed, size)
    xs = layout["aisle_x"]
    rng = np.random.default_rng(None if seed is None else seed * 7919 + 13)
    g = int(rng.integers(len(xs)))
    s = g + lateral * (1 if g + lateral < len(xs) else -1)
    s = int(np.clip(s, 0, len(xs) - 1))
    y_start = -size[1] / 2 + FRONT_AISLE / 2
    goal = (xs[g], layout["y_rack"][1] - 2.0)
    start = (xs[s], y_start)
    yaw = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    return (start[0], start[1], yaw), goal


def warehouse_terrain(difficulty: float, cfg: WarehouseTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    sx, sy = cfg.size
    layout = warehouse_layout(cfg.seed, cfg.size)
    rng = np.random.default_rng(None if cfg.seed is None else cfg.seed + 1)

    def box(x0, y0, x1, y1, z0, z1):  # corners in the centred frame; the mesh frame starts at the tile corner
        size = (x1 - x0, y1 - y0, z1 - z0)
        centre = ((x0 + x1) / 2 + sx / 2, (y0 + y1) / 2 + sy / 2, (z0 + z1) / 2)
        return trimesh.creation.box(size, trimesh.transformations.translation_matrix(centre))

    t, h = WALL_THICKNESS, WALL_HEIGHT
    meshes = [box(-sx / 2, -sy / 2, sx / 2, sy / 2, -0.1, 0.0)]
    meshes += [  # outer walls
        box(-sx / 2, -sy / 2, -sx / 2 + t, sy / 2, 0.0, h),
        box(sx / 2 - t, -sy / 2, sx / 2, sy / 2, 0.0, h),
        box(-sx / 2, -sy / 2, sx / 2, -sy / 2 + t, 0.0, h),
        box(-sx / 2, sy / 2 - t, sx / 2, sy / 2, 0.0, h),
    ]
    levels = (0.0, 0.6, 1.2, 1.8)  # shelf levels (bottoms of the goods boxes)
    for x0, y0, x1, y1 in layout["racks"]:
        core = 0.08
        meshes.append(box(x0 + core, y0, x1 - core, y1, 0.0, RACK_HEIGHT))  # solid core
        for z in (0.0, 0.6, 1.2, 1.8, RACK_HEIGHT - 0.08):  # shelf beams, full depth
            meshes.append(box(x0, y0, x1, y1, z, z + 0.06))
        n_frames = max(2, int((y1 - y0) / rng.uniform(3.0, 4.5)) + 1)
        for fy in np.linspace(y0 + 0.05, y1 - 0.05, n_frames):  # upright frames on both faces
            meshes.append(box(x0, fy - 0.05, x1, fy + 0.05, 0.0, RACK_HEIGHT + 0.1))
        for face in (-1, 1):  # goods boxes on both faces
            y = y0 + 0.2
            while y < y1 - 0.5:
                length = float(rng.uniform(0.4, 0.9))
                if rng.random() < 0.45:
                    z0 = float(rng.choice(levels)) + 0.06
                    depth = float(rng.uniform(0.05, BOX_PROTRUSION))
                    top = z0 + float(rng.uniform(0.25, 0.5))
                    if face < 0:
                        meshes.append(box(x0 - depth, y, x0 + core, min(y + length, y1 - 0.05), z0, top))
                    else:
                        meshes.append(box(x1 - core, y, x1 + depth, min(y + length, y1 - 0.05), z0, top))
                y += length + float(rng.uniform(0.05, 0.4))
    t = WALL_THICKNESS
    rects = [(0.0, 0.0, t, sy), (sx - t, 0.0, sx, sy), (0.0, 0.0, sx, t), (0.0, sy - t, sx, sy)]  # for PathField
    rects += [(x0 + sx / 2 - BOX_PROTRUSION, y0 + sy / 2, x1 + sx / 2 + BOX_PROTRUSION, y1 + sy / 2) for x0, y0, x1, y1 in layout["racks"]]
    meta = {  # corner frame, for the training resets (``PathField.aisle_task``)
        "aisle_x": [x + sx / 2 for x in layout["aisle_x"]],
        "y_start": FRONT_AISLE / 2,
        "y_goal": layout["y_rack"][1] - 2.0 + sy / 2,
    }
    record_tile((sx, sy), rects, meta)
    return meshes, np.array([sx / 2, sy / 2, 0.0])


@configclass
class WarehouseTerrainCfg(SubTerrainBaseCfg):
    """One tile = one warehouse: rack rows, aisles, cross aisles. ``seed`` fixes the layout (None: random)."""

    function = warehouse_terrain

    seed: int | None = None
