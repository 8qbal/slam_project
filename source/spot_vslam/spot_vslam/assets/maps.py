"""Paper evaluation maps (references/main.pdf, Sec. 6), each with one fixed start and goal."""

import math
from dataclasses import dataclass

from . import WAREHOUSE_SHELVES_USD_PATH, WAREHOUSE_USD_PATH


@dataclass(frozen=True)
class PaperMap:
    """A paper evaluation map with its fixed start and goal (map frame, metres)."""

    usd_path: str
    start: tuple[float, float]
    goal: tuple[float, float]
    goal_radius: float

    @property
    def start_yaw(self) -> float:
        """Heading at the start in evaluation: facing the goal (rad, map frame)."""
        return math.atan2(self.goal[1] - self.start[1], self.goal[0] - self.start[0])


PAPER_MAPS = {
    # Map A "Room": start, goal and radius from the original code (spawn comments of the original depth-training
    # config, scripts/legacy/high_level/test_environment.py --target_x/--target_y/--success_radius).
    "A": PaperMap(WAREHOUSE_USD_PATH, start=(-9.0, -10.0), goal=(-4.0, 15.5), goal_radius=1.0),
    # Map B "Warehouse": goal and radius from the original code. Its start (-22, 3.5) lies outside the asset
    # (x -12..12), so the start is the free corner farthest from the goal (23 m in a straight line).
    "B": PaperMap(WAREHOUSE_SHELVES_USD_PATH, start=(-9.0, 17.0), goal=(7.0, 0.0), goal_radius=2.0),
    # Map C "Corridor" (corrider_map.usd, start (0, 0) -> goal (15, 0.5)) is lost.
}

PAPER_MAP_SPACING = 50.0
"""Distance between env origins. Each env gets its own copy of the map (about 24 x 39 m with solid outer walls),
so robots in different envs never see each other."""
