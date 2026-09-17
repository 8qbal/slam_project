from __future__ import annotations
import numpy as np
import trimesh
from collections.abc import Callable
from dataclasses import MISSING
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

@configclass
class FlatPatchSamplingCfg:
    num_patches: int = MISSING
    patch_radius: float | list[float] = MISSING
    x_range: tuple[float, float] = (-1e6, 1e6)
    y_range: tuple[float, float] = (-1e6, 1e6)
    z_range: tuple[float, float] = (-1e6, 1e6)
    max_height_diff: float = MISSING

@configclass
class SubTerrainBaseCfg:
    """Base class for terrain configurations with optional USD walls."""
    
    function: Callable[[float, SubTerrainBaseCfg], tuple[list[trimesh.Trimesh], np.ndarray]] = MISSING
    proportion: float = 1.0
    size: tuple[float, float] = (10.0, 10.0)
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    # ----------------- 新增欄位 -----------------
    walls_usd_path: str | None = None
