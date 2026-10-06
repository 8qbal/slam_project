"""Task configs for the policies described in the paper (references/main.pdf, Sec. 5).

The rl_vslam_v0 baseline was trained on ``use_depth_training``, whose depth camera renders 64x48. That gives a
12x16 = 192-dim depth observation (245 in total) instead of the paper's 15x20 = 300 (353 in total, Table 2). The
depth regions used by ``camera_depth_stats``, ``visual_wall_penalty`` and ``open_space_seeker_reward``
(e.g. ``depth[:, 20:50, 5:75]``) are written for a 60x80 image, and were silently cropped on the 48x64 one.

These configs change the depth camera to 80x60, so the same observation functions produce the paper's 353-dim
observation, and the low-level training task uses the reward equations of Sec. 5.1 (``mdp/paper_rewards.py``) with
the Table 3 weights. Commands, terrain and timing (Table 1) are unchanged from ``use_depth_training``. The paper
renders 320x240 and resizes to 60x80; rendering 80x60 directly gives the same field of view and observation layout
at a fraction of the render cost.

The high-level tasks use the paper's evaluation maps with a fixed start and goal each (``PAPER_MAPS``).

The old task ids are unchanged, so the rl_vslam_v0 policies still load under them.
"""

from dataclasses import dataclass

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

import spot_vslam.mdp.paper_rewards as paper_rewards
from spot_vslam.assets import WAREHOUSE_SHELVES_USD_PATH, WAREHOUSE_USD_PATH
from spot_vslam.managers.ros2_manager import Ros2ManagerCfg

from .use_depth_training import (
    SpotHighLevelTrainEnvCfg,
    SpotHighLevelTrainEnvCfg_Play,
    SpotRoughEnvCfg,
    SpotRoughEnvCfg_Play,
    SpotRewardsCfg,
)

PAPER_DEPTH_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body/train_camera",
    update_period=0.04,  # Table 1: depth update period
    height=60,
    width=80,
    data_types=["distance_to_image_plane"],
    spawn=SpotRoughEnvCfg().scene.train_camera.spawn,  # same pinhole / clipping range (0.1-10 m) as before
    offset=CameraCfg.OffsetCfg(pos=(0.4, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
)
"""Depth camera for the paper's 353-dim low-level observation (300 pooled depth + 5 depth stats + 48)."""

# In Isaac Lab 3.0 a depth-only camera in the scene makes every camera's RGB output black, including tilted_camera
# which feeds ORB-SLAM3, so whenever tilted_camera is present the depth camera also renders RGB (unused by the policy).
PAPER_DEPTH_CAMERA_WITH_RGB_CFG = PAPER_DEPTH_CAMERA_CFG.replace(data_types=["rgb", "distance_to_image_plane"])


# ==========================================================
# Low-level locomotion policy (paper Sec. 5.1)
# ==========================================================
@configclass
class SpotPaperRewardsCfg(SpotRewardsCfg):
    """Table 3 weights, with the terms the paper writes out (Eqs. 2-6) replaced by those equations.

    The other terms have no equation in the paper and keep the original Isaac Lab Spot functions.
    """

    base_linear_velocity = RewTerm(func=paper_rewards.base_linear_velocity_exp, weight=5.0, params={"sigma": 1.0})
    gait = RewTerm(
        func=paper_rewards.gait_phase_indicator,
        weight=12.0,
        params={
            "velocity_threshold": 0.5,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["fl_foot", "hr_foot", "fr_foot", "hl_foot"], preserve_order=True
            ),
        },
    )
    foot_clearance = RewTerm(
        func=paper_rewards.foot_clearance_hinge,
        weight=0.5,
        params={"h_min": 0.1, "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )
    visual_wall_penalty = RewTerm(
        func=paper_rewards.wall_min_depth, weight=-0.1, params={"sensor_cfg": SceneEntityCfg("train_camera")}
    )
    foot_slip = RewTerm(
        func=paper_rewards.foot_contact_velocity_sq,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )


@configclass
class SpotPaperEnvCfg(SpotRoughEnvCfg):
    rewards: SpotPaperRewardsCfg = SpotPaperRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.train_camera = PAPER_DEPTH_CAMERA_CFG
        # The 320x240 RGB-D camera is only used by ORB-SLAM3 / TSDF at play time; the policy never reads it.
        self.scene.tilted_camera = None
        self.scene.env_spacing = 8.0


@configclass
class SpotPaperEnvCfg_Play(SpotRoughEnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()
        self.scene.train_camera = PAPER_DEPTH_CAMERA_WITH_RGB_CFG


# ==========================================================
# High-level navigation policy (paper Sec. 5.2), frozen low-level policy underneath
# ==========================================================
@dataclass(frozen=True)
class PaperMap:
    """A paper evaluation map with its fixed start and goal (world frame, metres)."""

    usd_path: str
    start: tuple[float, float]
    goal: tuple[float, float]
    goal_radius: float


PAPER_MAPS = {
    # Map A "Room": start, goal and radius from the original code (use_depth_training.py spawn comments,
    # test_environment.py --target_x/--target_y/--success_radius).
    "A": PaperMap(WAREHOUSE_USD_PATH, start=(-9.0, -10.0), goal=(-4.0, 15.5), goal_radius=1.0),
    # Map B "Warehouse": goal and radius from the original code. Its start (-22, 3.5) lies outside the asset
    # (x -12..12), so the start is the free corner farthest from the goal (23 m in a straight line).
    "B": PaperMap(WAREHOUSE_SHELVES_USD_PATH, start=(-9.0, 17.0), goal=(7.0, 0.0), goal_radius=2.0),
    # Map C "Corridor" (corrider_map.usd, start (0, 0) -> goal (15, 0.5)) is lost.
}


PAPER_MAP_SPACING = 50.0
"""Distance between env origins. Each env gets its own copy of the map (about 24 x 39 m with solid outer walls),
so robots in different envs never see each other."""


def _use_paper_map(cfg, map_name: str):
    """Give every env its own copy of the map, with the robot at the map's start.

    Env origins follow the scene's env grid (``use_terrain_origins=False``), the map is cloned under each env, and
    start / goal are relative to the env origin. Env 0 sits at (0, 0), so with one env (play) world and map
    coordinates are the same. One flat 200x200 m ground tile covers up to 16 envs.
    """
    m = PAPER_MAPS[map_name]
    cfg.scene.env_spacing = PAPER_MAP_SPACING
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        use_terrain_origins=False,
        env_spacing=PAPER_MAP_SPACING,
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            size=(200.0, 200.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            use_cache=False,
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    cfg.scene.maze = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Maze", spawn=sim_utils.UsdFileCfg(usd_path=m.usd_path))
    # the terrain-level curriculum needs terrain tiles; there is only one flat floor here
    cfg.curriculum.terrain_levels = None
    cfg.scene.robot.init_state.pos = (m.start[0], m.start[1], 0.6)


@configclass
class SpotPaperHighLevelEnvCfg(SpotHighLevelTrainEnvCfg):
    """High-level training: SLAM is faked from the ground-truth pose, so ROS 2 and the ORB-SLAM3 camera are off."""

    ros2: Ros2ManagerCfg = None
    paper_map: str = "A"  # key of PAPER_MAPS; gives the arena, start and goal

    def __post_init__(self):
        super().__post_init__()
        self.scene.train_camera = PAPER_DEPTH_CAMERA_CFG
        self.scene.tilted_camera = None
        # Table 6 caps episodes at 400 HL steps (= 800 low-level steps = 32 s); the low-level 20 s timeout would cut
        # them at 250 HL steps.
        self.episode_length_s = 40.0
        _use_paper_map(self, self.paper_map)


@configclass
class SpotPaperHighLevelEnvCfg_MapB(SpotPaperHighLevelEnvCfg):
    paper_map: str = "B"


@configclass
class SpotPaperHighLevelEnvCfg_Play(SpotHighLevelTrainEnvCfg_Play):
    """Evaluation with ORB-SLAM3 (ROS 2 on)."""

    paper_map: str = "A"

    def __post_init__(self):
        super().__post_init__()
        self.scene.train_camera = PAPER_DEPTH_CAMERA_WITH_RGB_CFG
        _use_paper_map(self, self.paper_map)
        # Table 9 counts collisions during a run (Map B: 0.4 per run at 100% success), so a bump must not end the
        # episode; load_high_level.py --paper counts body / leg contacts instead. A robot that falls times out.
        self.terminations.body_contact = None
        self.terminations.leg_contact = None


@configclass
class SpotPaperHighLevelEnvCfg_MapB_Play(SpotPaperHighLevelEnvCfg_Play):
    paper_map: str = "B"
