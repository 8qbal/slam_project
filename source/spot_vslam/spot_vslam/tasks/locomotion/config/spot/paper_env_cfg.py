"""Paper low-level policy (references/main.pdf Sec. 5.1): 353-dim observation, rewards of Eqs. 2-6 with the
Table 3 weights. The terms the paper doesn't write out keep Isaac Lab's Spot functions.

The ``HighLevel`` configs are the low-level env under the paper's SB3 high-level wrapper
(``spot_vslam.legacy.high_level.paper_vec_env``), on a paper map with its fixed start and goal.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from spot_vslam.assets import PAPER_MAP_SPACING, PAPER_MAPS

from ... import mdp
from ...locomotion_env_cfg import (
    GROUND_MATERIAL_CFG,
    SLAM_CAMERA_CFG,
    LocomotionEnvCfg,
    LocomotionEnvCfg_PLAY,
    RewardsCfg,
)


@configclass
class PaperRewardsCfg(RewardsCfg):
    """Table 3 weights, with the terms the paper writes out (Eqs. 2-6) replaced by those equations."""

    base_linear_velocity = RewTerm(func=mdp.base_linear_velocity_exp, weight=5.0, params={"sigma": 1.0})
    gait = RewTerm(
        func=mdp.gait_phase_indicator,
        weight=12.0,
        params={
            "velocity_threshold": 0.5,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["fl_foot", "hr_foot", "fr_foot", "hl_foot"], preserve_order=True
            ),
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_hinge,
        weight=0.5,
        params={"h_min": 0.1, "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )
    visual_wall_penalty = RewTerm(
        func=mdp.wall_min_depth, weight=-0.1, params={"sensor_cfg": SceneEntityCfg("train_camera")}
    )
    foot_slip = RewTerm(
        func=mdp.foot_contact_velocity_sq,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )


@configclass
class SpotPaperEnvCfg(LocomotionEnvCfg):
    """Low-level training (``Spot-Paper-v0``)."""

    rewards: PaperRewardsCfg = PaperRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 8.0


@configclass
class SpotPaperEnvCfg_PLAY(LocomotionEnvCfg_PLAY):
    rewards: PaperRewardsCfg = PaperRewardsCfg()


def use_paper_map(cfg: LocomotionEnvCfg, map_name: str):
    """Give every env its own copy of a paper map (``PAPER_MAPS``), with the robot at the map's start.

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
        physics_material=GROUND_MATERIAL_CFG,
        debug_vis=False,
    )
    cfg.scene.maze = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Maze", spawn=sim_utils.UsdFileCfg(usd_path=m.usd_path))
    # the terrain-level curriculum needs terrain tiles; there is only one flat floor here
    cfg.curriculum.terrain_levels = None
    cfg.scene.robot.init_state.pos = (m.start[0], m.start[1], 0.6)


@configclass
class SpotPaperHighLevelEnvCfg(SpotPaperEnvCfg):
    """Low-level env under SB3 high-level training: no SLAM (the wrapper fakes it from the GT pose), no ROS 2."""

    paper_map: str = "A"

    def __post_init__(self):
        super().__post_init__()
        # Table 6 caps episodes at 400 HL steps (= 800 low-level steps = 32 s); the low-level 20 s timeout would cut
        # them at 250 HL steps.
        self.episode_length_s = 40.0
        use_paper_map(self, self.paper_map)


@configclass
class SpotPaperHighLevelEnvCfg_MAP_B(SpotPaperHighLevelEnvCfg):
    paper_map: str = "B"


@configclass
class SpotPaperHighLevelEnvCfg_PLAY(SpotPaperEnvCfg_PLAY):
    """Evaluation with ORB-SLAM3 (ROS 2 on)."""

    paper_map: str = "A"

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.tilted_camera = SLAM_CAMERA_CFG.replace(update_period=0.01)
        use_paper_map(self, self.paper_map)
        # Table 9 counts collisions during a run (Map B: 0.4 per run at 100% success), so a bump must not end the
        # episode; load_high_level.py --paper counts body / leg contacts instead. A robot that falls times out.
        self.terminations.body_contact = None
        self.terminations.leg_contact = None


@configclass
class SpotPaperHighLevelEnvCfg_MAP_B_PLAY(SpotPaperHighLevelEnvCfg_PLAY):
    paper_map: str = "B"
