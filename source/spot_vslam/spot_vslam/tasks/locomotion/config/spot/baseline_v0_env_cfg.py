"""rl_vslam_v0 baseline (git tag ``baseline/rl-vslam-v0``, ``baselines/rl_vslam_v0/``).

Same MDP as the base config, but with a 48x64 depth camera: 12x16 = 192 pooled depth dims, 245 in total. The
``HighLevel`` configs are the low-level env that the SB3 high-level wrapper (``spot_vslam.legacy.high_level``) steps.
Don't change these: the baseline policies must keep loading.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from spot_vslam.assets import WAREHOUSE_USD_PATH
from spot_vslam.slam import Ros2ManagerCfg

from ...locomotion_env_cfg import (
    CAMERA_PINHOLE_CFG,
    DEPTH_CAMERA_CFG,
    GROUND_MATERIAL_CFG,
    SLAM_CAMERA_CFG,
    LocomotionEnvCfg,
    LocomotionEnvCfg_PLAY,
)

V0_DEPTH_CAMERA_CFG = DEPTH_CAMERA_CFG.replace(height=48, width=64)
V0_DEPTH_CAMERA_WITH_RGB_CFG = V0_DEPTH_CAMERA_CFG.replace(data_types=["rgb", "distance_to_image_plane"])
V0_SLAM_CAMERA_CFG = SLAM_CAMERA_CFG.replace(spawn=CAMERA_PINHOLE_CFG)  # 10 m far clip, as the v0 tables were measured


@configclass
class SpotV0EnvCfg(LocomotionEnvCfg):
    """Low-level training (``Spot-Vslam-Depth-v0``)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.train_camera = V0_DEPTH_CAMERA_CFG
        self.scene.tilted_camera = V0_SLAM_CAMERA_CFG


@configclass
class SpotV0EnvCfg_PLAY(LocomotionEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.train_camera = V0_DEPTH_CAMERA_WITH_RGB_CFG


@configclass
class SpotV0HighLevelEnvCfg(SpotV0EnvCfg):
    """Low-level env under the v0 SB3 high-level training (``Spot-Vslam-high-level-v0``)."""

    ros2: Ros2ManagerCfg | None = Ros2ManagerCfg(camera_name="tilted_camera", topic_prefix="/spot")

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.6)
        self.scene.train_camera = V0_DEPTH_CAMERA_WITH_RGB_CFG


@configclass
class SpotV0HighLevelEnvCfg_PLAY(SpotV0HighLevelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 10000.0
        self.observations.policy.enable_corruption = False
        self.scene.tilted_camera = V0_SLAM_CAMERA_CFG.replace(update_period=0.01)


@configclass
class SpotV0HighLevelEnvCfg_PLAY_WAREHOUSE(SpotV0HighLevelEnvCfg_PLAY):
    """Play in Simple_Warehouse/warehouse.usd (paper Map A), robot at (0, 0): the rl_vslam_v0 Tables 7-8 runs.

    One flat 40x40 m tile puts the env origin at (0, 0). With the 5x5 terrain grid the single env sits at a tile
    origin (-8, -16), outside the warehouse (x -12..12, y -18..20.8).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_gen.TerrainGeneratorCfg(
                size=(40.0, 40.0),
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
        self.scene.maze = AssetBaseCfg(
            prim_path="/World/Maze",
            spawn=sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD_PATH),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
