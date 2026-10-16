"""Paper low-level policy (references/main.pdf Sec. 5.1): 353-dim observation, Isaac Lab's Spot rewards with the
Table 3 weights, and the paper's wall term (Eq. 5).

The ``HighLevel`` configs are the low-level env under the paper's SB3 high-level wrapper
(``spot_vslam.legacy.high_level.paper_vec_env``), on a paper map with its fixed start and goal.
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from spot_vslam.assets import PAPER_MAP_SPACING, PAPER_MAPS

from ... import mdp
from ...locomotion_env_cfg import (
    GROUND_MATERIAL_CFG,
    SLAM_CAMERA_CFG,
    TERRAIN_CFG,
    LocomotionEnvCfg,
    LocomotionEnvCfg_PLAY,
    RewardsCfg,
)


@configclass
class PaperRewardsCfg(RewardsCfg):
    """Table 3 weights. Eqs. 2-4 and 6 use Isaac Lab's Spot reward functions (inherited), Eq. 5 is our own.

    Table 3 has exactly the weights of Isaac Lab's Spot rewards, so Eqs. 2-6 are read as a summary of those functions.
    Taken literally (binary trot indicator for Eq. 3, weight 12) the policy learned to trot in place and never moved
    (docs/low_level_reward.md).
    """

    visual_wall_penalty = RewTerm(
        func=mdp.wall_min_depth, weight=-0.1, params={"sensor_cfg": SceneEntityCfg("train_camera")}
    )

    def __post_init__(self):
        # Speed tracking (docs/low_level_reward.md): with Isaac Lab's sigma = 1 and commands of only 0-1 m/s, the
        # policy walked at ~0.6 m/s for every command. A sharper kernel makes a 0.35 m/s error cost 3.8 of 5, not 1.5.
        self.base_linear_velocity.params["std"] = 0.25
        # Same for turning: with sigma = 2 a 0.2 rad/s error cost 0.5 of 5, and the policy turned at 40-75 % of the
        # command (weaker to the right). Now 0.2 rad/s costs 1.6.
        self.base_angular_velocity.params["std"] = 0.5
        # Gait and air time also paid when a standing env walked faster than 0.5 m/s (12 + 5 > the speed term), so
        # it never stood still. Enforce them only when a speed is commanded.
        self.gait.params["velocity_threshold"] = float("inf")
        self.air_time.params["velocity_threshold"] = float("inf")


@configclass
class SpotPaperEnvCfg(LocomotionEnvCfg):
    """Low-level training (``Spot-Paper-v0``)."""

    rewards: PaperRewardsCfg = PaperRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 8.0
        self.commands.base_velocity.rel_standing_envs = 0.1  # Isaac Lab: 0.05; more zero commands to learn standing


@configclass
class SpotPaperEnvCfg_PLAY(LocomotionEnvCfg_PLAY):
    rewards: PaperRewardsCfg = PaperRewardsCfg()


# -- training-terrain study: same reward and PPO as Spot-Paper-v0, other terrain (docs/low_level_reward.md)

ROUGH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)
"""Isaac Lab's Spot terrain (``COBBLESTONE_ROAD_CFG``): 9x21 tiles of 8 m, half flat, half rough (2-5 cm)."""

OBSTACLE_TERRAIN_CFG = TERRAIN_CFG.replace(
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.4),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.01, 0.03), noise_step=0.01, border_width=0.25
        ),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.3,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.5, 2.0),
            obstacle_height_range=(1.0, 2.0),
            num_obstacles=8,
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
"""The paper terrain with 30 % of the tiles holding 8 pillars / wall pieces (1-2 m high) the depth camera sees."""

WALLS_TERRAIN_CFG = OBSTACLE_TERRAIN_CFG.replace(
    sub_terrains={
        **OBSTACLE_TERRAIN_CFG.sub_terrains,
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "random_rough": OBSTACLE_TERRAIN_CFG.sub_terrains["random_rough"].replace(proportion=0.2),
        "obstacles": OBSTACLE_TERRAIN_CFG.sub_terrains["obstacles"].replace(proportion=0.25),
        "walls": mdp.WallsTerrainCfg(proportion=0.25),
    },
)
"""The obstacle terrain plus 25 % tiles with long walls (2-7 m long, 2.5-6 m high, any angle) and up to two 6 m
room walls on the tile edges, like the shelves and walls of the paper maps: the obstacle-trained policy stopped
turning on Map A, where a wall near the start fills its whole depth image."""


@configclass
class SpotPaperRoughEnvCfg(SpotPaperEnvCfg):
    """``Spot-Paper-Rough-v0``: Isaac Lab's Spot terrain, all levels from the start and no curriculum, as there."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = ROUGH_TERRAIN_CFG.num_rows - 1
        self.curriculum.terrain_levels = None


@configclass
class SpotPaperObstaclesEnvCfg(SpotPaperEnvCfg):
    """``Spot-Paper-Obstacles-v0``: the paper terrain plus obstacle tiles (same terrain curriculum)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = OBSTACLE_TERRAIN_CFG


@configclass
class SpotPaperWallsEnvCfg(SpotPaperEnvCfg):
    """``Spot-Paper-Walls-v0``: the obstacle terrain plus wall tiles (same terrain curriculum)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = WALLS_TERRAIN_CFG


@configclass
class SpotPaperWallsSoftEnvCfg(SpotPaperWallsEnvCfg):
    """``Spot-Paper-WallsSoft-v0``: the wall terrain, but a body / leg contact costs a penalty instead of ending the
    episode; only a fall ends it. With contact = termination the policy learned to barely move (or turn) near walls,
    and on Map A, whose start is a corner, the high level's turn commands were ignored (docs/low_level_reward.md).
    """

    def __post_init__(self):
        super().__post_init__()
        self.terminations.body_contact = None
        self.terminations.leg_contact = None
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
        self.terminations.base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
        # -5 x dt (0.04) = -0.2 per touching body and step, about the whole per-step reward
        self.rewards.wall_contact = RewTerm(
            func=mdp.undesired_contacts,
            weight=-5.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
        )


@configclass
class SpotPaperWallsSoftNoiseEnvCfg(SpotPaperWallsSoftEnvCfg):
    """``Spot-Paper-WallsSoftNoise-v0``: WallsSoft with noise on every sensor (domain randomization, comparison 3).

    Body velocity, gravity and joints get Isaac Lab's Spot observation noise; the depth image gets range noise,
    missing pixels and edge dropout (``mdp.noisy_camera_depth_obs``). The two depth terms become one term with the same
    305 values in the same order, so the observation stays 353-dim and WallsSoft checkpoints load.
    """

    def __post_init__(self):
        super().__post_init__()
        policy = self.observations.policy
        policy.base_lin_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        policy.base_ang_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        policy.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)
        policy.joint_pos.noise = Unoise(n_min=-0.05, n_max=0.05)
        policy.joint_vel.noise = Unoise(n_min=-0.5, n_max=0.5)
        policy.camera_depth.func = mdp.noisy_camera_depth_obs
        policy.camera_depth_stats = None


@configclass
class SpotPaperWallsSoftSymEnvCfg(SpotPaperWallsSoftEnvCfg):
    """``Spot-Paper-WallsSoftSym-v0``: WallsSoft plus a penalty on uneven trot diagonals (``diagonal_duty``).

    Every paper-version policy limps: FL+HR are down 75 % of the cycle, FR+HL only 25 % (short stamps of the hind left
    leg), and the gait reward does not see it. Fine-tuned from the WallsSoft checkpoint (same 353-dim obs).

    ``nan_guard`` and the ``sim_blowup`` termination: the from-scratch run blew up at ep ~8750 (reward -1e10, then NaN
    weights). Likely cause: the unclipped actions grew (ep 8700 already feeds |action| > 100 back as observation) and
    ``action_smoothness`` squares them. A blown-up env now ends at once, and rewards / observations are made finite
    and clipped. Checked on WallsSoft and ep 8700 (256 envs, 1000 steps): ``sim_blowup`` never fires; normal maxima
    are 5 m/s, 21 rad/s, 63 rad/s joint velocity, 2.7 reward per step.
    """

    nan_guard: bool = True
    nan_guard_reward: float = 10.0  # per step; a normal step is < 1
    nan_guard_obs: float = 100.0  # normal |obs| < 30 (joint velocities)

    def __post_init__(self):
        super().__post_init__()
        # the limp (0.5) costs -2.5 per second, about a fifth of a full gait reward (12)
        self.rewards.diagonal_duty = RewTerm(
            func=mdp.diagonal_duty_penalty,
            weight=-5.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["fl_foot", "fr_foot", "hl_foot", "hr_foot"], preserve_order=True
                )
            },
        )
        self.terminations.sim_blowup = DoneTerm(func=mdp.sim_blowup)


@configclass
class SpotPaperWallsSoftTrotEnvCfg(SpotPaperWallsSoftSymEnvCfg):
    """``Spot-Paper-WallsSoftTrot-v0``: WallsSoft (with the Sym task's ``nan_guard`` / ``sim_blowup``) asking for a
    real trot instead of only even diagonals: every foot down ~50 % of its cycle (``foot_duty``) and each swing at
    least 0.2 s (``short_swing``). ``diagonal_duty`` alone cost the fine-tune its turning and let the from-scratch run
    shuffle at 8 Hz (docs/low_level_reward.md). Fine-tuned from the WallsSoft checkpoint (same 353-dim obs).
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.diagonal_duty = None
        feet = SceneEntityCfg("contact_forces", body_names=".*_foot")
        # gait_check.py, mean weighted value per step (gait = 10.4 / 9.4): WallsSoft limp (duty 0.73 / 0.31) -0.6 at
        # weight -3 -> -1.0; the shuffle has duty ~0.5 (0.06 s air / 0.07 s contact) and is not caught here
        self.rewards.foot_duty = RewTerm(func=mdp.foot_duty_penalty, weight=-5.0, params={"sensor_cfg": feet})
        # one value per touchdown, not per second: the shuffle (8.3 touchdowns/s per foot, 0.06 s swings) -0.32 at
        # weight -2 -> -4.8; the limp's 0.13 s FL / HR swings -0.02 -> -0.3
        self.rewards.short_swing = RewTerm(func=mdp.short_swing_penalty, weight=-30.0, params={"sensor_cfg": feet})


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
        # Fixed start: at the map start, facing the goal, no push. Training's reset adds up to +-0.5 m, any heading
        # and up to 1.5 m/s: the robot slid ~0.4 m before the first command, and some runs started facing out of the
        # building (sky and empty floor), where SLAM has nothing to track (docs/cuvslam.md).
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_base.params["velocity_range"] = {}
        yaw = PAPER_MAPS[self.paper_map].start_yaw
        self.scene.robot.init_state.rot = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))  # xyzw
        # Table 9 counts collisions during a run (Map B: 0.4 per run at 100% success), so a bump must not end the
        # episode; load_high_level.py --paper counts body / leg contacts instead. A robot that falls times out.
        self.terminations.body_contact = None
        self.terminations.leg_contact = None


@configclass
class SpotPaperHighLevelEnvCfg_MAP_B_PLAY(SpotPaperHighLevelEnvCfg_PLAY):
    paper_map: str = "B"
