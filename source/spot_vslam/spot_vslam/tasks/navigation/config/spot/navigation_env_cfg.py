"""Spot high-level navigation on the paper maps (references/main.pdf Sec. 5.2, Tables 5-6).

High level at 12.5 Hz: action (vx, wz) -> EMA -> frozen paper low-level policy at 25 Hz. Observation (9, Table 5):
pose (x, y, yaw) in the start frame, goal dx/dy, goal heading error, base vx/vy, base wz.

The pose source is the experiment knob: training uses the ground-truth pose with noise (condition A); ``_PLAY``
evaluates with the ORB-SLAM3 pose, like the paper; ``_GT_PLAY`` evaluates with the ground-truth pose, without ROS 2.
"""

import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.visualizers import VisualizerCfg

from spot_vslam.assets import PAPER_MAPS
from spot_vslam.slam import CuVslamManagerCfg, Ros2ManagerCfg
from spot_vslam.tasks.locomotion.config.spot.paper_env_cfg import (
    SpotPaperHighLevelEnvCfg,
    SpotPaperHighLevelEnvCfg_PLAY,
)
from spot_vslam.tasks.locomotion.locomotion_env_cfg import (
    DEPTH_CAMERA_WITH_RGB_CFG,
    GROUND_MATERIAL_CFG,
    SLAM_CAMERA_CFG,
    SLAM_RIGHT_CAMERA_CFG,
)

from ... import mdp

LOW_LEVEL_TASK = "Spot-Paper-WallsSoftMirrorTurn-v0"
"""Task of the frozen low level (its rl_games agent cfg builds the network): ``policies/walls_soft_mirror_turn`` (no
limp, turns in place). All paper low levels share the 353-dim obs and network, so any of their checkpoints loads."""
LOW_LEVEL_DECIMATION = 20  # physics steps per low-level action (25 Hz)
HIGH_LEVEL_DECIMATION = 2  # low-level steps per high-level action (12.5 Hz, Table 6)
HL_DT = 0.002 * LOW_LEVEL_DECIMATION * HIGH_LEVEL_DECIMATION
"""High-level step (0.08 s). Isaac Lab scales rewards by the step time, so weights are per-step values / HL_DT."""


@configclass
class ActionsCfg:
    pre_trained_policy: mdp.LowLevelPolicyActionCfg = MISSING  # set from the low-level env cfg


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        pose = ObsTerm(func=mdp.robot_pose, params={"pose_source": "gt"}, noise=Gnoise(std=0.01))
        goal_position = ObsTerm(func=mdp.goal_position, params={"pose_source": "gt"})
        goal_heading_error = ObsTerm(func=mdp.goal_heading_error, params={"pose_source": "gt"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel_xy)
        base_yaw_rate = ObsTerm(func=mdp.base_yaw_rate)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

        def use_pose_source(self, pose_source: str):
            for term in (self.pose, self.goal_position, self.goal_heading_error):
                term.params["pose_source"] = pose_source

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Per high-level step: +5 per metre towards the goal, +10 on success, -0.01 per step, -5 on a fall."""

    progress = RewTerm(func=mdp.goal_progress, weight=5.0 / HL_DT)
    success = RewTerm(func=mdp.goal_reached, weight=10.0 / HL_DT)
    time = RewTerm(func=mdp.is_alive, weight=-0.01 / HL_DT)
    fall = RewTerm(func=mdp.terminated_by, weight=-5.0 / HL_DT, params={"term_names": ["body_contact", "leg_contact"]})


@configclass
class RewardsPathCfg:
    """Rewards of the warehouse + maze training (``Spot-Nav-WarehouseMaze-*``), after GUIDE (Wang et al. 2026, Table I).
    Per high-level step: +5 per metre of new best path distance, +10 on success, -0.01 per step, -0.1 while standing
    still away from the goal, -0.05 * |change of the action|^2, -0.3 while the body or a leg touches something (a soft
    cost: the episode ends only on a real fall, -5)."""

    progress = RewTerm(func=mdp.path_progress, weight=5.0 / HL_DT)
    success = RewTerm(func=mdp.goal_reached, weight=10.0 / HL_DT)
    time = RewTerm(func=mdp.is_alive, weight=-0.01 / HL_DT)
    fall = RewTerm(func=mdp.terminated_by, weight=-5.0 / HL_DT, params={"term_names": ["bad_orientation", "base_too_low"]})
    static = RewTerm(func=mdp.not_moving, weight=-0.1 / HL_DT)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05 / HL_DT)
    contact = RewTerm(func=mdp.body_contact, weight=-0.3 / HL_DT)


@configclass
class TerminationsFallCfg:
    """Ends on the goal, the time limit or a fall; touching a wall does not end the episode."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]), "threshold": 1.0},
    )
    leg_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*leg"]), "threshold": 1.0},
    )


@configclass
class CommandsCfg:
    goal = mdp.GoalCommandCfg(goal=PAPER_MAPS["A"].goal, goal_radius=PAPER_MAPS["A"].goal_radius)


@configclass
class SpotNavigationEnvCfg(ManagerBasedRLEnvCfg):
    paper_map: str = "A"

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None
    ros2: Ros2ManagerCfg | None = None
    cuvslam: CuVslamManagerCfg | None = None

    def low_level_env_cfg(self):
        """The low-level env this task wraps: scene, events, timing, and the frozen policy's action/observation."""
        return SpotPaperHighLevelEnvCfg(paper_map=self.paper_map)

    def __post_init__(self):
        low_level = self.low_level_env_cfg()
        self.scene = low_level.scene
        self.events = low_level.events
        self.sim = low_level.sim
        self.decimation = LOW_LEVEL_DECIMATION * HIGH_LEVEL_DECIMATION
        self.sim.render_interval = LOW_LEVEL_DECIMATION
        self.actions.pre_trained_policy = mdp.LowLevelPolicyActionCfg(
            low_level_task=LOW_LEVEL_TASK,
            low_level_decimation=LOW_LEVEL_DECIMATION,
            low_level_actions=low_level.actions.joint_pos,
            low_level_observations=low_level.observations.policy,
        )

        m = PAPER_MAPS[self.paper_map]
        self.commands.goal.goal, self.commands.goal.goal_radius = m.goal, m.goal_radius

        self.scene.num_envs = 16  # one 200 x 200 m floor holds 16 map copies (50 m apart)
        self.episode_length_s = 400 * HL_DT  # Table 6: 400 high-level steps


@configclass
class SpotNavigationEnvCfg_PLAY(SpotNavigationEnvCfg):
    """One env, no observation noise, ORB-SLAM3 pose (start orbslam3_rgbd_node first), no fall termination: the
    paper counts collisions instead (Table 9)."""

    def low_level_env_cfg(self):
        return SpotPaperHighLevelEnvCfg_PLAY(paper_map=self.paper_map)

    def __post_init__(self):
        super().__post_init__()
        self.ros2 = Ros2ManagerCfg(camera_name="tilted_camera")
        self.commands.goal.debug_vis = True  # green circle + pole at the goal (viewport overlay, cameras do not see it)
        self.scene.num_envs = 1
        self.episode_length_s = 3000 * HL_DT  # one run of the navigation test
        # inside the room (9 m walls and a roof, x -9..9, y -12..18), high in its south-east corner, so that the
        # start and the goal are both in view. The low level's default eye (10.5, 10.5, 0.3) is outside the east wall
        views = {"A": ((8.5, -11.0, 6.0), (-4.0, -3.0, 0.0)), "B": ((8.5, 17.0, 6.0), (-3.0, 10.0, 0.0))}
        eye, lookat = views[self.paper_map]
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=eye, lookat=lookat)
        self.observations.policy.enable_corruption = False
        self.observations.policy.use_pose_source("slam")
        self.terminations.body_contact = None
        self.terminations.leg_contact = None
        self.rewards.fall = None


@configclass
class SpotNavigationEnvCfg_GT_PLAY(SpotNavigationEnvCfg_PLAY):
    """Like ``_PLAY`` but with the ground-truth pose and no ROS 2 / SLAM camera."""

    def __post_init__(self):
        super().__post_init__()
        self.ros2 = None
        self.scene.tilted_camera = None
        self.scene.train_camera = self.scene.train_camera.replace(data_types=["distance_to_image_plane"])
        self.observations.policy.use_pose_source("gt")


@configclass
class SpotNavigationEnvCfg_CUVSLAM_PLAY(SpotNavigationEnvCfg_PLAY):
    """Like ``_PLAY`` but with in-process cuVSLAM (stereo, loop closure) instead of ROS 2 / ORB-SLAM3.

    See docs/cuvslam.md."""

    def __post_init__(self):
        super().__post_init__()
        self.ros2 = None
        # same update period as the left camera (the Play scene renders it at 100 Hz), or every other pair mixes
        # two instants
        self.scene.right_camera = SLAM_RIGHT_CAMERA_CFG.replace(update_period=self.scene.tilted_camera.update_period)
        self.cuvslam = CuVslamManagerCfg(camera_name="tilted_camera", right_camera_name="right_camera")


@configclass
class SpotNavigationEnvCfg_STEREO_PLAY(SpotNavigationEnvCfg_PLAY):
    """Like ``_PLAY`` but ORB-SLAM3 in stereo (start ``orbslam3_stereo.launch.py``): the same left / right pair as
    cuVSLAM in ``_CUVSLAM_PLAY``, for the SLAM comparison. ``_PLAY`` stays RGB-D (paper reproduction)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.right_camera = SLAM_RIGHT_CAMERA_CFG.replace(update_period=self.scene.tilted_camera.update_period)
        self.ros2 = Ros2ManagerCfg(camera_name="tilted_camera", right_camera_name="right_camera")


@configclass
class SpotNavigationEnvCfg_MAP_B(SpotNavigationEnvCfg):
    paper_map: str = "B"


@configclass
class SpotNavigationEnvCfg_MAP_B_PLAY(SpotNavigationEnvCfg_PLAY):
    paper_map: str = "B"


@configclass
class SpotNavigationEnvCfg_MAP_B_GT_PLAY(SpotNavigationEnvCfg_GT_PLAY):
    paper_map: str = "B"


@configclass
class SpotNavigationEnvCfg_MAP_B_CUVSLAM_PLAY(SpotNavigationEnvCfg_CUVSLAM_PLAY):
    paper_map: str = "B"


@configclass
class SpotNavigationEnvCfg_MAP_B_STEREO_PLAY(SpotNavigationEnvCfg_STEREO_PLAY):
    paper_map: str = "B"


@configclass
class SpotNavigationEnvCfg_CUVSLAM_QUALITY_PLAY(SpotNavigationEnvCfg_CUVSLAM_PLAY):
    """``_CUVSLAM_PLAY`` with the 12-dim observation of condition C (``Spot-Nav-HallRacks-cuVSLAM-Quality-v0``)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.slam_quality = ObsTerm(func=mdp.slam_quality)


@configclass
class SpotNavigationEnvCfg_MAP_B_CUVSLAM_QUALITY_PLAY(SpotNavigationEnvCfg_CUVSLAM_QUALITY_PLAY):
    paper_map: str = "B"


MAZE_TILE_SIZE = 21.0
MAZE_TILES = 8
"""Maze tiles per side: 64 different mazes, each 21 x 21 m (7 x 7 cells of 3 m)."""


def random_world_low_level(
    paper_map: str, sub_terrains: dict, tile_size: tuple[float, float], goal_distance, **reset_params
):
    """The low-level env of a navigation task on random terrain tiles: ``reset_in_maze`` picks a random tile, start
    (any heading) and goal ``goal_distance`` (min, max) metres away at every reset. 8 x 8 tiles."""
    cfg = SpotPaperHighLevelEnvCfg(paper_map=paper_map)
    cfg.scene.maze = None
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            size=tile_size,
            border_width=0.0,
            num_rows=MAZE_TILES,
            num_cols=MAZE_TILES,
            use_cache=False,
            sub_terrains=sub_terrains,
        ),
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=GROUND_MATERIAL_CFG,
        debug_vis=False,
    )
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.6)
    reset_base = cfg.events.reset_base
    cfg.events.reset_base = EventTerm(
        func=mdp.reset_in_maze,
        mode="reset",
        params={
            "goal_distance": goal_distance,
            "velocity_range": reset_base.params["velocity_range"],
            **reset_params,
        },
    )
    return cfg


def _patches(radius: float, num: int = 64):
    return {
        mdp.maze.PATCHES: FlatPatchSamplingCfg(
            num_patches=num, patch_radius=radius, max_height_diff=0.5, z_range=(-0.5, 0.5)
        )
    }


@configclass
class SpotNavigationMazeEnvCfg(SpotNavigationEnvCfg):
    """``Spot-Nav-Maze-v0``: training on random mazes with a random start and goal at every reset.

    The paper's training maze is lost, and training on Map A or B makes their evaluation a memory test. Here the
    layout, start (any heading) and goal (6-14 m away in a straight line) change every episode; evaluate on the
    paper maps. Goal radius 1 m (Map A's). Everything else, including the observation, is as in the map tasks.
    """

    def low_level_env_cfg(self):
        return random_world_low_level(
            self.paper_map,
            {"maze": mdp.MazeTerrainCfg(flat_patch_sampling=_patches(0.7))},
            (MAZE_TILE_SIZE, MAZE_TILE_SIZE),
            (6.0, 14.0),
        )


HALL_TILE_SIZE = (22.0, 32.0)
"""Like the paper maps (20 x 30 m)."""


@configclass
class SpotNavigationHallEnvCfg(SpotNavigationEnvCfg):
    """``Spot-Nav-Hall-v0``: random mazes (30 % of the tiles) and halls like Maps A / B (70 %): rows of long walls with
    1.3-2.4 m doorways and free-standing shelves (``mdp.HallTerrainCfg``), goals 8-22 m away. Fine-tuned from the
    ``Spot-Nav-Maze-v0`` policy: Maps A and B stay unseen, but their doorway-and-shelf layout is no longer new."""

    def low_level_env_cfg(self):
        return random_world_low_level(
            self.paper_map,
            {
                "maze": mdp.MazeTerrainCfg(proportion=0.3, flat_patch_sampling=_patches(0.6, 96)),
                "hall": mdp.HallTerrainCfg(proportion=0.7, flat_patch_sampling=_patches(0.6, 96)),
            },
            HALL_TILE_SIZE,
            (8.0, 22.0),
        )


@configclass
class SpotNavigationHallRacksEnvCfg(SpotNavigationEnvCfg):
    """``Spot-Nav-HallRacks-v0``: ``Spot-Nav-Hall-v0`` plus open racks standing in 40 % of the doorways, like Map B's
    shelves (boards from 0.85-1.1 m up, upright frames every 3-5 m, 0.86-1.26 m between the posts): the robot has to
    walk under them. Random width, height and length, not a copy of Map B. Fine-tuned from the Hall policy."""

    def low_level_env_cfg(self):
        return random_world_low_level(
            self.paper_map,
            {
                "maze": mdp.MazeTerrainCfg(proportion=0.3, flat_patch_sampling=_patches(0.6, 96)),
                "hall": mdp.HallTerrainCfg(proportion=0.7, rack_prob=0.4, flat_patch_sampling=_patches(0.6, 96)),
            },
            HALL_TILE_SIZE,
            (8.0, 22.0),
        )


def _apply_cuvslam(cfg):
    """Condition B setup (HallRacks-cuVSLAM and WarehouseMaze-cuVSLAM): pose from one cuVSLAM stereo tracker per env."""
    # same cameras as the cuVSLAM Play tasks; an RGB-D camera in the scene needs the policy camera to render RGB too
    cfg.scene.train_camera = DEPTH_CAMERA_WITH_RGB_CFG
    cfg.scene.tilted_camera = SLAM_CAMERA_CFG.replace(update_period=0.01)
    cfg.scene.right_camera = SLAM_RIGHT_CAMERA_CFG.replace(update_period=0.01)
    cfg.cuvslam = CuVslamManagerCfg(camera_name="tilted_camera", right_camera_name="right_camera")
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.use_pose_source("slam")
    # The plain grey terrain gives cuVSLAM nothing to track (median error 8.8 m, 83 % of the steps above 1.5 m on
    # 2026-10-20; Map A: 0.3 m). Textured walls and floor; the policy sees depth only, so GT tasks keep the old
    # look.
    cfg.scene.terrain.visual_material = sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.75, 0.9),
    )


@configclass
class SpotNavigationHallRacksCuVslamEnvCfg(SpotNavigationHallRacksEnvCfg):
    """``Spot-Nav-HallRacks-cuVSLAM-v0``: condition B of the A / B / C experiment.

    ``Spot-Nav-HallRacks-v0`` with the pose from one in-process cuVSLAM stereo tracker per env instead of the ground
    truth; reward, success and the goal command still use the true pose. No observation noise (the tracker has its
    own). Fine-tuned from the ``Spot-Nav-HallRacks-v0`` policy (``policies/nav_hall_racks``). See docs/ROADMAP.md
    section 5.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_cuvslam(self)


@configclass
class SpotNavigationHallRacksCuVslamQualityEnvCfg(SpotNavigationHallRacksCuVslamEnvCfg):
    """``Spot-Nav-HallRacks-cuVSLAM-Quality-v0``: condition C. Condition B plus three cuVSLAM quality inputs
    (``mdp.slam_quality``: features, averaged features, odometry vs loop-closure gap), 12 inputs in all. The checkpoint
    of ``Spot-Nav-HallRacks-v0`` has 9, so widen it first with ``scripts/tools/extend_checkpoint_obs.py --extra 3``."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.slam_quality = ObsTerm(func=mdp.slam_quality)  # last term: the new input columns


class _DepthSectorsMixin:
    """Put before the task config in the bases: appends ``mdp.depth_sectors`` (5 inputs, the last columns) to the
    policy observation. Widen an existing checkpoint with ``scripts/tools/extend_checkpoint_obs.py --extra 5``."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.depth_sectors = ObsTerm(func=mdp.depth_sectors)


class _CuVslamQualityMixin:
    """Appends ``mdp.slam_quality`` (3 inputs) to the policy observation; listed after ``_DepthSectorsMixin`` so that
    the quality columns come before the depth columns."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.slam_quality = ObsTerm(func=mdp.slam_quality)


@configclass
class SpotNavigationHallRacksDepthEnvCfg(_DepthSectorsMixin, SpotNavigationHallRacksEnvCfg):
    """``Spot-Nav-HallRacks-Depth-v0``: ``Spot-Nav-HallRacks-v0`` (ground-truth pose) plus 5 depth-sector inputs
    (14 in all): the high level sees how far the walls are."""


WAREHOUSE_MAZE_TILE = (22.0, 24.0)


@configclass
class SpotNavigationWarehouseMazeEnvCfg(SpotNavigationEnvCfg):
    """Worlds, rewards and terminations of the three ``Spot-Nav-WarehouseMaze-*`` training tasks (GT pose, 9 inputs; the
    tasks add depth sectors, cuVSLAM and quality inputs). Random mazes (50 %, 7 x 7 cells) and random warehouses (50 %: solid rack rows, 2.2-2.8 m aisles) as
    the tiles, goals 6-18 m away in a straight line, 800 high-level steps per episode. Differences from
    ``Spot-Nav-HallRacks-Depth-v0``: path-distance progress (``RewardsPathCfg``), standing-still and contact costs
    instead of ending on a touch, and 30 % of the starts face a wall 0.5-2.5 m ahead."""

    rewards: RewardsPathCfg = RewardsPathCfg()
    terminations: TerminationsFallCfg = TerminationsFallCfg()

    def low_level_env_cfg(self):
        return random_world_low_level(
            self.paper_map,
            {
                "maze": mdp.MazeTerrainCfg(proportion=0.5, flat_patch_sampling=_patches(0.6, 96)),
                "warehouse": mdp.WarehouseTerrainCfg(proportion=0.5, flat_patch_sampling=_patches(0.6, 96)),
            },
            WAREHOUSE_MAZE_TILE,
            (6.0, 18.0),
            path_field=True,
            wall_start_prob=0.3,
        )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 800 * HL_DT


@configclass
class SpotNavigationWarehouseMazeDepthEnvCfg(_DepthSectorsMixin, SpotNavigationWarehouseMazeEnvCfg):
    """``Spot-Nav-WarehouseMaze-Depth-v0``: GT pose + 5 depth sectors (14 inputs)."""


@configclass
class SpotNavigationWarehouseMazeCuVslamEnvCfg(SpotNavigationWarehouseMazeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_cuvslam(self)


@configclass
class SpotNavigationWarehouseMazeCuVslamDepthEnvCfg(_DepthSectorsMixin, SpotNavigationWarehouseMazeCuVslamEnvCfg):
    """``Spot-Nav-WarehouseMaze-cuVSLAM-Depth-v0``: cuVSLAM pose + 5 depth sectors (14 inputs)."""


@configclass
class SpotNavigationWarehouseMazeCuVslamQualityDepthEnvCfg(
    _DepthSectorsMixin, _CuVslamQualityMixin, SpotNavigationWarehouseMazeCuVslamEnvCfg
):
    """``Spot-Nav-WarehouseMaze-cuVSLAM-QualityDepth-v0``: cuVSLAM pose, 3 quality inputs, 5 depth sectors (17)."""


@configclass
class SpotNavigationHallRacksCuVslamDepthEnvCfg(_DepthSectorsMixin, SpotNavigationHallRacksCuVslamEnvCfg):
    """``Spot-Nav-HallRacks-cuVSLAM-Depth-v0``: cuVSLAM pose plus 5 depth-sector inputs (14 in all)."""


@configclass
class SpotNavigationHallRacksCuVslamQualityDepthEnvCfg(_DepthSectorsMixin, SpotNavigationHallRacksCuVslamQualityEnvCfg):
    """``Spot-Nav-HallRacks-cuVSLAM-QualityDepth-v0``: cuVSLAM pose, 3 quality inputs, then 5 depth-sector inputs
    (17 in all, the depth columns last)."""


SMALL_MAZE_CELLS = 4
SMALL_MAZE_CELL_SIZE = 3.0
SMALL_MAZE_SIZE = SMALL_MAZE_CELLS * SMALL_MAZE_CELL_SIZE
"""12 x 12 m: the start-to-goal diagonal (12.7 m between cell centres) is inside the 6-14 m of the maze training."""


def _small_maze_settings() -> tuple[int, float, list[tuple[float, float]]]:
    """Seed, loop probability and the targets (waypoints, then the goal; world xy) of the small maze, from the
    environment: ``SPOT_VSLAM_MAZE_SEED`` (0), ``SPOT_VSLAM_MAZE_LOOPS`` (0.3, higher opens more walls),
    ``SPOT_VSLAM_MAZE_WAYPOINTS=1`` (steer to the corners of the shortest path, default 0: straight at the goal)."""
    seed = int(os.environ.get("SPOT_VSLAM_MAZE_SEED", "0"))
    loops = float(os.environ.get("SPOT_VSLAM_MAZE_LOOPS", "0.3"))
    targets = mdp.maze_waypoints(SMALL_MAZE_CELLS, SMALL_MAZE_CELL_SIZE, seed, loops)
    if os.environ.get("SPOT_VSLAM_MAZE_WAYPOINTS", "0") != "1":
        targets = targets[-1:]
    return seed, loops, targets


class _SmallMazeMixin:
    """The small-maze world for the Play tasks below: put it before the Play config in the bases.

    The policy has no map and steers at its goal with the depth camera. Optional (``SPOT_VSLAM_MAZE_WAYPOINTS=1``):
    feed it the corners of the shortest path one by one (yellow balls; the green disc and pole mark the real goal).
    Measured 2026-10-27 with GT pose: without waypoints 27 / 27 episodes reached the goal in ~230 steps (maze seeds
    1-3), with waypoints about half of the runs, in 330-400 steps: goals 3-6 m away are outside the 6-22 m the
    policies trained on. Env variables: see ``_small_maze_settings``."""

    textured_walls = False
    """SLAM trackers need texture: grey walls gave cuVSLAM a median error of 8.8 m (see HallRacks-cuVSLAM)."""

    def low_level_env_cfg(self):
        seed, loops, targets = _small_maze_settings()
        cfg = SpotPaperHighLevelEnvCfg_PLAY(paper_map=self.paper_map)
        cfg.scene.maze = None
        cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            use_terrain_origins=False,
            env_spacing=cfg.scene.env_spacing,
            terrain_generator=terrain_gen.TerrainGeneratorCfg(
                size=(SMALL_MAZE_SIZE, SMALL_MAZE_SIZE),
                border_width=0.0,
                num_rows=1,
                num_cols=1,
                use_cache=False,
                seed=seed,  # Isaac Lab copies this seed into every sub-terrain; a seed on the sub-terrain alone is overwritten
                sub_terrains={"maze": mdp.MazeTerrainCfg(cells=SMALL_MAZE_CELLS, seed=seed, loop_prob=loops)},
            ),
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=GROUND_MATERIAL_CFG,
            # the default material is black
            visual_material=(
                sim_utils.MdlFileCfg(
                    mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                    project_uvw=True,
                    texture_scale=(0.75, 0.9),
                )
                if self.textured_walls
                else sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.78))
            ),
            debug_vis=False,
        )
        cfg.scene.sky_light.spawn.intensity = 1500.0
        # the maze is centred on the origin (terrain mesh bounds -6..6): the start is the centre of the south-west cell
        near = (SMALL_MAZE_CELL_SIZE - SMALL_MAZE_SIZE) / 2
        yaw = math.atan2(targets[0][1] - near, targets[0][0] - near)  # facing the first waypoint
        cfg.scene.robot.init_state.pos = (near, near, 0.6)
        cfg.scene.robot.init_state.rot = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))  # xyzw
        return cfg

    def __post_init__(self):
        super().__post_init__()
        *waypoints, goal = _small_maze_settings()[2]
        self.commands.goal.goal, self.commands.goal.goal_radius = goal, 1.0
        self.commands.goal.waypoints = tuple(waypoints)
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.0, -10.0, 16.0), lookat=(0.0, 0.0, 0.0))


@configclass
class SpotNavigationSmallMazeGTPlayEnvCfg(_SmallMazeMixin, SpotNavigationEnvCfg_GT_PLAY):
    """``Spot-Nav-SmallMaze-GT-Play-v0``: one small, bright maze with the same layout every time, to watch a
    high-level policy. The robot starts in the south-west cell, the goal (green disc and a pole above the walls) is in
    the north-east cell, the far end of the map; yellow balls mark the waypoints along the way. GT pose, no ROS 2."""


@configclass
class SpotNavigationSmallMazeCuVslamPlayEnvCfg(_SmallMazeMixin, SpotNavigationEnvCfg_CUVSLAM_PLAY):
    """``Spot-Nav-SmallMaze-cuVSLAM-Play-v0``: the small maze with the cuVSLAM stereo pose (in-process)."""

    textured_walls = True


@configclass
class SpotNavigationSmallMazePlayEnvCfg(_SmallMazeMixin, SpotNavigationEnvCfg_PLAY):
    """``Spot-Nav-SmallMaze-Play-v0``: the small maze with the ORB-SLAM3 RGB-D pose (start the node first)."""

    textured_walls = True


@configclass
class SpotNavigationSmallMazeStereoPlayEnvCfg(_SmallMazeMixin, SpotNavigationEnvCfg_STEREO_PLAY):
    """``Spot-Nav-SmallMaze-Stereo-Play-v0``: the small maze with the ORB-SLAM3 stereo pose (start the stereo node)."""

    textured_walls = True


@configclass
class SpotNavigationSmallMazeGTDepthPlayEnvCfg(_DepthSectorsMixin, SpotNavigationSmallMazeGTPlayEnvCfg):
    """``Spot-Nav-SmallMaze-GT-Depth-Play-v0``: the small maze, ground-truth pose, 14 inputs."""


@configclass
class SpotNavigationSmallMazeCuVslamDepthPlayEnvCfg(_DepthSectorsMixin, SpotNavigationSmallMazeCuVslamPlayEnvCfg):
    """``Spot-Nav-SmallMaze-cuVSLAM-Depth-Play-v0``: the small maze, cuVSLAM pose, 14 inputs."""


@configclass
class SpotNavigationSmallMazeCuVslamQualityDepthPlayEnvCfg(
    _DepthSectorsMixin, _CuVslamQualityMixin, SpotNavigationSmallMazeCuVslamPlayEnvCfg
):
    """``Spot-Nav-SmallMaze-cuVSLAM-QualityDepth-Play-v0``: the small maze, cuVSLAM pose, 3 quality + 5 depth inputs
    (17), the same layout as ``Spot-Nav-HallRacks-cuVSLAM-QualityDepth-v0``."""


WAREHOUSE_SIZE = (20.0, 24.0)
"""Like the paper maps (20 x 30 m), a little shorter: start to goal is about 18 m, inside the 6-22 m of the training."""


def _warehouse_settings() -> tuple[int, tuple[float, float, float], tuple[float, float]]:
    """Layout seed, start (x, y, yaw) and goal (x, y) of the warehouse, from the environment:
    ``SPOT_VSLAM_WAREHOUSE_SEED`` (0) and ``SPOT_VSLAM_WAREHOUSE_LATERAL`` (1: the start is one aisle to the side of the
    goal aisle, so the robot has to move along the cross aisle first; 0: straight in front of the goal aisle)."""
    seed = int(os.environ.get("SPOT_VSLAM_WAREHOUSE_SEED", "0"))
    lateral = int(os.environ.get("SPOT_VSLAM_WAREHOUSE_LATERAL", "1"))
    start, goal = mdp.warehouse_task(seed, lateral, WAREHOUSE_SIZE)
    return seed, start, goal


class _WarehouseMixin:
    """The warehouse world for the Play tasks below (put it before the Play config in the bases): one 20 x 24 m tile
    with solid rack rows and aisles (``mdp.warehouse``), the layout fixed by ``SPOT_VSLAM_WAREHOUSE_SEED``. The robot
    starts in the front cross aisle facing the goal, which is deep inside one aisle (green disc). Like the small maze,
    no map: the same seed gives the same layout, start and goal to every policy and pose source."""

    textured_walls = False
    """SLAM trackers need texture: grey walls gave cuVSLAM a median error of 8.8 m (see HallRacks-cuVSLAM)."""

    def low_level_env_cfg(self):
        seed, start, _ = _warehouse_settings()
        cfg = SpotPaperHighLevelEnvCfg_PLAY(paper_map=self.paper_map)
        cfg.scene.maze = None
        cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            use_terrain_origins=False,
            env_spacing=cfg.scene.env_spacing,
            terrain_generator=terrain_gen.TerrainGeneratorCfg(
                size=WAREHOUSE_SIZE,
                border_width=0.0,
                num_rows=1,
                num_cols=1,
                use_cache=False,
                seed=seed,  # Isaac Lab copies this seed into every sub-terrain; a seed on the sub-terrain alone is overwritten
                sub_terrains={"warehouse": mdp.WarehouseTerrainCfg(seed=seed)},
            ),
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=GROUND_MATERIAL_CFG,
            visual_material=(
                sim_utils.MdlFileCfg(
                    mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                    project_uvw=True,
                    texture_scale=(0.75, 0.9),
                )
                if self.textured_walls
                else sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.78))
            ),
            debug_vis=False,
        )
        cfg.scene.sky_light.spawn.intensity = 1500.0
        # the tile is centred on the origin (terrain mesh bounds -10..10, -12..12)
        cfg.scene.robot.init_state.pos = (start[0], start[1], 0.6)
        cfg.scene.robot.init_state.rot = (0.0, 0.0, math.sin(start[2] / 2), math.cos(start[2] / 2))  # xyzw
        return cfg

    def __post_init__(self):
        super().__post_init__()
        self.commands.goal.goal, self.commands.goal.goal_radius = _warehouse_settings()[2], 1.0
        self.commands.goal.waypoints = ()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.0, -16.0, 24.0), lookat=(0.0, 0.0, 0.0))


@configclass
class SpotNavigationWarehouseGTPlayEnvCfg(_WarehouseMixin, SpotNavigationEnvCfg_GT_PLAY):
    """``Spot-Nav-Warehouse-GT-Play-v0``: one warehouse layout (``SPOT_VSLAM_WAREHOUSE_SEED``), ground-truth pose."""


@configclass
class SpotNavigationWarehouseCuVslamPlayEnvCfg(_WarehouseMixin, SpotNavigationEnvCfg_CUVSLAM_PLAY):
    """``Spot-Nav-Warehouse-cuVSLAM-Play-v0``: the warehouse with the cuVSLAM stereo pose (in-process)."""

    textured_walls = True


@configclass
class SpotNavigationWarehouseGTDepthPlayEnvCfg(_DepthSectorsMixin, SpotNavigationWarehouseGTPlayEnvCfg):
    """``Spot-Nav-Warehouse-GT-Depth-Play-v0``: ground-truth pose, 14 inputs (9 + 5 depth sectors)."""


@configclass
class SpotNavigationWarehouseCuVslamDepthPlayEnvCfg(_DepthSectorsMixin, SpotNavigationWarehouseCuVslamPlayEnvCfg):
    """``Spot-Nav-Warehouse-cuVSLAM-Depth-Play-v0``: cuVSLAM pose, 14 inputs."""


@configclass
class SpotNavigationWarehouseCuVslamQualityDepthPlayEnvCfg(
    _DepthSectorsMixin, _CuVslamQualityMixin, SpotNavigationWarehouseCuVslamPlayEnvCfg
):
    """``Spot-Nav-Warehouse-cuVSLAM-QualityDepth-Play-v0``: cuVSLAM pose, 3 quality + 5 depth inputs (17)."""
