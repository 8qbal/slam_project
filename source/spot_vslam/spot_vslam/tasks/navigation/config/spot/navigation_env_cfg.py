"""Spot high-level navigation on the paper maps (references/main.pdf Sec. 5.2, Tables 5-6).

High level at 12.5 Hz: action (vx, wz) -> EMA -> frozen paper low-level policy at 25 Hz. Observation (9, Table 5):
pose (x, y, yaw) in the start frame, goal dx/dy, goal heading error, base vx/vy, base wz.

The pose source is the experiment knob: training uses the ground-truth pose with noise (condition A); ``_PLAY``
evaluates with the ORB-SLAM3 pose, like the paper; ``_GT_PLAY`` evaluates with the ground-truth pose, without ROS 2.
"""

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


def random_world_low_level(paper_map: str, sub_terrains: dict, tile_size: tuple[float, float], goal_distance):
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
        params={"goal_distance": goal_distance, "velocity_range": reset_base.params["velocity_range"]},
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
        # same cameras as the cuVSLAM Play tasks; an RGB-D camera in the scene needs the policy camera to render RGB too
        self.scene.train_camera = DEPTH_CAMERA_WITH_RGB_CFG
        self.scene.tilted_camera = SLAM_CAMERA_CFG.replace(update_period=0.01)
        self.scene.right_camera = SLAM_RIGHT_CAMERA_CFG.replace(update_period=0.01)
        self.cuvslam = CuVslamManagerCfg(camera_name="tilted_camera", right_camera_name="right_camera")
        self.observations.policy.enable_corruption = False
        self.observations.policy.use_pose_source("slam")
        # The plain grey terrain gives cuVSLAM nothing to track (median error 8.8 m, 83 % of the steps above 1.5 m on
        # 2026-10-20; Map A: 0.3 m). Textured walls and floor; the policy sees depth only, so GT tasks keep the old
        # look.
        self.scene.terrain.visual_material = sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.75, 0.9),
        )


@configclass
class SpotNavigationHallRacksCuVslamQualityEnvCfg(SpotNavigationHallRacksCuVslamEnvCfg):
    """``Spot-Nav-HallRacks-cuVSLAM-Quality-v0``: condition C. Condition B plus three cuVSLAM quality inputs
    (``mdp.slam_quality``: features, averaged features, odometry vs loop-closure gap), 12 inputs in all. The checkpoint
    of ``Spot-Nav-HallRacks-v0`` has 9, so widen it first with ``scripts/tools/extend_checkpoint_obs.py --extra 3``."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.slam_quality = ObsTerm(func=mdp.slam_quality)  # last term: the new input columns
