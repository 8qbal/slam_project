"""Spot high-level navigation on the paper maps (references/main.pdf Sec. 5.2, Tables 5-6).

High level at 12.5 Hz: action (vx, wz) -> EMA -> frozen paper low-level policy at 25 Hz. Observation (9, Table 5):
pose (x, y, yaw) in the start frame, goal dx/dy, goal heading error, base vx/vy, base wz.

The pose source is the experiment knob: training uses the ground-truth pose with noise (condition A); ``_PLAY``
evaluates with the ORB-SLAM3 pose, like the paper; ``_GT_PLAY`` evaluates with the ground-truth pose, without ROS 2.
"""

from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

from spot_vslam.assets import PAPER_MAPS
from spot_vslam.slam import CuVslamManagerCfg, Ros2ManagerCfg
from spot_vslam.tasks.locomotion.config.spot.paper_env_cfg import (
    SpotPaperHighLevelEnvCfg,
    SpotPaperHighLevelEnvCfg_PLAY,
)

from ... import mdp

LOW_LEVEL_TASK = "Spot-Paper-v0"
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
        self.ros2 = Ros2ManagerCfg(camera_name="tilted_camera", topic_prefix="/spot")
        self.scene.num_envs = 1
        self.episode_length_s = 3000 * HL_DT  # one run of the navigation test
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
    """Like ``_PLAY`` but with in-process cuVSLAM (RGB-D, loop closure) instead of ROS 2 / ORB-SLAM3.

    See docs/cuvslam.md."""

    def __post_init__(self):
        super().__post_init__()
        self.ros2 = None
        self.cuvslam = CuVslamManagerCfg(camera_name="tilted_camera")


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
