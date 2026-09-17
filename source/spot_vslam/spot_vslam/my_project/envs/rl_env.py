# # manager_based_rl_env_ros.py
# from __future__ import annotations

# import gymnasium as gym
# import math
# import numpy as np
# import torch
# from collections.abc import Sequence
# from typing import Any, ClassVar

# from isaacsim.core.version import get_version
# from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
# from isaaclab.envs.common import VecEnvStepReturn
# from isaaclab.envs import ManagerBasedRLEnv as BaseManagerBasedRLEnv
# from isaaclab.sensors import Camera, CameraCfg
# from isaaclab.ui.widgets import ManagerLiveVisualizer

# # 導入你自己的 ROS2 Manager
# from ..manager.ros2_manager import Ros2Manager
# from ..manager.orb_slam_subscriber_manager import OrbSlamSubscriberManager


# class ManagerBasedRLEnv(BaseManagerBasedRLEnv, gym.Env):
#     """The superclass for the manager-based workflow reinforcement learning-based environments.

#     This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
#     reinforcement learning-based environments. It is designed to be used with any RL
#     library. The class is designed to be used with vectorized environments, i.e., the
#     environment is expected to be run in parallel with multiple sub-environments. The
#     number of sub-environments is specified using the ``num_envs``.

#     Each observation from the environment is a batch of observations for each sub-
#     environments. The method :meth:`step` is also expected to receive a batch of actions
#     for each sub-environment.

#     While the environment itself is implemented as a vectorized environment, we do not
#     inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
#     various methods (for wait and asynchronous updates) which are not required.
#     Additionally, each RL library typically has its own definition for a vectorized
#     environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
#     here and leave it up to library-defined wrappers to take care of wrapping this
#     environment for their agents.

#     Note:
#         For vectorized environments, it is recommended to **only** call the :meth:`reset`
#         method once before the first call to :meth:`step`, i.e. after the environment is created.
#         After that, the :meth:`step` function handles the reset of terminated sub-environments.
#         This is because the simulator does not support resetting individual sub-environments
#         in a vectorized environment.

#     """

#     is_vector_env: ClassVar[bool] = True
#     """Whether the environment is a vectorized environment."""
#     metadata: ClassVar[dict[str, Any]] = {
#         "render_modes": [None, "human", "rgb_array"],
#         "isaac_sim_version": get_version(),
#     }
#     """Metadata for the environment."""

#     cfg: BaseManagerBasedRLEnv
#     """Configuration for the environment."""

#     def __init__(self, cfg: BaseManagerBasedRLEnv, render_mode: str | None = None, **kwargs):
#         """Initialize the environment.

#         Args:
#             cfg: The configuration for the environment.
#             render_mode: The render mode for the environment. Defaults to None, which
#                 is similar to ``"human"``.
#         """
#         # -- counter for curriculum
#         self.common_step_counter = 0

#         # initialize the episode length buffer BEFORE loading the managers to use it in mdp functions.
#         self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)

#         self.ros2_manager = None
#         self.slam_subscriber_manager = None
#         # initialize the base class to setup the scene.
#         super().__init__(cfg=cfg)
#         # store the render mode
#         self.render_mode = render_mode

#         # initialize data and constants
#         # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
#         self.metadata["render_fps"] = 1 / self.step_dt

#         print("[INFO]: Completed setting up the environment...")

#     """
#     Properties.
#     """

#     @property
#     def max_episode_length_s(self) -> float:
#         """Maximum episode length in seconds."""
#         return self.cfg.episode_length_s

#     @property
#     def max_episode_length(self) -> int:
#         """Maximum episode length in environment steps."""
#         return math.ceil(self.max_episode_length_s / self.step_dt)

#     """
#     Operations - Setup.
#     """

#     def load_managers(self):
#         # note: this order is important since observation manager needs to know the command and action managers
#         # and the reward manager needs to know the termination manager
        
#         # 1. Command Manager
#         self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
#         print("[INFO] Command Manager: ", self.command_manager)

#         # 2. Call parent to load Observation and Action managers
#         super().load_managers()
        
#         # 3. Standard Isaac Lab Managers (Termination, Reward, Curriculum)
#         self.termination_manager = TerminationManager(self.cfg.terminations, self)
#         print("[INFO] Termination Manager: ", self.termination_manager)
        
#         self.reward_manager = RewardManager(self.cfg.rewards, self)
#         print("[INFO] Reward Manager: ", self.reward_manager)
        
#         self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
#         print("[INFO] Curriculum Manager: ", self.curriculum_manager)

#         # --- [修改與整併] 自訂 ROS2 與 SLAM Managers ---
#         print("-" * 30)
#         print("[DEBUG] 開始檢查與載入自訂 Managers (ROS2/SLAM)...")

#         # 初始化為 None，避免未定義變數
#         self.ros2_manager = None
#         self.slam_subscriber_manager = None

#         # A. 載入 Ros2Manager
#         # 使用 getattr(obj, name, default) 是最安全的寫法，防止屬性不存在報錯
#         # 同時檢查是否為 None (我們在 config 設定為 None 的情況)
#         if getattr(self.cfg, "ros2", None) is not None:
#             self.ros2_manager = Ros2Manager(self.cfg.ros2, self)
#             print("[INFO] Ros2 Manager: Loaded Successfully.")
#         else:
#             print("[WARN] 'ros2' config is None or missing. Ros2Manager skipped.")

#         # B. 載入 SLAM Manager
#         if getattr(self.cfg, "slam_subscriber", None) is not None:
#             self.slam_subscriber_manager = OrbSlamSubscriberManager(self.cfg.slam_subscriber, self)
#             print("[INFO] SLAM Subscriber Manager: Loaded Successfully.")
#         else:
#             print("[WARN] 'slam_subscriber' config is None or missing. Manager skipped.")
            
#         print("-" * 30)
#         # --- [修改結束] ---

#         # setup the action and observation spaces for Gym
#         self._configure_gym_env_spaces()

#         # perform events at the start of the simulation
#         if "startup" in self.event_manager.available_modes:
#             self.event_manager.apply(mode="startup")

#     def setup_manager_visualizers(self):
#         """Creates live visualizers for manager terms."""

#         self.manager_visualizers = {
#             "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
#             "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
#             "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
#             "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
#             "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
#             "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
#         }

#     """
#     Operations - MDP
#     """

#     def step(self, action: torch.Tensor) -> VecEnvStepReturn:
#         """Execute one time-step of the environment's dynamics and reset terminated environments.

#         Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

#         1. Process the actions.
#         2. Perform physics stepping.
#         3. Perform rendering if gui is enabled.
#         4. Update the environment counters and compute the rewards and terminations.
#         5. Reset the environments that terminated.
#         6. Compute the observations.
#         7. Return the observations, rewards, resets and extras.

#         Args:
#             action: The actions to apply on the environment. Shape is (num_envs, action_dim).

#         Returns:
#             A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
#         """
#         # process actions
#         self.action_manager.process_action(action.to(self.device))

#         self.recorder_manager.record_pre_step()

#         # check if we need to do rendering within the physics loop
#         # note: checked here once to avoid multiple checks within the loop
#         is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

#         # perform physics stepping
#         for _ in range(self.cfg.decimation):
#             self._sim_step_counter += 1
#             # set actions into buffers
#             self.action_manager.apply_action()
#             # set actions into simulator
#             self.scene.write_data_to_sim()
#             # simulate
#             self.sim.step(render=False)
#             # render between steps only if the GUI or an RTX sensor needs it
#             # note: we assume the render interval to be the shortest accepted rendering interval.
#             #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
#             if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
#                 self.sim.render()
#             # update buffers at sim dt
#             self.scene.update(dt=self.physics_dt)

#         # post-step:
#         # -- update env counters (used for curriculum generation)
#         self.episode_length_buf += 1  # step in current episode (per env)
#         self.common_step_counter += 1  # total step (common for all envs)
#         # -- check terminations
#         self.reset_buf = self.termination_manager.compute()
#         self.reset_terminated = self.termination_manager.terminated
#         self.reset_time_outs = self.termination_manager.time_outs
#         # -- reward computation
#         self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

#         if len(self.recorder_manager.active_terms) > 0:
#             # update observations for recording if needed
#             self.obs_buf = self.observation_manager.compute(update_history=True)
#             self.recorder_manager.record_post_step()

#         # -- reset envs that terminated/timed-out and log the episode information
#         reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
#         if len(reset_env_ids) > 0:
#             # trigger recorder terms for pre-reset calls
#             self.recorder_manager.record_pre_reset(reset_env_ids)

#             self._reset_idx(reset_env_ids)
#             # update articulation kinematics
#             self.scene.write_data_to_sim()
#             self.sim.forward()

#             # if sensors are added to the scene, make sure we render to reflect changes in reset
#             if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
#                 self.sim.render()

#             # trigger recorder terms for post-reset calls
#             self.recorder_manager.record_post_reset(reset_env_ids)

#         # -- update command
#         self.command_manager.compute(dt=self.step_dt)

#         # 呼叫 Ros2Manager 來發布影像
#         if self.ros2_manager:
#             self.ros2_manager.update(dt=self.step_dt)
#         # 更新 SLAM Subscriber
#         if self.slam_subscriber_manager:
#             self.slam_subscriber_manager.update(dt=self.step_dt)

#         # -- step interval events
#         if "interval" in self.event_manager.available_modes:
#             self.event_manager.apply(mode="interval", dt=self.step_dt)
#         # -- compute observations
#         # note: done after reset to get the correct observations for reset envs
#         self.obs_buf = self.observation_manager.compute(update_history=True)

#         # --- ▼▼▼ 在這裡新增 DEBUG 打印 ▼▼▼ ---
#         # if self.common_step_counter % 10 == 1: # 每 10 步打印一次
#         #     print("-" * 50)
#         #     print(f"[DEBUG] Step {self.common_step_counter}")
#         #     if isinstance(self.obs_buf, dict) and "policy" in self.obs_buf:
#         #         # 假設 'policy' group 使用了 concatenate_terms=True
#         #         if torch.is_tensor(self.obs_buf["policy"]):
#         #             actual_shape = self.obs_buf["policy"].shape
#         #             print(f"  Actual obs_buf['policy'] shape: {actual_shape}")
#         #             # 打印設定檔中定義的 shape
#         #             try:
#         #                 expected_shape = self.observation_space["policy"].shape
#         #                 print(f"  Expected observation_space['policy'].shape: {expected_shape}")
#         #                 if actual_shape[-1] != expected_shape[-1]:
#         #                     print(f"  [!!! WARNING !!!] Shape mismatch detected!")
#         #             except Exception as e:
#         #                 print(f"  Could not get expected shape: {e}")
#         #         else:
#         #             print(f"  obs_buf['policy'] is not a tensor: {type(self.obs_buf['policy'])}")
#         #     elif torch.is_tensor(self.obs_buf):
#         #         print(f"  Actual obs_buf shape: {self.obs_buf.shape}")
#         #         print(f"  Expected observation_space.shape: {self.observation_space.shape}")
#         #         if self.obs_buf.shape[-1] != self.observation_space.shape[-1]:
#         #             print(f"  [!!! WARNING !!!] Shape mismatch detected!")
#         #     else:
#         #         print(f"  obs_buf type: {type(self.obs_buf)}")
#         #     print("-" * 50)
#         # --- ▲▲▲ DEBUG 打印結束 ▲▲▲ ---

#         # return observations, rewards, resets and extras
#         return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

#     def render(self, recompute: bool = False) -> np.ndarray | None:
#         """Run rendering without stepping through the physics.

#         By convention, if mode is:

#         - **human**: Render to the current display and return nothing. Usually for human consumption.
#         - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
#           x-by-y pixel image, suitable for turning into a video.

#         Args:
#             recompute: Whether to force a render even if the simulator has already rendered the scene.
#                 Defaults to False.

#         Returns:
#             The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

#         Raises:
#             RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
#                 In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
#                 or ``RenderMode.FULL_RENDERING``.
#             NotImplementedError: If an unsupported rendering mode is specified.
#         """
#         # run a rendering step of the simulator
#         # if we have rtx sensors, we do not need to render again sin
#         if not self.sim.has_rtx_sensors() and not recompute:
#             self.sim.render()
#         # decide the rendering mode
#         if self.render_mode == "human" or self.render_mode is None:
#             return None
#         elif self.render_mode == "rgb_array":
#             # check that if any render could have happened
#             if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
#                 raise RuntimeError(
#                     f"Cannot render '{self.render_mode}' when the simulation render mode is"
#                     f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
#                     f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
#                     " If running headless, make sure --enable_cameras is set."
#                 )
#             # create the annotator if it does not exist
#             if not hasattr(self, "_rgb_annotator"):
#                 import omni.replicator.core as rep

#                 # create render product
#                 self._render_product = rep.create.render_product(
#                     self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
#                 )
#                 # create rgb annotator -- used to read data from the render product
#                 self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
#                 self._rgb_annotator.attach([self._render_product])
#             # obtain the rgb data
#             rgb_data = self._rgb_annotator.get_data()
#             # convert to numpy array
#             rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
#             # return the rgb data
#             # note: initially the renerer is warming up and returns empty data
#             if rgb_data.size == 0:
#                 return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
#             else:
#                 return rgb_data[:, :, :3]
#         else:
#             raise NotImplementedError(
#                 f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
#             )

#     def close(self):
#         if not self._is_closed:
#             # destructor is order-sensitive
#             del self.command_manager
#             del self.reward_manager
#             del self.termination_manager
#             del self.curriculum_manager
#             # call the parent class to close the environment

#             # 先關閉 Subscriber Manager
#             if hasattr(self, "slam_subscriber_manager") and self.slam_subscriber_manager:
#                 self.slam_subscriber_manager.close()
#                 del self.slam_subscriber_manager
#             # 檢查 ros2_manager 是否存在 (在 load_managers 中建立的)
#             if hasattr(self, "ros2_manager") and self.ros2_manager:
#                 # 先呼叫我們在 ros2_manager.py 中定義的 close() 函式
#                 self.ros2_manager.close()
#                 # 再刪除物件
#                 del self.ros2_manager
            
                
#             super().close()

#     # def close(self):
#     #     if not self._is_closed:
#     #         # destructor is order-sensitive

#     #         # --- 1. 只關閉和刪除你自訂的 managers ---

#     #         # 先關閉 Subscriber Manager
#     #         if hasattr(self, "slam_subscriber_manager") and self.slam_subscriber_manager:
#     #             self.slam_subscriber_manager.close()
#     #             del self.slam_subscriber_manager

#     #         # 檢查 ros2_manager 是否存在
#     #         if hasattr(self, "ros2_manager") and self.ros2_manager:
#     #             self.ros2_manager.close()
#     #             del self.ros2_manager

#     #         # --- 2. 呼叫父類別的 close ---
#     #         # super().close() 會自動處理 command_manager, reward_manager 等標準 managers
#     #         super().close()

#     """
#     Helper functions.
#     """

#     def _reset_idx(self, env_ids: Sequence[int]):
#         """Reset environments based on specified indices.

#         Args:
#             env_ids: List of environment ids which must be reset
#         """
#         # update the curriculum for environments that need a reset
#         self.curriculum_manager.compute(env_ids=env_ids)
#         # reset the internal buffers of the scene elements
#         self.scene.reset(env_ids)
#         # apply events such as randomizations for environments that need a reset
#         if "reset" in self.event_manager.available_modes:
#             env_step_count = self._sim_step_counter // self.cfg.decimation
#             self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

#         # iterate over all managers and reset them
#         # this returns a dictionary of information which is stored in the extras
#         # note: This is order-sensitive! Certain things need be reset before others.
#         self.extras["log"] = dict()
#         # -- observation manager
#         info = self.observation_manager.reset(env_ids)
#         self.extras["log"].update(info)
#         # -- action manager
#         info = self.action_manager.reset(env_ids)
#         self.extras["log"].update(info)
#         # -- rewards manager
#         info = self.reward_manager.reset(env_ids)
#         self.extras["log"].update(info)
#         # -- curriculum manager
#         info = self.curriculum_manager.reset(env_ids)
#         self.extras["log"].update(info)
#         # -- command manager
#         info = self.command_manager.reset(env_ids)
#         self.extras["log"].update(info)
#         # -- event manager
#         info = self.event_manager.reset(env_ids)
#         self.extras["log"].update(info)
#         # -- termination manager
#         info = self.termination_manager.reset(env_ids)
#         self.extras["log"].update(info)
#         # -- recorder manager
#         info = self.recorder_manager.reset(env_ids)
#         self.extras["log"].update(info)

#         # --- ▼▼▼ 在這裡新增 ▼▼▼ ---
#         if self.ros2_manager:
#             info = self.ros2_manager.reset(env_ids)
#             self.extras["log"].update(info)
#         # --- ▲▲▲ 新增結束 ▲▲▲ ---
#         # 重置 SLAM Subscriber Manager
#         if self.slam_subscriber_manager:
#             info = self.slam_subscriber_manager.reset(env_ids)
#             if info: # 確保 info 不是 None
#                 self.extras["log"].update(info)
        
#         # reset the episode length buffer
#         self.episode_length_buf[env_ids] = 0
# manager_based_rl_env_ros.py
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from isaacsim.core.version import get_version
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs import ManagerBasedRLEnv as BaseManagerBasedRLEnv
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.ui.widgets import ManagerLiveVisualizer

# 導入你自己的 ROS2 Manager
from ..manager.ros2_manager import Ros2Manager
from ..manager.orb_slam_subscriber_manager import OrbSlamSubscriberManager


class ManagerBasedRLEnv(BaseManagerBasedRLEnv, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: BaseManagerBasedRLEnv
    """Configuration for the environment."""

    def __init__(self, cfg: BaseManagerBasedRLEnv, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # -- counter for curriculum
        self.common_step_counter = 0

        # initialize the episode length buffer BEFORE loading the managers to use it in mdp functions.
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)

        self.ros2_manager = None
        self.slam_subscriber_manager = None
        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)
        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")

    """
    Properties.
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        
        # 1. Command Manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # 2. Call parent to load Observation and Action managers
        super().load_managers()
        
        # 3. Standard Isaac Lab Managers (Termination, Reward, Curriculum)
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # --- [修改與整併] 自訂 ROS2 與 SLAM Managers ---
        print("-" * 30)
        print("[DEBUG] 開始檢查與載入自訂 Managers (ROS2/SLAM)...")

        # 初始化為 None，避免未定義變數
        self.ros2_manager = None
        self.slam_subscriber_manager = None

        # A. 載入 Ros2Manager
        # 使用 getattr(obj, name, default) 是最安全的寫法，防止屬性不存在報錯
        # 同時檢查是否為 None (我們在 config 設定為 None 的情況)
        if getattr(self.cfg, "ros2", None) is not None:
            self.ros2_manager = Ros2Manager(self.cfg.ros2, self)
            print("[INFO] Ros2 Manager: Loaded Successfully.")
        else:
            print("[WARN] 'ros2' config is None or missing. Ros2Manager skipped.")

        # B. 載入 SLAM Manager
        if getattr(self.cfg, "slam_subscriber", None) is not None:
            self.slam_subscriber_manager = OrbSlamSubscriberManager(self.cfg.slam_subscriber, self)
            print("[INFO] SLAM Subscriber Manager: Loaded Successfully.")
        else:
            print("[WARN] 'slam_subscriber' config is None or missing. Manager skipped.")
            
        print("-" * 30)
        # --- [修改結束] ---

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute(update_history=True)
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)

        # 呼叫 Ros2Manager 來發布影像
        if self.ros2_manager:
            self.ros2_manager.update(dt=self.step_dt)
        # 更新 SLAM Subscriber
        if self.slam_subscriber_manager:
            self.slam_subscriber_manager.update(dt=self.step_dt)

        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # --- ▼▼▼ 在這裡新增 DEBUG 打印 ▼▼▼ ---
        # if self.common_step_counter % 10 == 1: # 每 10 步打印一次
        #     print("-" * 50)
        #     print(f"[DEBUG] Step {self.common_step_counter}")
        #     if isinstance(self.obs_buf, dict) and "policy" in self.obs_buf:
        #         # 假設 'policy' group 使用了 concatenate_terms=True
        #         if torch.is_tensor(self.obs_buf["policy"]):
        #             actual_shape = self.obs_buf["policy"].shape
        #             print(f"  Actual obs_buf['policy'] shape: {actual_shape}")
        #             # 打印設定檔中定義的 shape
        #             try:
        #                 expected_shape = self.observation_space["policy"].shape
        #                 print(f"  Expected observation_space['policy'].shape: {expected_shape}")
        #                 if actual_shape[-1] != expected_shape[-1]:
        #                     print(f"  [!!! WARNING !!!] Shape mismatch detected!")
        #             except Exception as e:
        #                 print(f"  Could not get expected shape: {e}")
        #         else:
        #             print(f"  obs_buf['policy'] is not a tensor: {type(self.obs_buf['policy'])}")
        #     elif torch.is_tensor(self.obs_buf):
        #         print(f"  Actual obs_buf shape: {self.obs_buf.shape}")
        #         print(f"  Expected observation_space.shape: {self.observation_space.shape}")
        #         if self.obs_buf.shape[-1] != self.observation_space.shape[-1]:
        #             print(f"  [!!! WARNING !!!] Shape mismatch detected!")
        #     else:
        #         print(f"  obs_buf type: {type(self.obs_buf)}")
        #     print("-" * 50)
        # --- ▲▲▲ DEBUG 打印結束 ▲▲▲ ---

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # call the parent class to close the environment

            # 先關閉 Subscriber Manager
            if hasattr(self, "slam_subscriber_manager") and self.slam_subscriber_manager:
                self.slam_subscriber_manager.close()
                del self.slam_subscriber_manager
            # 檢查 ros2_manager 是否存在 (在 load_managers 中建立的)
            if hasattr(self, "ros2_manager") and self.ros2_manager:
                # 先呼叫我們在 ros2_manager.py 中定義的 close() 函式
                self.ros2_manager.close()
                # 再刪除物件
                del self.ros2_manager
            
                
            super().close()

    # def close(self):
    #     if not self._is_closed:
    #         # destructor is order-sensitive

    #         # --- 1. 只關閉和刪除你自訂的 managers ---

    #         # 先關閉 Subscriber Manager
    #         if hasattr(self, "slam_subscriber_manager") and self.slam_subscriber_manager:
    #             self.slam_subscriber_manager.close()
    #             del self.slam_subscriber_manager

    #         # 檢查 ros2_manager 是否存在
    #         if hasattr(self, "ros2_manager") and self.ros2_manager:
    #             self.ros2_manager.close()
    #             del self.ros2_manager

    #         # --- 2. 呼叫父類別的 close ---
    #         # super().close() 會自動處理 command_manager, reward_manager 等標準 managers
    #         super().close()

    """
    Helper functions.
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # --- ▼▼▼ 在這裡新增 ▼▼▼ ---
        if self.ros2_manager:
            info = self.ros2_manager.reset(env_ids)
            self.extras["log"].update(info)
        # --- ▲▲▲ 新增結束 ▲▲▲ ---
        # 重置 SLAM Subscriber Manager
        if self.slam_subscriber_manager:
            info = self.slam_subscriber_manager.reset(env_ids)
            if info: # 確保 info 不是 None
                self.extras["log"].update(info)
        
        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0