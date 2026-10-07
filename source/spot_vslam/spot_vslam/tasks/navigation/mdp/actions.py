"""Action term that runs the frozen low-level policy: the high-level action is a velocity command for it.

Based on Isaac Lab's ``PreTrainedPolicyAction`` (``isaaclab_tasks/contrib/navigation``), with three differences:
it loads the rl_games ``.pth`` checkpoint directly (no TorchScript export), its 2-dim action (forward speed, yaw rate)
is scaled to the paper's ranges, and the command is smoothed with an EMA (paper Table 6).
"""

from __future__ import annotations

import copy
import os
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from rl_games.algos_torch import model_builder, torch_ext

from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import load_cfg_from_registry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RlGamesPolicy(torch.nn.Module):
    """Deterministic action of an rl_games continuous PPO checkpoint, batched.

    Same result as ``BasePlayer.get_action(obs, is_deterministic=True)`` behind Isaac Lab's ``RlGamesVecEnvWrapper``
    with symmetric action clipping: normalized obs -> MLP mean, clamped to [-1, 1] (not clamped when the agent cfg
    sets ``config.clip_actions: False``, as the paper low level does).
    """

    def __init__(self, checkpoint: str, agent_cfg: dict, obs_dim: int, action_dim: int):
        super().__init__()
        params = copy.deepcopy(agent_cfg["params"])
        network = model_builder.ModelBuilder().load(params)
        self.model = network.build(
            {
                "actions_num": action_dim,
                "input_shape": (obs_dim,),
                "num_seqs": 1,
                "value_size": 1,
                "normalize_value": params["config"].get("normalize_value", False),
                "normalize_input": params["config"]["normalize_input"],
            }
        )
        self.model.load_state_dict(torch_ext.load_checkpoint(checkpoint)["model"])
        self.model.eval()
        # rl_games clamps to [-1, 1] unless config.clip_actions is False; the env wrapper clamps to env.clip_actions
        clip = params["config"].get("clip_actions", True) is not False
        self.clip_actions = min(float(params.get("env", {}).get("clip_actions", 1.0)), 1.0) if clip else None

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mu = self.model.a2c_network({"obs": self.model.norm_obs(obs)})[0]
        if self.clip_actions is None:
            return mu
        return torch.clamp(mu, -self.clip_actions, self.clip_actions)


class LowLevelPolicyAction(ActionTerm):
    """High-level action (vx, wz) in [-1, 1]^2 -> EMA-smoothed velocity command -> frozen low-level policy."""

    cfg: LowLevelPolicyActionCfg

    def __init__(self, cfg: LowLevelPolicyActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]

        checkpoint = cfg.checkpoint or os.environ.get("SPOT_VSLAM_LOW_LEVEL_CKPT", "")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"Low-level checkpoint '{checkpoint}' not found. Set SPOT_VSLAM_LOW_LEVEL_CKPT to the low-level .pth"
                " (e.g. logs/rl_games/spot_paper_ppo/<run>/nn/spot_paper_ppo.pth)."
            )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._velocity_command = torch.zeros(self.num_envs, 3, device=self.device)  # (vx, vy = 0, wz)

        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        # the low-level observation reads its last action and its velocity command from this term
        obs_cfg = copy.deepcopy(cfg.low_level_observations)
        obs_cfg.actions.func = lambda env: self.low_level_actions
        obs_cfg.actions.params = dict()
        obs_cfg.velocity_commands.func = lambda env: self._velocity_command
        obs_cfg.velocity_commands.params = dict()
        self._low_level_obs_manager = ObservationManager({"ll_policy": obs_cfg}, env)
        obs_dim = self._low_level_obs_manager.group_obs_dim["ll_policy"][0]

        agent_cfg = load_cfg_from_registry(cfg.low_level_task, "rl_games_cfg_entry_point")
        self.policy = RlGamesPolicy(checkpoint, agent_cfg, obs_dim, self.low_level_actions.shape[1]).to(self.device)
        print(f"[INFO] Low-level policy: {checkpoint} ({obs_dim} obs -> {self.low_level_actions.shape[1]} actions)")

        self._counter = 0

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._velocity_command

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        a = torch.clamp(actions, -1.0, 1.0)
        (vx_min, vx_max), (wz_min, wz_max) = self.cfg.vx_range, self.cfg.wz_range
        command = torch.zeros_like(self._velocity_command)
        command[:, 0] = (a[:, 0] + 1.0) * 0.5 * (vx_max - vx_min) + vx_min
        command[:, 2] = (a[:, 1] + 1.0) * 0.5 * (wz_max - wz_min) + wz_min
        alpha = self.cfg.smoothing_alpha
        self._velocity_command[:] = (1.0 - alpha) * self._velocity_command + alpha * command

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._velocity_command[env_ids] = 0.0
        self.low_level_actions[env_ids] = 0.0
        self._low_level_action_term.reset(env_ids)


@configclass
class LowLevelPolicyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = LowLevelPolicyAction

    asset_name: str = "robot"

    low_level_task: str = MISSING
    """Task id the low-level policy was trained on; gives its rl_games network config."""

    checkpoint: str = ""
    """Low-level rl_games ``.pth``. Empty: ``$SPOT_VSLAM_LOW_LEVEL_CKPT``."""

    low_level_decimation: int = 20
    """Physics steps per low-level action (paper Table 1: 25 Hz at dt = 0.002 s)."""

    low_level_actions: ActionTermCfg = MISSING
    low_level_observations: ObservationGroupCfg = MISSING
    """Observation group of the low-level policy. Its ``actions`` and ``velocity_commands`` terms are fed by this
    action term."""

    vx_range: tuple[float, float] = (0.0, 1.0)
    wz_range: tuple[float, float] = (-0.4, 0.4)
    smoothing_alpha: float = 0.25
    """Command EMA: cmd = (1 - alpha) * cmd + alpha * new (paper Table 6)."""
