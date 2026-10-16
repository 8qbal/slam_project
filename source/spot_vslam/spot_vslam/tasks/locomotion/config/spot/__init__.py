"""Spot low-level tasks. Task ids are kept from earlier versions so trained policies load under the same id."""

import gymnasium as gym

from . import agents

ENTRY_POINT = "spot_vslam.envs:ManagerBasedRLEnv"


def _register(task_id: str, env_cfg: str, agent_cfg: str):
    gym.register(
        id=task_id,
        entry_point=ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{env_cfg}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:{agent_cfg}",
        },
    )


##
# Paper (references/main.pdf): 353-dim observation
##

PAPER_AGENT = "rl_games_paper_ppo_cfg.yaml"
PAPER_MIRROR_AGENT = "rl_games_paper_mirror_ppo_cfg.yaml"  # + left-right mirror loss
_register("Spot-Paper-v0", "paper_env_cfg:SpotPaperEnvCfg", PAPER_AGENT)
_register("Spot-Paper-Play-v0", "paper_env_cfg:SpotPaperEnvCfg_PLAY", PAPER_AGENT)
# same reward and PPO, other training terrain
_register("Spot-Paper-Rough-v0", "paper_env_cfg:SpotPaperRoughEnvCfg", PAPER_AGENT)
_register("Spot-Paper-Obstacles-v0", "paper_env_cfg:SpotPaperObstaclesEnvCfg", PAPER_AGENT)
_register("Spot-Paper-Walls-v0", "paper_env_cfg:SpotPaperWallsEnvCfg", PAPER_AGENT)
_register("Spot-Paper-WallsSoft-v0", "paper_env_cfg:SpotPaperWallsSoftEnvCfg", PAPER_AGENT)
_register("Spot-Paper-WallsSoftNoise-v0", "paper_env_cfg:SpotPaperWallsSoftNoiseEnvCfg", PAPER_AGENT)
_register("Spot-Paper-WallsSoftSym-v0", "paper_env_cfg:SpotPaperWallsSoftSymEnvCfg", PAPER_AGENT)
_register("Spot-Paper-WallsSoftTrot-v0", "paper_env_cfg:SpotPaperWallsSoftTrotEnvCfg", PAPER_AGENT)
_register("Spot-Paper-WallsSoftMirror-v0", "paper_env_cfg:SpotPaperWallsSoftMirrorEnvCfg", PAPER_MIRROR_AGENT)
_register("Spot-Paper-WallsSoftMirrorTurn-v0", "paper_env_cfg:SpotPaperWallsSoftMirrorTurnEnvCfg", PAPER_MIRROR_AGENT)
# low-level env under the SB3 high-level wrapper, on paper Map A / B
_register("Spot-Paper-high-level-MapA-v0", "paper_env_cfg:SpotPaperHighLevelEnvCfg", PAPER_AGENT)
_register("Spot-Paper-high-level-MapA-Play-v0", "paper_env_cfg:SpotPaperHighLevelEnvCfg_PLAY", PAPER_AGENT)
_register("Spot-Paper-high-level-MapB-v0", "paper_env_cfg:SpotPaperHighLevelEnvCfg_MAP_B", PAPER_AGENT)
_register("Spot-Paper-high-level-MapB-Play-v0", "paper_env_cfg:SpotPaperHighLevelEnvCfg_MAP_B_PLAY", PAPER_AGENT)

##
# rl_vslam_v0 baseline: 245-dim observation
##

V0_AGENT = "rl_games_baseline_v0_ppo_cfg.yaml"
_register("Spot-Vslam-Depth-v0", "baseline_v0_env_cfg:SpotV0EnvCfg", V0_AGENT)
_register("Spot-Vslam-Depth-Play-v0", "baseline_v0_env_cfg:SpotV0EnvCfg_PLAY", V0_AGENT)
_register("Spot-Vslam-high-level-v0", "baseline_v0_env_cfg:SpotV0HighLevelEnvCfg", V0_AGENT)
_register("Spot-Vslam-high-level-Play-v0", "baseline_v0_env_cfg:SpotV0HighLevelEnvCfg_PLAY", V0_AGENT)
_register(
    "Spot-Vslam-high-level-Warehouse-Play-v0", "baseline_v0_env_cfg:SpotV0HighLevelEnvCfg_PLAY_WAREHOUSE", V0_AGENT
)
