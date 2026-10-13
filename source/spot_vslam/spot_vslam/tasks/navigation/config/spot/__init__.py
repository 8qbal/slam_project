"""Spot high-level navigation tasks on the paper maps."""

import gymnasium as gym

from . import agents

for _map, _suffix in (("A", ""), ("B", "_MAP_B")):
    for _variant, _cfg in (
        ("", ""),
        ("-Play", "_PLAY"),  # ORB-SLAM3 RGB-D
        ("-Stereo-Play", "_STEREO_PLAY"),  # ORB-SLAM3 stereo
        ("-GT-Play", "_GT_PLAY"),
        ("-cuVSLAM-Play", "_CUVSLAM_PLAY"),  # cuVSLAM stereo
    ):
        gym.register(
            id=f"Spot-Nav-Map{_map}{_variant}-v0",
            entry_point="spot_vslam.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:SpotNavigationEnvCfg{_suffix}{_cfg}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
            },
        )
