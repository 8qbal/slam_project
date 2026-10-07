"""SLAM test env: Spot in the warehouse with one RGB-D camera, ROS 2 and the ORB-SLAM3 subscriber
(``scripts/slam/test_*.py``)."""

import gymnasium as gym

gym.register(
    id="Spot-test180-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.slam_test_env_cfg:SlamTestEnvCfg"},
)
