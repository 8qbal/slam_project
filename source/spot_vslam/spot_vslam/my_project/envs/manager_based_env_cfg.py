# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RecorderManagerBaseCfg as DefaultEmptyRecorderManagerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from my_project.ros2_observation_term import ROS2ObservationTermCfg
from my_project.ros2_tracking_event_term import ROSTrackingEventTermCfg


def make_spot_observations(num_spots: int):
    """根據輸入數量生成對應的 ROS2 observation term"""
    obs_dict = {}
    for i in range(0, num_spots):
        # 訂閱 Spot 的 local_pc
        obs_dict[f"spot{i}_pc"] = ROS2ObservationTermCfg(
            topic=f"/spot{i}/orbslam/local_pc",
            msg_type="sensor_msgs/msg/PointCloud2",
            obs_dim=None,  # 可在 preprocess 裡處理成固定長度
            preprocess="pointcloud_to_vector",
        )
        # 訂閱 Spot 的 tracking_state
        obs_dict[f"spot{i}_state"] = ROS2ObservationTermCfg(
            topic=f"/spot{i}/orbslam/tracking_state",
            msg_type="std_msgs/msg/String",
            obs_dim=1,
            preprocess="tracking_state_to_int",
        )
    return obs_dict


@configclass
class MyIsaacEnvCfg:
    """完整可動態訂閱多個 Spot 的環境配置"""

    # 動態設定要訂閱的 Spot 數量
    num_spots: int = 1

    # simulation settings
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg()
    ui_window_class_type: type | None = BaseEnvWindow
    seed: int | None = None
    decimation: int = MISSING
    scene: InteractiveSceneCfg = MISSING
    recorders: object = DefaultEmptyRecorderManagerCfg()
    actions: object = MISSING
    rerender_on_reset: bool = False
    wait_for_textures: bool = True
    xr: XrCfg | None = None

    # observations 會在初始化時動態生成
    observations: object = None

    # events
    events: object = None

    def __post_init__(self):
        # 動態生成 observation dict
        self.observations = make_spot_observations(self.num_spots)

        # 事件設定
        self.events = {
            "reset_scene": EventTerm(func=mdp.reset_scene_to_default, mode="reset"),
        }
        # 每個 Spot 的 tracking_state 也可以作為事件使用
        for i in range(0, self.num_spots):
            self.events[f"spot{i}_tracking_state"] = ROSTrackingEventTermCfg(
                topic=f"/spot{i}/orbslam/tracking_state",
                msg_type="std_msgs/msg/String",
            )