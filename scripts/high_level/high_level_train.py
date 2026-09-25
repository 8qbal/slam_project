from spot_vslam.assets import SPOT_VSLAM_USD_DIR



@configclass
class SpotRoughEnvCfg_HighLevelTrain(SpotRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # 保持跟 low-level checkpoint 一致的時間尺度
        self.decimation = 20
        self.sim.render_interval = self.decimation

        # 多 env 訓練
        # 這裡不要硬寫 4，讓外部 --num_envs 控制
        # self.scene.num_envs = 16   # 不建議寫死

        # 不開 ROS2，避免拖慢與干擾
        self.ros2 = None

        # 需要 maze 就加 maze，但不要走 play 的 ROS 相機
        self.scene.maze = AssetBaseCfg(
            prim_path="/World/Maze",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{SPOT_VSLAM_USD_DIR}/flat_maze.usd"
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )