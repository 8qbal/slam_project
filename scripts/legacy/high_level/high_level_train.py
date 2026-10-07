from spot_vslam.assets import WAREHOUSE_USD_PATH



@configclass
class SpotRoughEnvCfg_HighLevelTrain(SpotRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # Same time scale as the low-level checkpoint
        self.decimation = 20
        self.sim.render_interval = self.decimation

        # Multi-env training:
        # don't hard-code 4 here, let --num_envs decide
        # self.scene.num_envs = 16   # better not hard-coded

        # No ROS 2: it slows things down and interferes
        self.ros2 = None

        # Add the maze if needed, but not the play ROS camera
        self.scene.maze = AssetBaseCfg(
            prim_path="/World/Maze",
            spawn=sim_utils.UsdFileCfg(
                usd_path=WAREHOUSE_USD_PATH
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )