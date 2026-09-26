# Plan: making the spot_vslam configs more universal

Status as of 2026-09-26. The project runs on Isaac Lab 3.0 (15/15 task configs load, and `Spot-test180-v0` and
`Spot-Vslam-Deploy-Play-v0` run headless). Machine-specific paths are gone: the low-level checkpoint comes from
`--checkpoint` or `$SPOT_VSLAM_LOW_LEVEL_CKPT`, and assets come from `$ISAACSIM_ASSET_ROOT`.

Remaining steps, in order:

## 1. One shared base config (remove duplication)

The env configs in `source/spot_vslam/spot_vslam/tasks/spot_vslam/` add up to about 4,900 lines, and many pieces are copy-pasted:

| Duplicated | Files |
| --- | --- |
| `SpotRewardsCfg`, `SpotObservationsCfg`, `COBBLESTONE_ROAD_CFG` | 7 |
| `SpotTerminationsCfg` | 6 |
| `get_orb_slam_pose` / `get_orb_slam_status` | 4 |
| `SpotEventCfg`, `SpotActionsCfg` | 3 |

- Move the shared classes into `tasks/spot_vslam/base_cfg.py`, and the shared MDP functions into `mdp/`.
- Each task only overrides what differs: cameras, arena, ROS 2 on/off, decimation.
- Keep the observation layout of each task unchanged, so a policy trained under one task id still loads.
- Check it: every task id's obs/action shapes must be identical before and after the refactor. Use `load_cfg_from_registry` and a short headless run.

## 2. Arena as a parameter (and fix spawns outside the warehouse)

- Add one arena setting (e.g. `warehouse` / `none` / custom USD path). Tasks should not hard-code `/World/Maze` + `WAREHOUSE_USD_PATH`.
- **Known bug:** in terrain-generator envs (5×5 tiles × 8 m), robots spawn at tile origins up to ±16 m from the center. The warehouse only spans x −12..12 and y −18..20.8, so some envs start outside it. `Spot-Vslam-Deploy-Play-v0` showed open sky. Fix this by making the spawn area fit the arena (smaller terrain grid, plane terrain, or restricted env origins). Scaling the warehouse is not the fix.
- The `camera_frustum` raycasters must follow the arena prim (`MultiMeshRayCasterCfg` over the arena root).
- Remove or replace references to the lost custom USDs: `corrider_map.usd` in `scripts/high_level/test_environment.py`, and the `SPOT_VSLAM_USD_DIR` outputs in `scripts/tools/add_light_to_spot_usd.py`.
- Lighting: the dome light (3000) plus the warehouse lights over-expose the camera images, which may hurt ORB-SLAM3 features.

## 3. Physics backend presets (PhysX + Newton), optional

- `velocity_spot_env_cfg.py` uses `isaaclab_physx.physics.PhysxCfg` directly. Switch to Isaac Lab 3.0's `PresetCfg` / `preset(...)`
  pattern (see `isaaclab_tasks/core/velocity/velocity_env_cfg.py`), so that `physics=isaacsim_physx|newton` can be chosen from the CLI.
- Deprecated schema cfgs (`RigidBodyPropertiesCfg`, `ArticulationRootPropertiesCfg`, `RigidBodyMaterialCfg`) still work until 5.0.
  Move to the solver-common or backend-specific fragments while doing this.
- Before claiming Newton support, verify the RTX camera, the ray casters, the contact sensors and the ROS 2 managers under Newton.

## Also open

- Retrain the policies trained under 2.x: the camera mount and arena changed, and the raw quaternion observation is now XYZW.
- `scripts/high_level/high_level_train.py` is a fragment with no imports.
- `scripts/slam/test_nav2_loop.py` defaults to the task `Spot-test-hybrid`, which is not registered.
- ROS 2 topic prefixes and frame names are hard-coded in 13 places. Consider moving them into `Ros2ManagerCfg` defaults only.
