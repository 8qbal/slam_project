# Roadmap

What to fix and what to build, in rough order. Tick items off as they land. Each bigger item gets its own design
note when we start it. Background: the advisor's feedback and our proposal (cuVSLAM, SLAM benchmark, Kalman filter,
depth noise, tracking signals in the high-level observation, goal observation, hopping gait).

Rules for every experiment: keep the `rl_vslam_v0` baseline runnable (`baselines/rl_vslam_v0/`, tag
`baseline/rl-vslam-v0`), and change one thing at a time. Keep the low-level policy, task, rewards and training budget
fixed unless that is what is being tested.

## 1. Fixes before any experiment

- [ ] Paper baseline reaches the goal. The paper policies fail every Table 9 run so far (0/10 on Map A,
      `output/paper_tables/navigation/navigation_results.csv`). Find out why (high-level reward, training map, play
      ranges, start/goal) and get a working baseline.
- [ ] ORB-SLAM3 reset: `orbslam3_rgbd_node` keeps `have_origin_`/`origin_` after `slam_->Reset()` or a new map, and
      `Ros2Manager.reset()` has `publish_orb_reset` commented out. Poses jump after an episode reset.
- [ ] Spawn outside the warehouse in terrain-generator envs (5×5 tiles × 8 m vs. warehouse x −12..12,
      y −18..20.8). Shrink the spawn area; don't scale the warehouse.
- [ ] "RL only" runs still attach `Ros2Manager` and publish images (`rl_vslam_v0`, known limitation 4).
- [ ] Declare open3d 0.19.0, pandas and psutil in `pyproject.toml` (open3d 0.20 breaks TSDF).

## 2. Measure the current system (E0)

- [ ] System diagram and policy inputs/outputs: `docs/SYSTEM.md` (done, keep it up to date).
- [ ] Latency: SLAM processing time per frame, and image capture → pose used by the policy (stamp the image in
      `Ros2Manager`, carry the stamp through the SLAM pose message, compare when the high level reads it).
- [ ] Pose error with evo (ATE/RPE): export GT and SLAM trajectories as TUM files.
- [ ] Tracking failures: count of lost frames, lost episodes, time to recover.
- [ ] CPU **and GPU** usage (the performance log has no GPU yet).
- [ ] Navigation: success %, collisions, steps, time (Table 9 script).
- [ ] TSDF on vs. off with the same policy and test runs. TSDF only builds a map; it does not feed control.
- [ ] GT-pose runs keep SLAM running in the background, so the compute load matches the SLAM runs.

## 3. cuVSLAM (E1)

- [ ] Record one sensor dataset (rosbag: RGB, depth, stereo pair if used, IMU, GT pose) from a fixed policy run.
- [ ] cuVSLAM wrapper (PyCuVSLAM, already installed in `.venv`) with the same output as ORB-SLAM3: pose, tracking
      state, feature count. Decide in-process vs. ROS node in its design note.
- [ ] Offline benchmark on the recorded data: ORB-SLAM3 vs. cuVSLAM with matched camera inputs and IMU use. evo
      ATE/RPE, tracking failures, recovery.
- [ ] Online benchmark with the same fixed policy: total latency, CPU/GPU, collisions, success. Check whether
      sharing the GPU with Isaac Lab becomes the new bottleneck.

## 4. Tracking quality in the high level (E2, main experiment)

- [ ] Check that SLAM behaves correctly across episode resets in a small setup first (needs fix 1.2).
- [ ] Define each quality signal carefully: tracking state, number of tracked features, and what each one does and
      doesn't tell us (a returned pose is not always healthy tracking).
- [ ] Train three high-level policies with the same low level, task, reward and budget:
  - [ ] A: GT pose, no quality inputs (the current paper high level is trained this way: GT + noise).
  - [ ] B: SLAM pose, no quality inputs.
  - [ ] C: SLAM pose + quality inputs.
- [ ] Evaluate A, B, C with the same SLAM system and scenarios. A vs. B = training with SLAM; B vs. C = quality inputs.

## 5. High level as an Isaac Lab manager-based env

Isaac Lab 3.0 has the pattern in `isaaclab_tasks/contrib/navigation/`: the high level is a normal
`ManagerBasedRLEnv`, and the frozen low-level policy runs inside an action term (`PreTrainedPolicyAction`). This would
replace our SB3 wrappers (`high_level_vec_env.py`, `paper_vec_env.py`) and make A/B/C plain config changes.

- [ ] Export the rl_games low-level policy to TorchScript, or write an action term that loads the `.pth`.
- [ ] Feed our 353-dim low-level observation (with depth) through `low_level_observations`.
- [ ] Goal as a command term, SLAM pose and quality signals as observation terms.
- [ ] Choose the trainer (rl_games / rsl_rl) and write down how it differs from the paper's SB3 PPO (Table 6).
- [ ] Many envs with SLAM in the loop need one tracker per env (easier with in-process cuVSLAM).
- [ ] Keep the SB3 wrappers so `rl_vslam_v0` and the current paper policy still run.

## 6. Separate studies (E3)

- [ ] IMU: first test cuVSLAM's own IMU fusion. Only then decide whether an extra Kalman filter is needed, which
      problem it solves, and which measurements it fuses.
- [ ] Depth noise: realistic errors (missing pixels, edge noise, range-dependent noise) during low-level training.
      Compare with and without, under the same evaluation.
- [ ] Hopping gait: diagnose first (reward terms, foot contacts, body motion, speed-tracking error). Compare any new
      policy at the same commanded speed.

## Engineering backlog

- [ ] One shared base config. The task configs repeat `SpotRewardsCfg`, `SpotObservationsCfg`,
      `COBBLESTONE_ROAD_CFG` (7 files), `SpotTerminationsCfg` (6), `get_orb_slam_pose`/`get_orb_slam_status` (4),
      `SpotEventCfg`/`SpotActionsCfg` (3). Move them into `tasks/spot_vslam/base_cfg.py` and `mdp/`. Every task id
      must keep the same obs/action shapes (check with `load_cfg_from_registry` and a short headless run).
- [ ] Arena as a parameter (`warehouse` / `none` / USD path) instead of hard-coded `/World/Maze`; the
      `camera_frustum` ray casters must follow the arena prim.
- [ ] Lighting: dome light 3000 + warehouse lights over-expose the camera; may hurt SLAM features.
- [ ] Physics presets (PhysX / Newton) via Isaac Lab 3.0 `PresetCfg`. Verify RTX camera, ray casters, contact
      sensors and ROS 2 under Newton before claiming support.
- [ ] ROS 2 topic prefixes and frame names are hard-coded in 13 places; keep them only in `Ros2ManagerCfg`.
- [ ] `scripts/slam/test_nav2_loop.py` defaults to the unregistered task `Spot-test-hybrid`.
- [ ] `scripts/tools/add_light_to_spot_usd.py` writes to `SPOT_VSLAM_USD_DIR` outputs of lost USDs.
