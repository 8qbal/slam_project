# Roadmap

What to fix and what to build, in rough order. Tick items off as they land. Each bigger item gets its own design
note when we start it. Background: the advisor's feedback and our proposal (cuVSLAM, SLAM benchmark, Kalman filter,
depth noise, tracking signals in the high-level observation, goal observation, hopping gait).

Rules for every experiment: keep the `rl_vslam_v0` baseline runnable (`baselines/rl_vslam_v0/`, tag
`baseline/rl-vslam-v0`), and change one thing at a time. Keep the low-level policy, task, rewards and training budget
fixed unless that is what is being tested.

## 1. Fixes before any experiment

- [ ] Paper baseline reaches the goal. The paper policies fail every Table 9 run so far (0/10 on Map A,
      `output/paper_tables/navigation/navigation_results.csv`). Found so far (2026-10-07):
  - [ ] **The paper low-level policy doesn't walk.** `logs/rl_games/spot_paper_ppo/2026-10-05_16-01-15` with a fixed
        command vx = 1 m/s in `Spot-Paper-Play-v0` moves ~0.25 m in 5 s (both through rl_games' player and the new
        batched loader). The rl_vslam_v0 low level walks at ~0.9 m/s in the same test. Retrain / fix the paper reward
        before training any high level on it (ties in with the hopping-gait item in section 6).
  - [ ] **Frequent falls at the map start.** With a zero command on Map A, ~1 in 5 low-level episodes per env ends in
        body/leg contact within 100 low-level steps (old SB3 path and new nav env alike). Check the spawn height,
        the random start velocity/yaw (`reset_base`), and walls next to the start.
- [ ] ORB-SLAM3 reset: `orbslam3_rgbd_node` keeps `have_origin_`/`origin_` after `slam_->Reset()` or a new map, and
      `Ros2Manager.reset()` has `publish_orb_reset` commented out. Poses jump after an episode reset.
- [ ] Spawn outside the warehouse in terrain-generator envs (5×5 tiles × 8 m vs. warehouse x −12..12,
      y −18..20.8). Shrink the spawn area; don't scale the warehouse.
- [ ] "RL only" runs still attach `Ros2Manager` and publish images (`rl_vslam_v0`, known limitation 4).
- [ ] Declare open3d 0.19.0, pandas and psutil in `pyproject.toml` (open3d 0.20 breaks TSDF).
- [x] `Spot-test180-v0` failed on a missing `default_ground_plane.usda`: now a generated flat floor.
- [x] The low-level SB3 wrapper left Isaac Lab's velocity command generator running, so ~5 % of the time
      (`rel_standing_envs`) the low level got a zero command. The manager-based `Spot-Nav-*` doesn't have this.

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

Done (2026-10-07): `tasks/navigation/`, tasks `Spot-Nav-Map{A,B}-v0` / `-Play-v0` / `-GT-Play-v0`, following Isaac
Lab's `isaaclab_tasks/contrib/navigation/`. The frozen low level runs inside an action term.

- [x] Action term that loads the rl_games `.pth` directly (`LowLevelPolicyAction`, no TorchScript export).
- [x] 353-dim low-level observation (with depth) through `low_level_observations`.
- [x] Goal as a command term (start-frame goal), pose as an observation term with `pose_source="gt" | "slam"`.
- [x] Trainer: rl_games, Table 6 mapped in `agents/rl_games_ppo_cfg.yaml` (16 envs x 64 steps = SB3's 4 x 256).
- [x] SB3 wrappers kept in `legacy/high_level/` so `rl_vslam_v0` and the current paper high level still run.
- [ ] Tracking-quality observation terms (condition C).
- [ ] Many envs with SLAM in the loop need one tracker per env (easier with in-process cuVSLAM).

## 6. Separate studies (E3)

- [ ] IMU: first test cuVSLAM's own IMU fusion. Only then decide whether an extra Kalman filter is needed, which
      problem it solves, and which measurements it fuses.
- [ ] Depth noise: realistic errors (missing pixels, edge noise, range-dependent noise) during low-level training.
      Compare with and without, under the same evaluation.
- [ ] Hopping gait: diagnose first (reward terms, foot contacts, body motion, speed-tracking error). Compare any new
      policy at the same commanded speed.

## Engineering backlog

- [x] One shared base config (2026-10-07): `tasks/locomotion/locomotion_env_cfg.py` + `config/spot/{paper,baseline_v0}_env_cfg.py`.
      Every kept task's resolved config (terms, order, weights, cameras, timing) was checked identical before/after,
      except intended changes: unused cfg fields removed, paper Play/HL low-level envs now use the paper reward
      (not used with a frozen policy), `Spot-Vslam-Depth-Play-v0` depth camera also renders RGB (else the SLAM
      camera is black), v0 Play tasks use the v0 training yaml (same network). Dropped task ids: `Spot-v0`,
      `Spot-Vslam-v0`, `-Deploy-`, `-Depth-2-`, `-Depth-3-`, `Isaac-Velocity-Rough-Spot-v0` (still in tag
      `baseline/rl-vslam-v0`).
- [ ] Arena as a parameter (`warehouse` / `none` / USD path) instead of hard-coded `/World/Maze`.
- [ ] Lighting: dome light 3000 + warehouse lights over-expose the camera; may hurt SLAM features.
- [ ] Physics presets (PhysX / Newton) via Isaac Lab 3.0 `PresetCfg`. Verify RTX camera, ray casters, contact
      sensors and ROS 2 under Newton before claiming support.
- [ ] ROS 2 topic prefixes and frame names are hard-coded in 13 places; keep them only in `Ros2ManagerCfg`.
- [ ] `scripts/slam/test_nav2_loop.py` defaults to the unregistered task `Spot-test-hybrid`.
- [x] `add_light_to_spot_usd.py` (wrote next to the lost USDs) moved to `scripts/legacy/`.
- [ ] Low-level scenes keep a height scanner nothing reads (from Isaac Lab's velocity config); remove it with the
      next low-level retrain (it doesn't change the observation).
