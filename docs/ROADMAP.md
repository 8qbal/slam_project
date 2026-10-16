# Roadmap

What to build next, in rough order. Tick items off as they land. Each bigger item gets its own design note when we
start it. Background: our proposal (cuVSLAM, SLAM benchmark, Kalman filter, depth noise, tracking signals in the
high-level observation, goal observation, hopping gait) and the advisor's feedback on it.

**The baseline is done.** `rl_vslam_v0` is frozen (`baselines/rl_vslam_v0/`, tag `baseline/rl-vslam-v0`); keep it
runnable, but don't build on it. All new work trains with the paper policies and rewards in this repo (see
`docs/SYSTEM.md`):

| | Task | Policy | Reward |
| --- | --- | --- | --- |
| Low level | `Spot-Paper-v0` (`tasks/locomotion/config/spot/paper_env_cfg.py`) | 353-dim obs (80x60 depth), [512, 256, 128] ELU | Table 3 weights, Isaac Lab Spot functions + Eq. 5 wall (`docs/low_level_reward.md`) |
| High level | `Spot-Nav-Map{A,B}-v0` (`tasks/navigation/`) | 9-dim obs with goal (Table 5), [256, 128] ELU, Table 6 PPO | goal progress + success bonus, time and fall penalties (ours) |

Rules for every experiment: change one thing at a time. Keep the low-level policy, task, rewards and training
budget fixed unless that is what is being tested. Compare policies under the same evaluation (same maps, start/goal,
SLAM system, commanded speed).

## Paper plan

| # | Comparison | Fixed | Varied | Section |
| --- | --- | --- | --- | --- |
| 1 | Low level: old vs. new reward | terrain, obs | literal Eqs. 2-6 (results in the original paper, no retrain) vs. Isaac Lab Spot functions | 1 |
| 2 | Low level: training terrain | reward, PPO, budget | paper terrain / rough / obstacles | 1 |
| 3 | Low level: depth noise | reward, PPO, terrain | without / with depth noise | 1 |
| 4 | SLAM: ORB-SLAM3 vs. cuVSLAM (± IMU / Kalman filter) | recorded data, policy | SLAM system | 2, 3 |
| 5 | High level with different SLAM | low level, high level, maps | GT / ORB-SLAM3 / cuVSLAM / cuVSLAM + IMU | 4 |
| 6 | High level: tracking quality (A/B/C) | low level, task, reward, budget | pose source, quality inputs | 5 |

Proposal items → sections: 1 cuVSLAM → 2; 2 benchmark (evo) → 2; 3 Kalman filter / IMU → 3; 4 depth noise → 1;
5 tracking signals → 5; 6 goal observation → 0 (done); 7 hopping gait → 1.

## 0. Baseline, up to E0 (done)

- [x] Code, settings and trained policies saved: `baselines/rl_vslam_v0/`, tag `baseline/rl-vslam-v0`.
- [x] System diagram and each policy's inputs/outputs: `docs/SYSTEM.md` (keep it up to date).
- [x] Goal information in the high level (proposal item 6): the paper high level gets goal dx, dy and heading error,
      one fixed start/goal per map (`assets/maps.py`), reward for progress toward the goal. This is a task change
      vs. `rl_vslam_v0` (no goal), documented in `docs/SYSTEM.md`.
- [x] High level as a manager-based env (`Spot-Nav-*`): frozen low level in `LowLevelPolicyAction`, goal as a command
      term, pose as an observation term with `pose_source="gt" | "slam"`, rl_games trainer.
- [x] Paper policies trained: low level `Spot-Paper-v0`, high level with GT pose (condition A).
- [x] E0, the current system measured: latency, pose error, tracking failures, CPU/GPU, navigation (Table 9),
      GT vs. SLAM pose with TSDF on and off. TSDF only builds a map; it does not feed control.
- [x] E0 redone 2026-10-13 (`output/paper_tables_e0_new/`): WallsSoft low level, same SB3 high level (trained on an
      older low level), ORB-SLAM3 RGB-D, 100 m far clip, start facing the goal. Success A 0 → 40 %, B 0 → 60 %;
      collisions 1 → 67 (A) / 39 (B); load unchanged (~6 FPS, +VSLAM +2 % CPU). Pose error 0.2 → 2-3.5 m: the old
      robot barely moved (stuck at the start), the new successful runs drift 4-7 m over the full path. Redo with a
      high level retrained on the new low level and with stereo SLAM; GPU and SLAM delay are still not logged.

## 1. Low level: fix the gait, then compare (proposal items 4, 7; paper comparisons 1-3)

The paper low level hopped in place. A hopping body shakes the camera and would hurt every SLAM measurement, so this
comes first. Details and result tables: `docs/low_level_reward.md`.

- [x] Diagnose (`scripts/tools/gait_check.py`): the policy trots in place and never moves. The literal Eq. 3 gait
      indicator (weight 12) pays more than speed tracking.
- [x] Reward switched to Isaac Lab's Spot functions (Eqs. 2-4, 6), same Table 3 weights; PPO settings match Isaac
      Lab's Spot (no action clipping, separate critic, no normalization, 20000 iterations).
**The five low-level policies used for comparisons 1-3 (decided 2026-10-12):**

| Policy | Role | Task / run | Status |
| --- | --- | --- | --- |
| Old paper | the paper with Eqs. 2-6 taken literally (trots in place) | `spot_paper_ppo/2026-10-05_16-01-15` (no longer loads; old network) | done |
| `paper-isaaclab-reward-reproduce` | the paper reproduction: Table 3 weights on Isaac Lab's Spot rewards, Isaac Lab PPO and defaults, paper terrain | `Spot-Paper-v0`, `spot_paper_ppo/2026-10-08_11-08-43` | done |
| WallsSoft | ours, the main policy: speed and turn fixes, wall terrain, wall contact = penalty (no termination) | `Spot-Paper-WallsSoft-v0`, fine-tuned from `2026-10-12_01-26-58` ep 12000 → `2026-10-12_12-50-26` ep 14000 | done (turns right weaker: 20-80 % of the command) |
| Rough | ours, other terrain (Isaac Lab's Spot terrain), same reward as WallsSoft but contact ends the episode | `Spot-Paper-Rough-v0`, `2026-10-12_18-25-42` ep 12000 | done |
| WallsSoft + noise | WallsSoft with noise on every sensor (comparison 3): Isaac Lab's Spot joint / IMU noise, depth range noise, missing pixels, edge dropout | `Spot-Paper-WallsSoftNoise-v0`, from the same ep 12000 → `2026-10-13_11-29-20` ep 14000 | done (tracks well with noise, drifts without; pitch rate ~2x WallsSoft; left turns now weaker) |
| WallsSoft + even trot | WallsSoft + `diagonal_duty` penalty: every policy limps (FL+HR down 75 %, FR+HL 25 %; hind left leg stamps), see `docs/low_level_reward.md` | `Spot-Paper-WallsSoftSym-v0`, fine-tuned from WallsSoft ep 14000 → `2026-10-14_22-23-23` ep 16000 | trained: limp 74/27 → 65/36 %, but turning weaker (wz 0.4 → 0.09); from scratch → `2026-10-15_17-39-13` ep 14000: 8 Hz four-foot shuffle, no trot; neither adopted |
| WallsSoft + trot shape | WallsSoft + `foot_duty` (each foot ~50 % down, −5) + `short_swing` (swings ≥ 0.2 s, −30 per touchdown) | `Spot-Paper-WallsSoftTrot-v0`, fine-tuned from WallsSoft ep 14000 | training |

`rl_vslam_v0` stays in the repo but is not used in our paper: the baseline is the numbers reported in the original
paper (a rerun here would differ anyway), and the high level is retrained on the chosen low level.

Main comparison: `paper-isaaclab-reward-reproduce` vs. WallsSoft (reward and terrain changes as one package). The paper terrain is not retrained
with our reward: flat-floor training already failed near Map A's walls.

- [x] Train `Spot-Paper-v0` and check it with `gait_check.py`. If it still hops or tracks poorly, change one reward
      term at a time (Unitree / legged_gym rewards as reference).
- [ ] Comparison 1, old vs. new reward: old results come from the original paper; new from `gait_check.py` at the
      same commanded speeds (gait pattern, speed error, falls).
- [ ] Comparison 2, training terrain: train `Spot-Paper-Rough-v0` (Isaac Lab's Spot terrain) and
      `Spot-Paper-Obstacles-v0` (paper terrain + 1-2 m pillars the depth camera sees). Evaluate every policy on every
      terrain (`gait_check.py --task <terrain task>`).
- [ ] Comparison 3, depth noise: add realistic errors to the policy's depth image during training (missing pixels,
      edge noise, range-dependent noise), as a variant of the best terrain. Evaluate with and without noise.
- [ ] Pick the low level for everything below. Keep the 353-dim observation so the high-level code still fits, and
      retrain the high level (condition A) on it.

## 2. cuVSLAM and the SLAM benchmark (proposal items 1, 2; paper comparison 4)

Goal: shorter perception delay and less CPU than ORB-SLAM3.

- [ ] Record one sensor dataset (rosbag: RGB, depth, stereo pair if used, IMU, GT pose) from a fixed policy run.
- [x] cuVSLAM wrapper written (`slam/cuvslam_manager.py`, design in `docs/cuvslam.md`): in-process, RGB-D like
      ORB-SLAM3, loop closure on, one tracker per env, pose + tracking state + feature count + track time.
      Tasks `Spot-Nav-Map{A,B}-cuVSLAM-Play-v0`.
- [x] First run and frame check (`scripts/slam/cuvslam_frame_check.py`): pose follows GT in sign and scale.
- [x] Odometry jumps (RGB-D, `docs/cuvslam.md` frame check 2): cause not found in RGB-D; switched to stereo (frame
      check 3: 0.27 m mean error, no jumps). ORB-SLAM3 gets a stereo node too, for the same input.
- [x] First live comparison, stereo, one drive start → goal on Map A (`cuvslam_frame_check.py --to_goal`, evo;
      `docs/cuvslam.md`): cuVSLAM APE 0.57 m, ORB-SLAM3 3.4-5.0 m. Still to do: more runs, Map B, IMU (below).
- [ ] Offline benchmark on the recorded data: ORB-SLAM3 vs. cuVSLAM with matched camera inputs and IMU use. evo
      ATE/RPE, tracking failures, recovery.
- [ ] Online benchmark with the same fixed policy: total latency (image capture → pose used by the policy), SLAM
      time per frame, CPU/GPU. Check whether sharing the GPU with Isaac Lab becomes the new bottleneck.

## 3. IMU and Kalman filter (proposal item 3; paper comparison 4)

- [ ] IMU in simulation: Isaac Lab's `Imu` sensor on the body (or the camera mount), or a ZED camera through the
      Stereolabs Isaac Lab integration (stereo + IMU, as on the real camera). Pick one in the design note.
- [ ] Later, optional (sim-to-real): low-level inputs from sensors instead of sim state (`docs/SYSTEM.md`): gyro and
      gravity from the `Imu` sensor with bias / drift, base lin vel estimated or dropped, encoder delay. Changes the
      observation, so the low level must be retrained.
- [ ] First test cuVSLAM's own visual-inertial mode on the recorded data (evo, tracking failures, recovery).
- [ ] Only then decide whether an extra Kalman filter is needed: which problem it solves (e.g. pose gaps when tracking
      is lost, delay between SLAM poses) and which measurements it fuses (SLAM pose, IMU, leg odometry). If yes,
      compare with and without it with evo and the same runs.

## 4. High level with different SLAM (paper comparison 5)

Same low level, same high level (condition A), same maps (A, B) and start/goal.

- [ ] SLAM must reset cleanly between episodes. Today `orbslam3_rgbd_node` keeps `have_origin_`/`origin_` after
      `slam_->Reset()` or a new map, and `Ros2Manager.reset()` has `publish_orb_reset` commented out, so poses jump
      after a reset. Fix it (or use cuVSLAM) and test in a small setup first.
- [ ] Pose source in `Spot-Nav-*-Play`: GT / ORB-SLAM3 / cuVSLAM / cuVSLAM + IMU (+ Kalman filter if section 3 adds
      one). During GT runs keep SLAM running in the background so the compute load matches.
- [ ] Measure per pose source: success %, collisions, steps, time (Table 9), pose error (evo), tracking failures,
      latency, CPU/GPU/RAM (Tables 7-8).

## 5. Tracking quality in the high level (proposal item 5; paper comparison 6, main experiment)

Question: can the robot learn to change how it moves when localization becomes unreliable?

- [ ] Define each quality signal carefully: tracking state (cuVSLAM `track()` valid/lost), number of tracked
      features, and what each one does and doesn't tell us (a returned pose is not always healthy tracking; feature
      count alone is not a confidence measure).
- [ ] Tracking-quality observation terms in `tasks/navigation/mdp/observations.py`.
- [ ] Many envs with SLAM in the loop need one tracker per env (easier with in-process cuVSLAM).
- [ ] Train three high-level policies with the same low level, task, reward and budget:
  - [ ] A: GT pose, no quality inputs (section 1's retrained high level).
  - [ ] B: SLAM pose, no quality inputs (`pose_source="slam"`, with cuVSLAM).
  - [ ] C: SLAM pose + quality inputs.
- [ ] Evaluate A, B, C with the same SLAM system and scenarios. A vs. B = training with SLAM; B vs. C = quality inputs.

## Engineering backlog

- [x] One shared base config (2026-10-07): `tasks/locomotion/locomotion_env_cfg.py` + `config/spot/{paper,baseline_v0}_env_cfg.py`.
      Every kept task's resolved config was checked identical before/after, except intended changes. Dropped task
      ids are still in tag `baseline/rl-vslam-v0`.
- [x] `Spot-test180-v0` uses a generated flat floor (was a missing `default_ground_plane.usda`).
- [x] The manager-based `Spot-Nav-*` has no zero-command bug (the SB3 wrapper left Isaac Lab's velocity command
      generator running, ~5 % zero commands).
- [ ] Spawn outside the warehouse in terrain-generator envs (5×5 tiles × 8 m vs. warehouse x −12..12, y −18..20.8).
      Shrink the spawn area; don't scale the warehouse. Map tasks are not affected.
- [ ] "RL only" runs still attach `Ros2Manager` and publish images (`rl_vslam_v0`, known limitation 4).
- [ ] Declare open3d 0.19.0, pandas and psutil in `pyproject.toml` (open3d 0.20 breaks TSDF).
- [ ] Arena as a parameter (`warehouse` / `none` / USD path) instead of hard-coded `/World/Maze`.
- [ ] Lighting: dome light 3000 + warehouse lights over-expose the camera; may hurt SLAM features.
- [x] Map A shows sky in the SLAM camera: not a missing roof (`warehouse.usd` has a roof and 9 m walls facing in);
  the SLAM camera's far clip was 10 m, so the far walls and the roof were cut. SLAM cameras now clip at 100 m
  (`SLAM_PINHOLE_CFG`; rl_vslam_v0 keeps 10 m). Earlier frame checks and replays used the 10 m clip.
- [ ] Physics presets (PhysX / Newton) via Isaac Lab 3.0 `PresetCfg`. Verify RTX camera, ray casters, contact
      sensors and ROS 2 under Newton before claiming support.
- [ ] ROS 2 topic prefixes and frame names are hard-coded in 13 places; keep them only in `Ros2ManagerCfg`.
- [ ] Two ORB-SLAM3 pose subscribers: `Ros2Manager` (`env.orb_slam_res`) and `OrbSlamSubscriberManager` (slam_test
      scripts) both subscribe to `orbslam/robot_pose`. Merge into one; the rl_vslam_v0 tables were measured through
      `Ros2Manager`, so re-check them after.
- [ ] `scripts/slam/test_nav2_loop.py` defaults to the unregistered task `Spot-test-hybrid`.
- [ ] Low-level scenes keep a height scanner nothing reads; remove it with the next low-level retrain (it doesn't
      change the observation).
