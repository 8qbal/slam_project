# Paper reproduction: Tables 7–8 (system load), first attempt, 2026-10-04

## How it was run

- Command: `uv run --no-sync python scripts/high_level/load_high_level.py --low_level_task Spot-Vslam-high-level-Play-v0 --steps 2000 --deterministic_policy`
- Policies were not changed: `high_level_policy.zip` and the low-level checkpoint
  `logs/rl_games/spot_rough_ppo/2026-09-30_15-57-30/nn/spot_rough_ppo.pth`.
- The VSLAM configurations start ORB-SLAM3 in parallel:
  `ros2 launch source/spot_vslam/ros2/spot_orbslam3/launch/orbslam3_rgbd.launch.py` with
  `ORB_SLAM3_ROOT=~/orbslam3/ORB_SLAM3`.
- The TSDF configuration adds `--enable_dense_mapping` (`pose_source="orb"`).
- Machine: RTX 4060 Laptop (8 GB), 16 GB RAM. The paper used an RTX 4080 SUPER, 64 GB RAM and a Ryzen 5 9600X,
  so absolute numbers aren't comparable. Only the trend between configurations is.
- 1 trial per configuration (the paper ran 5). Scene: flat terrain from the terrain generator, with no
  Room/Warehouse/Corridor map. Spawn is the corridor option (0, 0).

## Results

| Config | Avg FPS | Min FPS | RAM start→end (GB) | Avg CPU % | Peak CPU % | Avg read MB/s | Avg write MB/s | Peak write MB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL only | 6.63 | 5.42 | 9.24 → 9.21 | 15.05 | 16.50 | 0.01 | 0.07 | 5.99 |
| RL + VSLAM | 6.60 | 5.01 | 9.56 → 9.75 | 15.84 | 22.50 | 0.41 | 0.06 | 6.03 |
| RL + VSLAM + TSDF | 6.60 | 5.10 | 9.58 → 9.72 | 15.65 | 21.70 | 0.16 | 0.06 | 6.00 |
| *Paper: RL only* | *10.48* | *8.73* | *18.79 → 18.67* | *16.18* | *25.64* | *0.10* | *0.07* | *3.98* |
| *Paper: RL + VSLAM* | *9.82* | *8.39* | *19.27 → 19.48* | *19.94* | *26.80* | *0.33* | *0.07* | *3.03* |
| *Paper: + TSDF* | *5.78* | *1.57* | *20.94 → 29.40* | *29.59* | *43.66* | *0.12* | *1.42* | *118.12* |

What matches the paper: adding VSLAM raises RAM by about 0.5 GB, raises peak CPU, adds read spikes, and costs a
little FPS.

## Why this isn't a valid reproduction yet

1. **ORB-SLAM3 tracking is valid in only about 13–26% of steps** (524/2000 in the VSLAM run, 272/2000 in the
   TSDF run). The flat, uniformly textured floor gives ORB very few features. This matches the paper's Map A
   failure mode, but here the cause is that the arena is missing.
2. **TSDF barely ran.** With `pose_source="orb"`, only frames with valid tracking are integrated (272). The
   export happens every 1000 integrated frames and `close()` doesn't export, so no mesh was saved and the
   memory/disk load stayed near zero.
3. **The ORB-vs-GT error is wrong** (mean 6.6 m / 9.0 m). `OrbGtComparator` doesn't rotate the ORB frame by
   the initial yaw difference (reset randomizes yaw ±π), and it doesn't reset its origin when the episode
   resets every 400 HL steps.
4. Each run has one early episode termination at about step 8–9, right after the first reset. Every
   later episode ends by truncation at 400 HL steps.

## Changes made to the repo for this run (no policy changes)

- `tasks/spot_vslam/use_depth_training.py`: `train_camera.data_types` now includes `"rgb"`. Without it every
  camera's RGB output is black and ORB-SLAM3 can't track. The active spawn was switched from room (-9, -10)
  to corridor (0, 0).
- `scripts/high_level/load_high_level.py`: `--deterministic` renamed to `--deterministic_policy` (it clashed
  with AppLauncher's own `--deterministic` in Isaac Lab 3.0); `args_cli.enable_cameras = True`; `makedirs` for
  the output folder.
- `.venv`: `pandas==3.0.6` installed via `uv pip`. It isn't declared in `pyproject.toml`; `open3d` and
  `psutil` aren't declared either.

## Update: evaluator fixed (step 1), RL + VSLAM rerun as `rl_vslam_v2/`

`OrbGtComparator` now rotates both trajectories into the body frame at the start of tracking and resets its
origin when an episode resets. Episodes now last up to 3000 steps (the paper's evaluation timeout).

Result (2000 steps, 1 trial): mean position error 1.16 m, max 6.67 m; mean |yaw error| 31.5°. Tracking was valid in
218 of 2000 steps.

- **Tracked window (steps ~270–445):** error stays around 0.2 m and the ORB and GT trajectories overlap. The
  alignment is correct.
- **After tracking is lost (~445):** ORB builds a new map. When tracking comes back at around step 727, the ORB
  pose jumps and the error rises to 6.7 m. This looks like the "pose reset/jump" in paper Fig. 5–6 (Map A).
  Part of it, though, is a node bug: `orbslam3_rgbd_node` doesn't reset `have_origin_`/`origin_` on
  `slam_->Reset()` or when a new map is created, so poses from the new map are expressed relative to the old
  map's origin. In addition, `Ros2Manager.reset()` has `publish_orb_reset` commented out.

## Update: TSDF fixed (step 2), `rl_vslam_tsdf_gt/`

- Following the paper (Sec. 3: TSDF uses simulator GT poses), `load_high_level.py` gained
  `--dense_pose_source {orb,gt}` (default `orb`, as before). `DenseMapManager.close()` now exports whatever
  hasn't been exported yet.
- **Root cause of the empty TSDF:** Open3D **0.20.0** in `.venv` is broken. `ScalableTSDFVolume.integrate()`
  doesn't fill any voxel, even for a synthetic wall at 1 m. Open3D 0.19.0 (the version in the Isaac Lab
  venv) works. `.venv` is now pinned to `open3d==0.19.0` via `uv pip`.
- Result (2000 steps, GT pose, ORB-SLAM3 running): 2000 frames integrated; point cloud has 1.06 M points,
  mesh is 78 MB; see `tsdf_topview.png`.

| Config | Avg FPS | Min FPS | RAM start→end (GB) | Avg CPU % | Peak CPU % | Avg write MB/s | Peak write MB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RL only | 6.63 | 5.42 | 9.24 → 9.21 | 15.05 | 16.5 | 0.07 | 5.99 |
| RL + VSLAM (v2) | 6.61 | 5.32 | 9.52 → 9.72 | 15.89 | 22.5 | 0.06 | 5.95 |
| RL + VSLAM + TSDF (GT) | 6.15 | 2.99 | 9.53 → 10.72 | 16.69 | 23.2 | 4.24 | 54.84 |

The trend matches Tables 7–8: TSDF lowers average and especially minimum FPS, RAM grows steadily (+1.2 GB in
2000 steps), and disk writes jump from about 0.07 to 4.2 MB/s, peaking at 55 MB/s. The magnitudes are smaller
than in the paper (run length, scene and hardware all differ).

## Step 3: Tables 7–8 in the warehouse, 5 trials × 3 configs (`warehouse/`)

- New task `Spot-Vslam-high-level-Warehouse-Play-v0` (`SpotHighLevelTrainEnvCfg_PlayWarehouse`): the high-level
  play cfg (same observations) plus `Simple_Warehouse` and a single flat 40×40 m tile, so the env origin is
  (0, 0) and the robot starts inside the warehouse. The old task ids are unchanged.
- `load_high_level.py` gained `--seed` (seeds 1–5 = trials 1–5).
- Runner: `bash output/paper_repro/run_table7_8.sh`. Per-trial values are in `warehouse/all_trials.csv`.

Mean ± std over 5 trials (2000 HL steps each):

| Config | Avg FPS | Min FPS | RAM start→end (GB) | Avg CPU % | Peak CPU % | Avg read MB/s | Avg write MB/s | Peak write MB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL only | 6.03 ± 0.02 | 4.95 ± 0.09 | 8.68 → 8.70 | 15.27 ± 0.16 | 18.74 ± 1.30 | 0.47 ± 0.18 | 0.38 ± 0.20 | 25.0 ± 13.7* |
| RL + VSLAM | 5.97 ± 0.02 | 4.78 ± 0.07 | 9.08 → 9.32 | 16.73 ± 0.28 | 22.80 ± 1.41 | 0.39 ± 0.13 | 0.03 ± 0.01 | 1.27 ± 0.58 |
| RL + VSLAM + TSDF | 5.58 ± 0.03 | 2.67 ± 0.31 | 9.16 → 10.94 | 17.77 ± 0.48 | 23.74 ± 2.23 | 0.24 ± 0.13 | 1.57 ± 0.65 | 60.8 ± 16.3 |
| *Paper: RL only* | *10.48 ± 0.05* | *8.73 ± 0.06* | *18.79 → 18.67* | *16.18 ± 0.21* | *25.64 ± 1.27* | *0.10 ± 0.14* | *0.07 ± 0.05* | *3.98 ± 6.60* |
| *Paper: RL + VSLAM* | *9.82 ± 0.09* | *8.39 ± 0.06* | *19.27 → 19.48* | *19.94 ± 0.33* | *26.80 ± 0.62* | *0.33 ± 0.45* | *0.07 ± 0.04* | *3.03 ± 4.56* |
| *Paper: + TSDF* | *5.78 ± 0.21* | *1.57 ± 0.19* | *20.94 → 29.40* | *29.59 ± 1.32* | *43.66 ± 4.68* | *0.12 ± 0.12* | *1.42 ± 0.20* | *118.12 ± 7.68* |

\* The RL-only peak writes (about 30 MB/s in trials 2–5) aren't caused by the RL run itself. Its average write
rate is low. They are most likely the OS still flushing the large TSDF meshes from the previous trial.
psutil measures system-wide I/O.

What matches the paper:
- Adding VSLAM gives a small FPS drop, CPU +1.5 points, RAM +0.4 GB with slight growth.
- TSDF gives the largest drop: average FPS −7%, **minimum FPS 4.8 → 2.7**, RAM grows by about 1.8 GB per run,
  and disk writes rise from 0.03 to 1.6 MB/s (paper: 1.42), peaking at about 61 MB/s.

What differs:
- Absolute FPS is lower on this laptop (RTX 4060 Laptop vs RTX 4080S).
- TSDF RAM growth is smaller (+1.8 GB vs +8.5 GB). The runs are shorter (2000 steps) and the robot stays in a
  smaller area.
- CPU load is lower overall.

Other observations (useful for Table 9 later):
- ORB-SLAM3 tracking is much better in the warehouse: valid in about 760–860 of 2000 steps, versus about 220 on
  the flat floor. Mean ORB-vs-GT error is 2.1–2.5 m (paper Map B: about 1.75 m drift).
- About 13 early episode terminations per 2000 steps (only about 1 on the flat floor). Which termination term
  fires (body_contact / leg_contact) isn't logged yet. In a recorded warehouse episode (ends at step 211) the robot
  is in open floor, a few meters from the shelves, with its legs splayed in the last frames. So at least that
  termination looks like a stumble/fall rather than a collision. The policy only
  learned "move forward" with no obstacles or goal. Map B (paper) had 0.40 collisions and a 100% success rate.
