# cuVSLAM in the Isaac Lab pipeline (design note)

Roadmap section 2. Goal: shorter perception delay and less CPU than ORB-SLAM3, with the same input so the benchmark
is fair. Status (2026-10-08): manager and Play tasks run; frame check passed on the first 30 steps
(`scripts/slam/cuvslam_frame_check.py`). Loop closure is on. Longer runs wait on a low-level fix: the current
low-level policy falls over on Map A within ~2 s, with GT pose too (not a cuVSLAM issue).

## Same input as ORB-SLAM3

| | ORB-SLAM3 now | cuVSLAM |
| --- | --- | --- |
| Camera | `tilted_camera` (`SLAM_CAMERA_CFG`): 320x240 RGB + `distance_to_image_plane`, 10 Hz | same camera, same frames |
| Mode | RGB-D (`orbslam3_rgbd_node`) | `Tracker.OdometryMode.RGBD` |
| Intrinsics | from `camera_info` | from the camera cfg: fx = fy = 10 / 22 · 320 = 145.45 px, cx = 160, cy = 120, no distortion |
| IMU | none | none (inertial mode comes in roadmap section 3) |
| Loop closure | on (ORB-SLAM3 SLAM) | on: `slam_config` set. Odometry-only runs are a second variant |
| Transport | ROS 2 topics, separate process | in-process: images go straight from the camera tensor to `Tracker.track()` |

## Where it lives

| File | What |
| --- | --- |
| `source/spot_vslam/spot_vslam/slam/cuvslam_manager.py` | `CuVslamManagerCfg`, `CuVslamManager`: one `cuvslam.Tracker` per env |
| `source/spot_vslam/spot_vslam/envs/rl_env.py` | new optional manager `cfg.cuvslam`, updated every step like `cfg.ros2` / `cfg.slam_subscriber` |
| `tasks/navigation/mdp/observations.py` | `robot_pose(pose_source="slam")` reads `env.slam_res` (cuVSLAM) and falls back to `env.orb_slam_res` (ORB-SLAM3) |
| `tasks/navigation/config/spot/navigation_env_cfg.py` | `Spot-Nav-Map{A,B}-cuVSLAM-Play-v0`: Play with cuVSLAM instead of ROS 2 / ORB-SLAM3 |
| `scripts/slam/benchmark_slam.py` | offline benchmark on recorded data, writes TUM trajectories for evo |
| `output/slam_benchmark/` | benchmark results (git-ignored) |

`rl_vslam_v0` and the ORB-SLAM3 path stay as they are.

## Manager

Per step, for each env whose SLAM camera produced a new frame:

1. Take RGB (uint8, 240x320x3) and depth (float metres → uint16 millimetres, `depth_scale_factor = 1000`; NaN / inf
   → 0 = no depth) from the camera tensor.
2. `pose_estimate, slam_pose = tracker.track(timestamp_ns, [rgb], depths=[depth])`, timestamp = sim time in ns.
3. Write `env.slam_res`, the same keys as `env.orb_slam_res` plus the quality signals:

| Key | Shape | Meaning |
| --- | --- | --- |
| `pose_xyyaw` | (N, 3) | x, y, yaw of the robot body in its start frame (same frame as ORB-SLAM3 and the GT pose observation) |
| `status` | (N, 1) | 1 = the tracker returned a pose this frame, 0 = lost / not started |
| `num_features` | (N, 1) | tracked 2D features in the last frame, `len(get_last_observations(0))` (needs `enable_observations_export`) |
| `track_time_ms` | (N, 1) | wall time of `track()` (SLAM processing time) |
| `stamp_s` | (N,) | sensor time of the image the pose belongs to, seconds since the env's last reset (for the end-to-end delay) |

On episode reset (`reset(env_ids)`) the tracker of those envs is rebuilt, so every episode starts at the origin. This
avoids the pose jump ORB-SLAM3 has today (roadmap section 4).

## Frames

- The rig frame is the robot body: `rig_from_camera` = the camera offset (`CAMERA_OFFSET_CFG`: 0.4 m forward, ROS
  optical convention). cuVSLAM's world frame is the rig at the first frame, i.e. the body at the start of the episode.
- cuVSLAM's camera uses OpenCV axes (x right, y down, z forward). The rig axes follow the extrinsic we give it.
- Check before trusting it: drive the robot straight and turn in place with the GT pose next to it; x must grow
  forward and yaw must turn the same way as GT.

## Benchmark (paper comparison 4)

1. **Record** one run with a fixed policy: RGB, depth, GT pose and timestamps (rosbag of the `Ros2Manager` topics +
   GT, so ORB-SLAM3 can replay it unchanged).
2. **Offline**: ORB-SLAM3 replays the bag; cuVSLAM reads the same frames. Both write TUM files
   (`timestamp x y z qx qy qz qw`) next to the GT TUM file. Then
   `evo_ape tum gt.tum est.tum -a` and `evo_rpe tum gt.tum est.tum -a` (`-a` aligns the start frames).
   Also count lost frames and time to recover.
3. **Online**: the same fixed policy with each SLAM in the loop: total delay (image stamp → pose read by the high
   level), SLAM time per frame, CPU / GPU / RAM, success and collisions.

Watch: cuVSLAM runs on the same GPU as Isaac Lab (8 GB laptop GPU). Measure GPU time and memory, not only CPU.

## Open questions

- Does `track()` take RGB directly, or does it want grayscale? Check with a 10-frame test.
- Is one tracker per env fine for 16+ envs (GPU memory)? Needed for training with SLAM (roadmap section 5).
- Decided: loop closure on, to match ORB-SLAM3.

## First run (after training)

```bash
SPOT_VSLAM_LOW_LEVEL_CKPT=<low-level .pth> uv run scripts/rl_games/play.py --task Spot-Nav-MapA-cuVSLAM-Play-v0 \
  --cameras --checkpoint <high-level .pth>
```

Check: no errors from `track()` (RGB / uint16 GPU tensors accepted), `env.slam_res["status"]` goes to 1, the pose
follows GT in sign and scale (frames above), `track_time_ms` per frame.

## Frame check (2026-10-08)

`SPOT_VSLAM_LOW_LEVEL_CKPT=<low-level .pth> uv run scripts/slam/cuvslam_frame_check.py` (fixed commands, no high
level; `--task Spot-Nav-MapA-GT-Play-v0` runs the same drive without SLAM). Before the robot fell:

| Step | GT x, y, yaw | cuVSLAM x, y, yaw |
| --- | --- | --- |
| 5 | 0.10, −0.32, 3° | 0.12, −0.30, 6° |
| 10 | 0.22, −0.59, −9° | 0.27, −0.49, −11° |
| 15 | 0.40, −0.81, −6° | 0.48, −1.21, −8° |
| 30 | yaw 44° | yaw 37° |

x, y and yaw have the same sign and scale as GT, so the rig and frames are right. `track()` takes the RGB and the
uint16 GPU depth tensors directly. Tracked 100 % of frames, 200-450 features, 1-7 ms per `track()`.
