# cuVSLAM in the Isaac Lab pipeline (design note)

Roadmap section 2. Goal: shorter perception delay and less CPU than ORB-SLAM3, with the same input so the benchmark
is fair. Status (2026-10-10): manager and Play tasks run, frames are right (`scripts/slam/cuvslam_frame_check.py`).
Loop closure is on. Since 2026-10-13 the Play tasks run cuVSLAM in **stereo** (frame check 3 below): RGB-D
odometry jumped metres on single frames (frame check 2).

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
| `odom_xyyaw` | (N, 3) | odometry pose alone |
| `slam_xyyaw` | (N, 3) | loop-closure pose alone (NaN until SLAM gives one) |
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

## Frame check 2 (2026-10-10, low level `2026-10-09_19-27-34`)

The new low level walks on Map A without falling, so the full 285-step drive (23 s, ~10 m) runs:

| | Mean xy error | Max | Final | Mean yaw error | Jumps > 0.5 m in one step |
| --- | --- | --- | --- | --- | --- |
| odometry | 3.2 m | 5.2 m | 4.3 m | 5.8° | 2 |
| loop closure | same as odometry on every step | | | | |

- The loop-closure pose equals the odometry pose on every step: no loop is closed on this short drive, so loop
  closure is not the cause of the error.
- All the xy error comes from two single-frame odometry jumps: 4.4 m at step 87 (t = 7.0 s) and 2.7 m at step 145
  (t = 11.6 s). Between them each step follows GT. Yaw stays within ~6°. Tracking status stays 1.
- An earlier odometry-only run (robot stuck near the start) had a 0.34 m mean error, so short runs look fine.
- GT moves ~0.5 m in y in the first second (robot settling after spawn) before cuVSLAM sees it: small fixed offset.

Next: dump the RGB / depth frames around a jump (depth holes or far wall only? RGB and depth from the same render?)
and try `max_depth_m`, grayscale input, or a lower camera rate.

## Frame check 3: stereo (2026-10-13, low level WallsSoft `2026-10-12_12-50-26` ep 14000)

RGB-D replays of the frame-check-2 drive (`--dump_frames`, cuVSLAM offline) ruled out: depth cut at 10 m (with depth
to 40 m 96 % of tracked features have depth, still 35 jumps), RGB / depth one frame apart, wrong intrinsics (floor
depth gives a steady camera height), exposure, floor reflections, grey input, motion model, denoising. Mono on the same
images tracked without jumps, so the problem is in cuVSLAM's RGB-D path; we did not find it and switched to stereo,
cuVSLAM's main mode and the ZED's.

Stereo: `right_camera` (`SLAM_RIGHT_CAMERA_CFG`) 0.12 m to the right of `tilted_camera`, same intrinsics and update
period, `OdometryMode.Multicamera` with `rectified_stereo_camera=True`. Three bugs on the way:

- The right camera first kept the 10 Hz period while the Play scene renders the left one at 100 Hz: every other pair
  mixed two instants (142 of 285 right frames stale). The manager now refuses different periods.
- Play tasks reused the training reset: up to 1.5 m/s push at spawn (the robot slid ~0.4 m before the first command)
  and a random heading, some runs facing out of the building (sky, empty floor). Play / evaluation now spawns at the
  map start, facing the goal (`PaperMap.start_yaw`), without push. E0 was measured with the old random start.
- With RTX anti-aliasing off the floor renders as per-pixel noise (DLSS hides it); keep the default DLSS.

| Drive (285 steps, ~8 m, seed 0) | Mean xy error | Max | Yaw error | Jumps > 0.5 m | ms / frame |
| --- | --- | --- | --- | --- | --- |
| RGB-D, random start | 19.3 m | 32.7 m | 100° | 26 | 3.4 |
| stereo, random start | 1.6-1.8 m | 2.9-3.5 m | 14-19° | 0-1 | 0.9 |
| stereo, start facing the goal | 0.27 m | 0.56 m | 3.5° | 0 | 1.0 |

The camera sees sky over Map A, which leaves fewer features. Cause: the SLAM camera's far clip was 10 m, so walls and
roof farther away were not rendered (the map has both). Fixed: SLAM cameras clip at 100 m (`SLAM_PINHOLE_CFG`).

## ORB-SLAM3 stereo (2026-10-13)

`orbslam3_stereo_node` (same source as `orbslam3_rgbd_node`, `src/orbslam3_node.cpp`, built with
`SPOT_ORBSLAM3_STEREO`), `launch/orbslam3_stereo.launch.py`, tasks `Spot-Nav-Map{A,B}-Stereo-Play-v0` (`Ros2Manager`
also publishes `camera/right/image_raw` with the left image's stamp). Settings: `Camera.type: PinHole` with Camera1 /
Camera2 and `Stereo.T_c1_c2` (identity rotation, 0.12 m along x); `Camera.type: Rectified` would be the natural
choice, but ORB-SLAM3 then segfaults printing its settings (null Camera2 calibration). Both nodes now drop the old
origin on reset (roadmap section 4).

Replay of the 285 frames of frame check 3 (start facing the goal) over ROS 2 to each node, 0.1 s per frame, two runs
each (ORB-SLAM3 is not deterministic); error on frames 0-229 after aligning GT to the first tracked frame. (A first
version of this table was wrong: killed `ros2 run` wrappers had left up to 12 old nodes publishing on the same topic.)

| Same frames | Tracked (0-229) | Mean xy error | Yaw error | After frame ~230 |
| --- | --- | --- | --- | --- |
| cuVSLAM stereo (in-process) | 100 % | 0.27 m | 3.5° | tracked |
| ORB-SLAM3 RGB-D, defaults | 93 % | 1.0 / 2.8 m | 2.2 / 19.6° | lost |
| ORB-SLAM3 stereo, defaults (1000 features, FAST 20/7, th_depth 40) | 76 % | 2.2 / 2.3 m | 11-13° | lost |
| ORB-SLAM3 stereo, new defaults (2000, FAST 12/5, th_depth 80) | 96-100 % | 2.3 / 4.0 m | 4-13° | lost |

Sweep (th_depth 40-120, 1000-3000 features, FAST 20/7-10/4): more features and a lower FAST threshold start the
stereo map at frame 0 (default: 16 frames, < 500 points) and lose tracking less often mid-run; accuracy stays at 1-8 m
and varies more between identical runs than between settings. Every ORB-SLAM3 run loses tracking from about frame
230 (right turn facing sky and empty floor) and starts a new map, whose origin the node cannot relate to the first
one. Map A's sky (the 10 m far clip, since fixed) and plain floor hurt ORB features; fixing the scene would help both systems. The
Stereo-Play task has not run in the sim yet (OOM with other apps open).
