# Baseline `rl_vslam_v0`: RL + VSLAM paper reproduction, 2026-10-05

This is a frozen snapshot of the code, settings and trained policies that produced the first full reproduction of
paper Tables 7–8 (system load). The paper is `references/main.pdf`. It was saved before any of the fixes listed
under "Known limitations", so later changes can be compared against it.

- **Code:** git tag `baseline/rl-vslam-v0`. No file under `source/` or `scripts/` changed between the trials
  (2026-10-04) and this tag.
- **Task:** `Spot-Vslam-high-level-Warehouse-Play-v0` (`SpotHighLevelTrainEnvCfg_PlayWarehouse`).
- **Map:** the arena is `Simple_Warehouse/warehouse.usd`, which is the paper's **Map A ("Room")**, not Map B. The
  original code uses `warehouse_multiple_shelves.usd` for Map B. "Warehouse" in these results means Map A.

## Contents

| Path | What |
| --- | --- |
| `policies/low_level/spot_rough_ppo.pth` | Low-level locomotion policy (rl_games), copied from `logs/rl_games/spot_rough_ppo/2026-09-30_15-57-30/nn/` |
| `policies/low_level/{agent,env}.yaml` | rl_games agent and env params dumped by that training run |
| `policies/high_level/high_level_policy.zip` | High-level SB3 PPO policy, copied from `checkpoints_high_level/` (final model, 2026-10-01) |
| `settings/orbslam3_settings_robot0.yaml` | ORB-SLAM3 settings that `orbslam3_rgbd_node` generated (320×240, fx = fy = 145.45, 1000 ORB features) |
| `settings/pip_freeze.txt` | Package versions in `.venv` |
| `results/` | `SUMMARY.md`, `all_trials.csv` and the per-trial performance CSVs (3 configs × seeds 1–5). The meshes, plots and logs are left out; they are still in `output/paper_repro/` (git-ignored) |
| `run_table7_8.sh` | Re-runs all 15 trials with these policies. Output goes to `output/baseline_rl_vslam_v0_rerun/` |

SHA-256:

```
5cef0f0212b62f407179801430ec3507690b0cc85b526b943944611ce0a91ca3  policies/low_level/spot_rough_ppo.pth
1a2f6533506dbd52e15a7907d36b7ff64a7e0093444293905b618b072f91d194  policies/high_level/high_level_policy.zip
```

## Environment

- Isaac Lab 3.0.0rc1 (`~/IsaacLab` @ `255de9cc9`), Isaac Sim 6.1.0, torch 2.11.0, stable-baselines3 2.9.0, numpy 2.3.1
- open3d **0.19.0** (0.20.0 breaks TSDF), pandas 3.0.6, psutil 5.9.8. None of these three is declared in
  `pyproject.toml`, so `uv sync` can remove them or change their versions.
- ORB-SLAM3 `~/orbslam3/ORB_SLAM3` @ `4452a3c`, ROS 2 Jazzy
- RTX 4060 Laptop (8 GB), 16 GB RAM

## Results (5 trials × 2000 HL steps, mean ± std)

| Config | Avg FPS | Min FPS | RAM start→end (GB) | Avg CPU % | Peak CPU % | Avg write MB/s | Peak write MB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RL only | 6.03 ± 0.02 | 4.95 ± 0.09 | 8.68 → 8.70 | 15.27 ± 0.16 | 18.74 ± 1.30 | 0.38 ± 0.20 | 25.0 ± 13.7 |
| RL + VSLAM | 5.97 ± 0.02 | 4.78 ± 0.07 | 9.08 → 9.32 | 16.73 ± 0.28 | 22.80 ± 1.41 | 0.03 ± 0.01 | 1.27 ± 0.58 |
| RL + VSLAM + TSDF (GT pose) | 5.58 ± 0.03 | 2.67 ± 0.31 | 9.16 → 10.94 | 17.77 ± 0.48 | 23.74 ± 2.23 | 1.57 ± 0.65 | 60.8 ± 16.3 |

RL + VSLAM, per trial: ORB tracking was valid in 553–1194 of 2000 steps, mean ORB-vs-GT position error was
0.87–5.52 m, and there were 10–15 early terminations.

## Known limitations

These are problems in this baseline itself. Fix them in later versions, not here.

1. `orbslam3_rgbd_node` doesn't clear `have_origin_`/`origin_` on `slam_->Reset()` or when ORB-SLAM3 starts a
   new map, so poses jump after a reset. `Ros2Manager.reset()` also has its `publish_orb_reset` call commented
   out. Both make the ORB-vs-GT error numbers unreliable.
2. The high-level policy has no goal in its observation (11 dims, against the paper's 9) and no goal-reaching term
   in its reward. It uses an SB3 `MlpPolicy` with [64, 64] layers; the paper uses [256, 128]. Table 9 (success
   rate, collisions) can't be reproduced with this policy.
3. About 13 early terminations per 2000 steps in the warehouse. At least one of them is a stumble, not a
   collision.
4. "RL only" still attaches `Ros2Manager` and publishes images. Only ORB-SLAM3 is left out.
5. RL-only disk-write peaks are inflated. The OS is still flushing the previous trial's TSDF meshes, and psutil
   measures I/O for the whole system.
