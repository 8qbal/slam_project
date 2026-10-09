# Low-level training: paper equations vs. Isaac Lab Spot

2026-10-07. Why the paper low level (`Spot-Paper-v0`) "hops", and what changed.

## Problem

The paper checkpoint (`logs/rl_games/spot_paper_ppo/2026-10-05_16-01-15`, 6000 epochs) **does not move**. It trots
in place: diagonal feet alternate, slow steps, low foot lift, zero forward speed for every command. That looks like
hopping.

Cause: Eqs. 2-6 were coded literally. The literal gait term (Eq. 3) is a binary "feet in a trot pattern" indicator
with weight 12. A slow trot in place hits it ~70 % of the time, a fast walk much less. The velocity term (Eq. 2,
squared error, σ = 1) loses only ~3 points per step when standing still at 1 m/s, while gait pays ~7.5. So standing
in place wins.

All Table 3 weights are exactly Isaac Lab's Spot reward weights, so the paper's equations are best read as a short
summary of Isaac Lab's Spot reward functions. `rl_vslam_v0` uses those functions with the same weights, and it walks.

## Change

Eqs. 2, 3, 4 and 6 now use Isaac Lab's Spot functions (`isaaclab_tasks/contrib/velocity/config/spot/mdp/rewards.py`).
Eq. 5 (wall) has no Isaac Lab version and stays ours. Weights, observation (353 dims), actions and PPO settings are
unchanged. The resulting reward equals the `rl_vslam_v0` reward except for the wall term.

| Term (Table 3 weight) | Paper equation | Before (literal) | Now (Isaac Lab) |
| --- | --- | --- | --- |
| Base linear velocity (5.0) | Eq. 2: exp(−‖v − v_cmd‖² / σ) | `base_linear_velocity_exp`, σ = 1, squared error | `base_linear_velocity_reward`: exp(−‖v − v_cmd‖ / 1.0) × speed ramp (above 1 m/s). Not squared, so small errors cost more |
| Gait (12.0) | Eq. 3: ∏ 1[foot in correct phase] | `gait_phase_indicator`: 1 if contacts form a trot pattern, else 0 | `GaitReward`: smooth score of how well the air/contact *times* of diagonal pairs match (and of the other pairs differ). Rewards steady stepping timing, not a snapshot pattern |
| Foot clearance (0.5) | Eq. 4: Σ max(0, h − h_min) | `foot_clearance_hinge`, h_min = 0.1 m, unbounded | `foot_clearance_reward`: exp(−Σ (h − 0.1)² · tanh(2‖v_foot‖) / 0.05). Swing feet near 0.1 m, bounded |
| Visual wall (−0.1) | Eq. 5: −min(depth) | `wall_min_depth` | unchanged (`wall_min_depth`) |
| Foot slip (−0.5) | Eq. 6: −Σ ‖v_foot‖² | `foot_contact_velocity_sq`: squared 3D foot speed in contact | `foot_slip_penalty`: planar foot speed (not squared) in contact |
| Other 11 terms | not written out | Isaac Lab Spot functions | unchanged |

Code: `source/spot_vslam/spot_vslam/tasks/locomotion/config/spot/paper_env_cfg.py` (`PaperRewardsCfg`), the four
literal functions were removed from `tasks/locomotion/mdp/rewards.py` (they are in git history).

## PPO settings: same as Isaac Lab's Spot

The paper lists no low-level PPO settings (Table 4 is only the network). Isaac Lab trains Spot with RSL-RL
(`isaaclab_tasks/contrib/velocity/config/spot/agents/rsl_rl_ppo_cfg.py`); `rl_games_paper_ppo_cfg.yaml` now uses the
same settings in rl_games terms. The rl_vslam_v0 yaml is unchanged.

| Setting | Before | Isaac Lab Spot (RSL-RL) | Now (rl_games) |
| --- | --- | --- | --- |
| Action clipping | [−1, 1] (±0.2 rad per joint) | none | none (`config.clip_actions: False`, `env: {}`) |
| Actor / critic | shared trunk | separate | `separate: True` |
| Obs / value normalization | on / on | off / off | off / off |
| Reward scaling | ×2 | none | ×1 |
| Weight init | orthogonal | PyTorch default | default |
| Initial action std | 1.0 | 1.0 | 1.0 |
| Learning rate | 5e-4 adaptive | 1e-3 adaptive, KL 0.01 | 1e-3 adaptive, KL 0.01 |
| Rollout | 48 steps/env | 24 steps/env | `horizon_length: 24` |
| Epochs / minibatches | 4 / 96 (minibatch 256) | 5 / 4 | 5 / 4 (`minibatch_size: 3072` for 512 envs) |
| Value loss | 0.5 · 2.0 · MSE | 0.5 · MSE | 0.5 · 1.0 · MSE |
| Entropy coef | 0.002 | 0.0025 | 0.0025 |
| Iterations | 6000 | 20000 | 20000 |
| Mixed precision | on | off | off |
| γ, λ, clip, grad norm | 0.99, 0.95, 0.2, 1.0 | same | same |

Kept from the paper (not Isaac Lab): network [512, 256, 128] ELU (Table 4, also Isaac Lab's), reward weights
(Table 3: gait 12, Isaac Lab uses 10), command ranges vx 0-1 m/s, wz ±0.4 rad/s (the high level's action range;
Isaac Lab uses vx −2..3, vy ±1.5, wz ±2), the 353-dim observation and the wall term.

`minibatch_size` must divide `num_envs × 24`: 3072 fits 128, 256 and 512 envs (512 envs = 4 minibatches).

The old paper checkpoint (`2026-10-05_16-01-15`) no longer loads under `Spot-Paper-*`: the network now has a
separate critic and no input normalizer. It never walked, so nothing that works is lost. `RlGamesPolicy`
(`tasks/navigation/mdp/actions.py`, used by `Spot-Nav-*`) now skips the clamp when the agent cfg does not clip.

## Measurements

`scripts/tools/gait_check.py`, 8 envs, fixed commands, 10 s, no pushes, first 2 s skipped.

Tracked forward speed (m/s):

| Command vx | Paper, literal reward | `rl_vslam_v0` (Isaac Lab functions) | Paper, Isaac Lab reward + PPO |
| --- | --- | --- | --- |
| 0.00 | | | 0.54 |
| 0.25 | 0.00 | 0.22 | 0.59 |
| 0.50 | 0.00 | 0.46 | 0.62 |
| 0.75 | 0.00 | 0.62 | 0.65 |
| 1.00 | 0.00 / 0.00 | 0.81 / 0.57 (one env fell 4×) | 0.69 / 0.69 (1 reset in 8 envs) |
| 0.50, wz ±0.4 | wz 0.01 / 0.00 | wz 0.26 / −0.30 | wz 0.38 / −0.40 |

Gait (averaged over the moving commands):

| | Paper, literal reward | `rl_vslam_v0` | Paper, Isaac Lab reward + PPO |
| --- | --- | --- | --- |
| Trot-pattern share of time | 0.64-0.80 | ~0.03 | 1.00 |
| Step rate (touchdowns / foot / s) | 0.7-2.0 Hz | ~4 Hz | 2.2 Hz |
| Foot lift (swing dz) | 3-5 cm | 1-4 cm | 3-6 cm |
| Outputs clipped to [−1, 1] | 92 % | 81 % | 75 % above 1 (not clipped) |

Mean per-step reward (weighted, ÷ dt), largest terms:

| Term | Paper, literal reward | `rl_vslam_v0` | Paper, Isaac Lab reward + PPO |
| --- | --- | --- | --- |
| gait | 7.46 | 6.52 | 11.90 |
| base_angular_velocity | 4.55 | 4.70 | 4.95 |
| base_linear_velocity | 3.47 | 4.44 | 3.95 |
| action_smoothness | −0.98 | −1.84 | −1.44 |
| air_time | 0.70 | 1.55 | 2.44 |
| joint_pos | −0.80 | −0.70 | −0.75 |
| foot_clearance | 0.003 | 0.46 | 0.48 |

Training log of the literal run: speed error stayed ~0.57 m/s from epoch 1500 to 6000; gait was the largest reward.

Paper, Isaac Lab reward + PPO = `2026-10-08_11-08-43`, epoch 20000 (reward 413). It walks with a clean trot and
turns well, but its speed barely follows the command: 0.54-0.69 m/s for commands 0-1 m/s. It cannot stand still or
walk slowly, and is too slow at 1 m/s. Likely cause: gait (12) and air time (5) pay for any steady trot, while the
speed term with σ = 1 is flat (0.35 m/s error costs only 1.5 of 5). The commands span only 0-1 m/s (Isaac Lab:
−2..3), so one fixed speed in the middle is a good enough answer. The epoch-8500 checkpoint gives the same numbers
(0.57-0.75 m/s, no resets): the reward is flat after ~8000 epochs, so longer training does not help.

### Speed-tracking fix (2026-10-08, retrain needed)

Only in `PaperRewardsCfg` / `SpotPaperEnvCfg` (`rl_vslam_v0` unchanged); weights stay at Table 3:

| Change | Isaac Lab | Now | Why |
| --- | --- | --- | --- |
| `base_linear_velocity` σ | 1.0 | 0.25 | a 0.35 m/s error now costs 3.8 of 5 instead of 1.5 (the paper gives no σ) |
| `gait`, `air_time` `velocity_threshold` | 0.5 m/s | ∞ | they paid a standing env for walking > 0.5 m/s (12 + 5 > 5), so it never stood; now only when a speed is commanded |
| `rel_standing_envs` | 0.05 | 0.1 | more zero commands, so standing still gets learned |

Retrained on `Spot-Paper-Obstacles-v0` (`2026-10-09_19-27-34`, 8000 epochs, 5.5 h, reward 179; not comparable
with 413 above, the reward is stricter). `gait_check.py`, tracked vx (m/s) and resets in 8 envs × 10 s:

| Command vx | Before the fix (paper terrain, ep 20000) | Fix: eval paper terrain | Eval obstacles | Eval Map A |
| --- | --- | --- | --- | --- |
| 0 | 0.54 | 0.05 | 0.14 | 0.10 |
| 0.25 | 0.59 | 0.31 | 0.28 | 0.27 |
| 0.50 | 0.62 | 0.51 | 0.52 | 0.52 |
| 0.75 | 0.65 | 0.73 | 0.79 | 0.18 (hit a wall, 2 resets) |
| 1.00 | 0.69 / 0.69 | 0.95 / 1.03 | 0.94 / 0.98 | 1.02 / 0.95 |
| 0.50, wz ±0.4 | wz 0.38 / −0.40 | wz 0.30 / −0.16 | wz 0.31 / −0.18 | wz 0.03 / −0.02 |
| Resets | 1 | 0 | 0 | 3 (31 before the fix) |

Speed now follows the command and the robot can stand. On Map A it no longer falls: the old policy (trained on an
empty floor) fell because its depth input showed walls it never saw in training (`gait_check.py --no_maze`: 1 reset;
`--hide_maze`, colliders kept but invisible: 14; visible map: 31). Still weak: turning, mostly to the right
(−0.16 for −0.4), and almost none on Map A. Its kernel is as flat as the old speed kernel (σ = 2: a 0.2 rad/s error
costs 0.5 of 5). |mu| > 1: 45 % (was 75 %).

## Training-terrain study

Same reward, PPO and budget; only the training terrain changes. All three use the same agent yaml, so any of their
checkpoints runs on any of the terrains (`gait_check.py --task <task> --checkpoint <.pth>`).

| Task | Terrain | Curriculum |
| --- | --- | --- |
| `Spot-Paper-v0` | paper (Table 1): 5x5 tiles, 70 % flat, 30 % rough 1-3 cm | terrain levels |
| `Spot-Paper-Rough-v0` | Isaac Lab's Spot (`COBBLESTONE_ROAD_CFG`): 9x21 tiles, 50 % flat, 50 % rough 2-5 cm | none, all levels from the start (as Isaac Lab) |
| `Spot-Paper-Obstacles-v0` | paper terrain with 30 % of the tiles holding 8 pillars / wall pieces, 0.5-2 m wide, 1-2 m high, 2 m clear spawn area | terrain levels |

On the paper terrain the depth terms (wall, open space) barely change, because the camera only sees floor. The
obstacle terrain gives them something to react to.

Results (fill after training; one row per trained policy, one column group per evaluation terrain):

| Trained on | Eval: paper terrain (speed error, falls) | Eval: rough | Eval: obstacles |
| --- | --- | --- | --- |
| paper terrain | | | |
| rough | | | |
| obstacles | wz −0.16 to right, vx ok, 0 resets | | vx ok, 0 resets |

## Still open

- Before: 81-92 % of policy outputs hit the [−1, 1] clip (both policies). The paper policy is now unclipped, like
  Isaac Lab's Spot. Check the new policy's output range in `gait_check.py` (`|mu|>1`).
- `rl_vslam_v0` sometimes falls at 1 m/s. Check the new policy at the same speeds.

## Retrain (the user runs this)

```bash
python scripts/rl_games/train.py --task Spot-Paper-v0 --num_envs 512 --cameras --max_iterations 8000  # ~5.5 h
python scripts/rl_games/train.py --task Spot-Paper-Rough-v0 --num_envs 512 --cameras
python scripts/rl_games/train.py --task Spot-Paper-Obstacles-v0 --num_envs 512 --cameras
python scripts/tools/gait_check.py --task Spot-Paper-v0 --checkpoint logs/rl_games/spot_paper_ppo/<run>/nn/spot_paper_ppo.pth
```
