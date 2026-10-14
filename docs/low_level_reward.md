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

### Turn fix (2026-10-10)

`base_angular_velocity` σ 2.0 → 0.5 (a 0.2 rad/s error now costs 1.6 of 5). Retrained on `Spot-Paper-Obstacles-v0`
(`2026-10-10_07-00-46`, 8000 epochs, 5.5 h, reward 161). `gait_check.py`, 8 envs × 10 s:

| Command | Eval paper terrain | Eval obstacles | Eval Map A | Map A `--hide_maze` | Map A `--no_maze` |
| --- | --- | --- | --- | --- | --- |
| vx 0.25 / 0.5 / 0.75 / 1.0 | 0.25 / 0.52 / 0.79 / 1.04 | 0.24 / 0.51 / 0.78 / 1.02 | 0.23 / 0.51 / 0.79 / 1.02 | 0.21 / 0.43 / 0.77 / 1.04 | 0.23 / 0.52 / 0.79 / 1.01 |
| vx 0.5, wz +0.4 | wz 0.37 | 0.08 | 0.04 | 0.34 | 0.37 |
| vx 0.5, wz −0.4 | wz −0.31 | −0.31 | −0.03 | −0.30 | −0.31 |
| Resets | 0 | 1 | 1 | 8 | 0 |

Speed tracking stayed good. Turning is better on an open floor (+0.37 / −0.31, was +0.30 / −0.16). On Map A the
robot still does not turn, but with the map hidden (same colliders, the depth camera sees no walls) it turns
normally. So the depth image of the map, not contact, stops the turn: the policy reacts to what its depth input
shows (the walls and shelves of the warehouse are unlike the training pillars). Next: train with obstacles that
look more like the map, or check whether `visual_wall_penalty` / `open_space_seeker` push it to go straight.

### Wall terrain (2026-10-11/12): did not fix the Map A turn

The policy's depth image at the Map A start shows a wall ~2 m away filling the whole image (start near a corner,
random yaw). `Spot-Paper-Walls-v0` adds 25 % tiles with long walls (`mdp/terrains.py`, now 2-7 m long, 2.5-6 m high,
plus 6 m walls on up to two tile edges).

- Training from scratch on it failed (episode length ~30 steps at epoch 3000, 270 before; the env itself is fine: the
  old policy survives on it). Stopped and deleted.
- Fine-tune from `2026-10-10_07-00-46` instead: `2026-10-11_23-19-23` (epochs 7399-10000, first wall sizes), then
  `2026-10-12_01-26-58` (10000-12000, taller walls). Speed tracking stays good (0.30 / 0.51 / 0.77 / 1.02 on the paper
  terrain). Note: `spot_paper_ppo.pth` of a resumed run is saved right after the start; use the `last_*` files.
- Map A turning is still weak and erratic, also turning in place (vx 0, wz ±0.4: 0.07 / −0.02 visible, 0.26 / −0.37
  with the map hidden). With one env per command the numbers are noisy, but the visible map clearly blocks turns.

Likely reason: near walls any motion risks a contact termination, so the policy learned to move little there, and
the Map A start is in a corner. Open: make the low level follow commands near walls (e.g. no termination on light
wall contact, or let the high level handle avoidance) or move the Map A start away from the corner.

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

## Limp: uneven trot diagonals (2026-10-14)

All four paper-version policies (paper reproduce, WallsSoft, Rough, WallsSoftNoise) limp: at 0.5 m/s FL and HR are on
the ground 73-82 % of the cycle and carry 100-150 N, FR and HL only 19-28 % and 30-45 N. The hind left leg stamps down
for ~0.12 s every ~0.45 s (its knee action has a second bump inside the swing), which shows as the leg twitching. The
gait reward does not see it: it only asks one diagonal to be up while the other is down, so 75/25 earns it in full
(~10 of 12), and `air_time_variance` (−1) costs only ~0.03.

`Spot-Paper-WallsSoftSym-v0` = WallsSoft + `diagonal_duty` (`mdp.diagonal_duty_penalty`, weight −5): the difference in
duty factor between the two diagonals, 0 for an even trot, ~0.42 for the WallsSoft limp (−1.85 per second on the
`gait_check.py` commands). Fine-tuned from the WallsSoft checkpoint. Check: `gait_check.py` term `diagonal_duty` near 0,
and the four feet each down ~50 %.

## Still open

- Before: 81-92 % of policy outputs hit the [−1, 1] clip (both policies). The paper policy is now unclipped, like
  Isaac Lab's Spot. Check the new policy's output range in `gait_check.py` (`|mu|>1`).
- `rl_vslam_v0` sometimes falls at 1 m/s. Check the new policy at the same speeds.

## Retrain (the user runs this)

```bash
python scripts/rl_games/train.py --task Spot-Paper-v0 --num_envs 512 --cameras --max_iterations 8000  # ~5.5 h
python scripts/rl_games/train.py --task Spot-Paper-Rough-v0 --num_envs 512 --cameras
python scripts/rl_games/train.py --task Spot-Paper-Obstacles-v0 --num_envs 512 --cameras
python scripts/rl_games/train.py --task Spot-Paper-WallsSoftSym-v0 --num_envs 512 --cameras \
  --checkpoint policies/walls_soft/model.pth --max_iterations 16000  # fine-tune from WallsSoft ep 14000
python scripts/tools/gait_check.py --task Spot-Paper-v0 --checkpoint logs/rl_games/spot_paper_ppo/<run>/nn/spot_paper_ppo.pth
```
