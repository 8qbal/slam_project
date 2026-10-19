# Paper outline (draft)

Working draft of our paper, a follow-up to the reproduced paper (`references/main.pdf`, Li & Jeong, IJCAS 2026). The
reasons behind each choice are in `docs/DESIGN_DECISIONS.md`; the experiment plan and open work are in
`docs/ROADMAP.md`. Numbers marked **TBD** are not measured yet. Everything else comes from a run in this repo.

## Working title

*Localization-aware hierarchical reinforcement learning for quadruped navigation with visual SLAM*
(placeholder; the real title depends on which results hold up, see "Claims and status").

## One-paragraph pitch

The reproduced paper runs a frozen low-level locomotion policy under a high-level navigation policy and measures the
cost of putting ORB-SLAM3 and dense mapping in the loop. Its weak points are the ones it names itself: the policies
saw little environment variety, and the high level was trained with perfect pose but evaluated with SLAM pose. We
(1) rebuild the low level so it walks symmetrically and turns in place, (2) train the high level on random mazes
instead of one fixed map, (3) replace ORB-SLAM3 by cuVSLAM where parallel training needs a tracker per environment,
and (4) test whether a high level that sees tracking-quality signals learns to move differently when localization
degrades.

## Claims and status

| # | Claim | Evidence | Status |
| --- | --- | --- | --- |
| C1 | The Isaac Lab Spot reward functions and PPO settings make the paper's low level walk; the literal Eqs. 2-6 make it trot in place | `docs/low_level_reward.md` (measurements table) | done |
| C2 | A reward that matches air and contact times only per foot pair can be satisfied by a limp (75 / 25 diagonal duty). A left-right mirror loss removes it (each foot down 49-54 %) without changing the reward | `docs/low_level_reward.md`, "Limp" | done, one seed |
| C3 | Turning in place is rarely commanded in the standard command ranges; 20 % turn-in-place commands fix it (wz 0.60 / -0.30 at +-0.4 in place) | same, `MirrorTurn` table | done, small turns still weak |
| C4 | A high level trained on one fixed map memorizes its route (Map A model: 100 % on Map A, 0 % on Map B). Random mazes train a policy that transfers: 100 % (fixed start) / 69-81 % (random start) on Map A, but only 20 % / 31-50 % on Map B | `Spot-Nav-Maze-v0`, E5 table below | measured; Map B not good enough yet |
| C5 | cuVSLAM stereo drifts less than ORB-SLAM3 on the same drive (APE 0.57 m vs 3.4-5.0 m, one drive on Map A) | `docs/cuvslam.md` | one drive only |
| C6 | Tracking-quality inputs let the high level adapt when SLAM degrades (A / B / C comparison) | `docs/ROADMAP.md` section 5 | **not started** (main experiment) |

C1-C3 are low-level engineering results and can stand as one section or a short paper of their own. C6 is the main
contribution and depends on C4 (a high level that reaches the goal with ground-truth pose) and on cuVSLAM running
one tracker per training env (tested with 16 envs: works, 55 env-steps/s, 6.5 of 7.6 GiB GPU; `docs/cuvslam.md`).

## Section skeleton

1. **Introduction.** Disaster-site navigation with a legged robot; why visual SLAM and RL are combined; what the
   reproduced paper left open (limited training variety, GT pose in training, Map C failure); our contributions
   C1-C6.
2. **Related work.** Learned legged locomotion (see `references.bib`), symmetry in locomotion RL (Mittal et
   al., ICRA 2024: mirror loss), hierarchical navigation on a frozen low level, visual SLAM (ORB-SLAM3, cuVSLAM),
   SLAM-aware or uncertainty-aware planning. Bibliography starts from `references/references.bib`.
3. **System.** Isaac Lab 3.0, Spot, cameras (depth for the low level, RGB-D / stereo for SLAM), ROS 2 bridge for
   ORB-SLAM3, in-process cuVSLAM, one diagram (`docs/SYSTEM.md`). Rates: physics 500 Hz, low level 25 Hz, high
   level 12.5 Hz.
4. **Low-level locomotion.**
   1. Observation (353 dims), action, network.
   2. Reward: paper Table 3 weights on Isaac Lab functions, Eq. 5 wall term (our own); why the literal equations fail.
   3. Training terrain: flat + rough, obstacles, walls; soft contact costs.
   4. Gait symmetry: the limp, why the gait reward misses it, the mirror loss.
   5. Turning in place: command distribution.
5. **High-level navigation.** Observation in the start frame (Table 5), action and smoothing (Eq. 7), reward (ours),
   PPO (Table 6), training on random mazes, why not on the evaluation maps.
6. **SLAM in the loop.** ORB-SLAM3 RGB-D and stereo (ROS 2), cuVSLAM stereo (in-process), reset between episodes,
   pose-error measurement (evo), quality signals (tracking state, feature count).
7. **Experiments.** Setup (maps, start / goal, protocol of Table 9, hardware), E1-E6 below.
8. **Results.** One subsection per experiment, tables and figures below.
9. **Discussion and limitations.** Simulation only, one seed per policy, Map C and the training maze of the original
   paper are lost, small number of trials, GT pose in training for conditions A, cuVSLAM and ORB-SLAM3 differ in more
   than the algorithm (inputs, loop closure, transport).
10. **Conclusion and future work.**

## Experiments

| # | Experiment | Fixed | Varied | Metrics | Status |
| --- | --- | --- | --- | --- | --- |
| E1 | Low level: reward | terrain, obs, PPO | literal Eqs. 2-6 vs Isaac Lab functions | tracking error, gait pattern, falls | done |
| E2 | Low level: terrain | reward, PPO | paper / rough / obstacles / walls | speed error, turn, wall contact, resets | done (walls chosen) |
| E3 | Low level: symmetry and turning | walls terrain | plain / diagonal penalty / foot-duty penalty / mirror loss / + turn commands | foot duty, swing time, wz at vx 0 and 0.5 | done, one seed each |
| E4 | cuVSLAM vs ORB-SLAM3 | drive, cameras | SLAM system | APE (evo), tracking loss, delay, CPU / GPU | one drive, **more runs TBD** |
| E5 | High level: training distribution | low level, PPO, reward | fixed Map A / random mazes | success, collisions, steps, time on A and B | first run done (table below); Map B fails |
| E6 | High level: pose source (A / B / C) | low level, task, reward, budget | GT / SLAM / SLAM + quality inputs | Table 9 metrics under cuVSLAM and ORB-SLAM3 | not started |
| E7 | System load (paper Tables 7-8) | | RL / + VSLAM / + TSDF | FPS, CPU, RAM, disk | script exists, rerun with new policies |

E5 protocol follows the reproduced paper's Table 9: one run per trial, 3000 high-level steps, a contact is counted
and does not end the run, success when the goal radius is reached (Map A 1 m, Map B 2 m), ground-truth pose first.
Script: `scripts/tools/eval_nav_protocol.py`. Results (2026-10-19, ground-truth pose; "fixed start" = `-GT-Play`, a
deterministic policy, so its 5 runs are nearly one run; "random start" = training start, 16 runs):

| Policy (trained on) | Map | Start | Runs | Success | Collisions | Steps (failure = 3000) |
| --- | --- | --- | --- | --- | --- | --- |
| Map A | A | fixed | 5 | 100 % | 0 | 357 |
| Map A | B | fixed | 5 | 0 % | 520 | 3000 |
| random mazes | A | fixed | 5 | 100 % | 0 | 428 |
| random mazes | B | fixed | 5 | 20 % | 64 | 2475 |
| random mazes | A | random | 16 | 69 % | 75 +- 137 | 1327 |
| random mazes | B | random | 16 | 50 % | 34 +- 64 | 1711 |
| mazes + halls (`Spot-Nav-Hall-v0`) | A | fixed | 5 | 100 % | 0 | 445 |
| mazes + halls | B | fixed | 5 | 20 % | 136 | 2670 |
| mazes + halls | A | random | 16 | 81 % | 47 +- 141 | 922 |
| mazes + halls | B | random | 16 | 31 % | 100 +- 142 | 2213 |
| *reproduced paper, SLAM pose* | A / B | fixed | 5 | *80 % / 100 %* | *0 / 0.4* | *1540 / 1228* |

Reading: both policies transfer to Map A (hall fine-tuning: 69 % -> 81 % with random start) and fail on Map B. The
hall layout (wall rows with doorways, shelves, goals 8-22 m) did **not** fix Map B (random start 50 % -> 31 %, fixed
start still 1 of 5), so the cause is not what the first guess said. Cause (trajectories of 16 random starts,
`docs/img/mapB_hall_policy_trajectories.png`): 9 of the 11 failed runs end motionless inside the center shelf at the
y = 3 doorway (x -1.1..0.6, y 1.6..4.7), 2 stick against the pallet bin near the start. Map B's `Shelf_1` is an open
rack (482 collision meshes: boards 1.08 m wide from z = 0.96 m up, upright frames every 4 m), so crossing the
doorways means walking under the shelf, a squeeze that no training world contains. The successful runs (5 of 16)
make it through. Condition A is good enough on Map A, not on Map B. A 400-step
test (the training length) was too short and is not comparable with Table 9.

## Configuration summary

### Low level (`Spot-Paper-WallsSoftMirrorTurn-v0`, `policies/walls_soft_mirror_turn/`)

| Item | Value | Source |
| --- | --- | --- |
| Sim dt / decimation / control rate | 0.002 s / 20 / 25 Hz | paper Table 1 |
| Observation | 353 = lin vel 3, ang vel 3, gravity 3, joint pos 12, joint vel 12, last action 12, command 3, depth 15 x 20 = 300, depth stats 5 | paper Table 2 |
| Action | 12 joint position offsets, scale 0.2, no clipping | paper, Isaac Lab Spot |
| Network | actor / critic [512, 256, 128] ELU, separate | paper Table 4 |
| Reward | Table 3 weights on Isaac Lab Spot functions; Eq. 5 wall term ours; base-velocity sigma 0.25; contact costs instead of termination | `docs/low_level_reward.md` |
| PPO | rl_games a2c, lr 1e-3 adaptive (KL 0.01), horizon 24, 5 epochs, 4 minibatches, entropy 0.0025, 512 envs | Isaac Lab Spot settings |
| Mirror loss | coefficient 1.0 on action and value, signed-permutation mirror of the 353-dim obs | `spot_vslam/rl_games_mirror.py` |
| Commands | vx 0-1 m/s, wz +-0.4 rad/s, 10 % standing, 20 % turn in place (|wz| 0.1-0.4) | `mdp/commands.py` |
| Terrain | 5 x 5 tiles of 8 m: flat, rough, obstacles, wall tiles, terrain-level curriculum | `paper_env_cfg.py` |
| Training budget | WallsSoft 14000 epochs, Mirror to 21000, MirrorTurn to 24000 (fine-tuned in steps) | `policies/README.md` |

### High level (`Spot-Nav-*`, `Spot-Nav-Maze-v0`)

| Item | Value | Source |
| --- | --- | --- |
| Rate | 12.5 Hz (decimation 2 on the 25 Hz low level), EMA alpha 0.25 | paper Table 6, Eq. 7 |
| Observation (9) | pose x, y, yaw in the start frame; goal dx, dy; goal heading error; base vx, vy; base wz | paper Table 5 |
| Action (2) | vx 0-1 m/s, wz +-0.4 rad/s (policy output in [-1, 1], scaled) | paper Table 6 |
| Reward (ours) | +5 per metre towards the goal, +10 on success, -0.01 per step, -5 on body / leg contact | `navigation_env_cfg.py` |
| PPO | rl_games a2c, lr 3e-4 constant, 16 envs x horizon 64 = 1024 per update, batch 128, 4 epochs, gamma 0.99, GAE 0.95, clip 0.2, entropy 0.01, [256, 128] ELU, no input normalization | paper Table 6 mapped from SB3 |
| Episode | 400 high-level steps in training (32 s), 3000 in evaluation | paper Tables 6, 9 |
| Training pose | ground truth + Gaussian noise 0.01 (condition A) | paper (trained without SLAM) |
| Training world | Map A or B (fixed start, goal), or random mazes | `Spot-Nav-Maze-v0` |
| Maze | 64 tiles of 21 x 21 m, 7 x 7 cells, random spanning tree + 30 % extra openings, walls 2.5 m high, start and heading random, goal 6-14 m away, goal radius 1 m | `tasks/navigation/mdp/maze.py` |
| Budget | 490 epochs = 500k steps (paper); runs here continued to 2000 epochs | `docs/ROADMAP.md` |

### SLAM and evaluation

| Item | Value |
| --- | --- |
| ORB-SLAM3 | RGB-D (`-Play`), stereo (`-Stereo-Play`), ROS 2 node, one robot |
| cuVSLAM | stereo, in-process, one `Tracker` per env, loop closure on (`-cuVSLAM-Play`) |
| Maps | A "Room": (-9, -10) -> (-4, 15.5), radius 1 m; B "Warehouse": (-9, 17) -> (7, 0), radius 2 m; C lost |
| Start | map start facing the goal, no push (evaluation) |
| Pose error | evo APE against ground truth |

## Figures and tables to make

| Item | Content | Data |
| --- | --- | --- |
| Fig. 1 | System diagram: two policies, SLAM, ROS 2 / in-process paths | `docs/SYSTEM.md` |
| Fig. 2 | Foot contact timelines: limp vs mirror-loss gait | `scripts/tools/gait_check.py`, `jitter2.py` style plot |
| Fig. 3 | Training mazes (6 of 64) with start / goal; Map A and B top views | `output/maze_examples.png` |
| Fig. 4 | Training curves: success rate vs epoch, maze vs fixed map | rl_games summaries |
| Fig. 5 | Trajectories on Map A and B per policy, GT vs SLAM | run logs |
| Fig. 6 | Pose error over time, ORB-SLAM3 vs cuVSLAM | evo |
| Tab. 1 | Low-level reward: literal vs Isaac Lab | `docs/low_level_reward.md` |
| Tab. 2 | Gait symmetry: duty per foot and turn rates for each policy | same |
| Tab. 3 | High-level training distribution (E5): success, collisions, steps, time | table in the E5 section; Map B to redo |
| Tab. 4 | A / B / C comparison with each SLAM (E6) | not started |
| Tab. 5 | System load (E7) | script exists |

## Order of work

1. E5: done once: Map A passes, Map B does not (20 % fixed start, 50 % random start).
2. Hall fine-tune (`Spot-Nav-Hall-v0`, 3300 epochs) done: A 81 %, B 31 % (random start). Failure found: the policy
   gets stuck in the shelf squeeze at Map B's y = 3 doorway. Next: add open racks (boards at 0.96 m, upright frames
   every 4 m, ~0.9 m between them) to the hall tiles and fine-tune again, or keep Map B as an out-of-distribution
   test like the paper's Map C. Condition A is done when Map B is close to Map A.
3. cuVSLAM with 16 envs: done, works (55 env-steps/s).
4. SLAM in the loop for the finished high level (E4, then the pose-source part of E6).
5. Train B and C, evaluate with cuVSLAM, ORB-SLAM3 and GT poses (E6).
6. Write up; fill the TBD cells.

## Honest limits to state in the paper

- Map C and the original high-level training maze are lost; Map C results cannot be reproduced. A self-made
  corridor map can serve as an out-of-distribution test instead.
- One seed per policy. Differences between low-level variants smaller than a seed-to-seed spread are not claims.
- Fixed start and a deterministic policy make repeated trials nearly identical; report random-start runs too.
- The high-level reward and the maze are ours; the original paper gives neither.
- cuVSLAM and ORB-SLAM3 are compared on one drive so far.
