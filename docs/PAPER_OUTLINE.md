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
| mazes + halls + racks (`Spot-Nav-HallRacks-v0`) | A | fixed | 5 | 100 % | 0 | 363 |
| mazes + halls + racks | B | fixed | 5 | 0 % | 70 | 3000 |
| mazes + halls + racks | A | random | 16 | 94 % | 22 +- 88 | 553 |
| mazes + halls + racks | B | random | 16 | 62 % | 53 +- 94 | 1381 |
| *reproduced paper, SLAM pose* | A / B | fixed | 5 | *80 % / 100 %* | *0 / 0.4* | *1540 / 1228* |

Reading: both policies transfer to Map A (hall fine-tuning: 69 % -> 81 % with random start) and fail on Map B. The
hall layout (wall rows with doorways, shelves, goals 8-22 m) did **not** fix Map B (random start 50 % -> 31 %, fixed
start still 1 of 5), so the cause is not what the first guess said. Cause (trajectories of 16 random starts,
`docs/img/mapB_hall_policy_trajectories.png`): 9 of the 11 failed runs end motionless inside the center shelf at the
y = 3 doorway (x -1.1..0.6, y 1.6..4.7), 2 stick against the pallet bin near the start. Map B's `Shelf_1` is an open
rack (482 collision meshes: boards 1.08 m wide from z = 0.96 m up, upright frames every 4 m), so crossing the
doorways means walking under the shelf, a squeeze that no training world contains. The successful runs (5 of 16)
make it through. Open racks (`Spot-Nav-HallRacks-v0`, boards from 0.85-1.1 m, 0.86-1.26 m between the posts) raised
Map B from 31 % to 62 % (random start) and Map A to 94 %, but the fixed start still sticks (0 of 5), always inside
the rack between y = 3 and 6 (`docs/img/mapB_racks_policy_trajectories.png`; all 9 failed runs of the trajectory
test end at x -1.0..-0.1, y 2.7..6.0). Cause found in the USD: Map B's rack posts are 0.2 m wide with 0.6 m between
them, narrower than our racks (0.86-1.26 m). Condition A is good enough on Map A, close on Map B (random start). A 400-step
test (the training length) was too short and is not comparable with Table 9.

## Pose source: GT vs SLAM (E6 part, condition A policy `nav_hall_racks`)

Table 9 protocol, fixed start facing the goal (`-GT-Play`, `-cuVSLAM-Play`), 5 runs, 3000 steps, 2026-10-19:

| Pose source | Map | Success | Collisions | Steps | SLAM position error, mean per run (m) | Steps without a pose |
| --- | --- | --- | --- | --- | --- | --- |
| GT | A | 100 % | 0 | 363 | | |
| cuVSLAM stereo | A | 100 % | 0 | 363 | 0.20-0.39 (max 1.0) | 0 % |
| GT | B | 0 % | 70 | 3000 | | |
| cuVSLAM stereo | B | 20 % | 28 | 2472 | 0.28 / 4.1 / 3.7 / 3.6 / 12.8 (max 14.6) | 0 % |

On Map A the SLAM pose costs nothing (same success, same steps, error under 0.4 m). On Map B GT is already stuck in the
rack, so the extra failures cannot be blamed on SLAM; but the four failed cuVSLAM runs also have 3.6-12.8 m error with
a pose still reported every step (`status` = 1), a case where a returned pose is not healthy tracking
(`docs/ROADMAP.md` section 5). ORB-SLAM3 RGB-D, 5 runs per map, ORB-SLAM3 restarted for every run (2026-10-20,
`scripts/tools/run_nav_protocol_orbslam.sh`):

| Pose source | Map | Success | Collisions | Steps | SLAM position error, mean per run (m) | Steps without a pose |
| --- | --- | --- | --- | --- | --- | --- |
| ORB-SLAM3 RGB-D | A | 0 % | 0 | 3000 | 5.9 / 5.9 / 6.2 / 10.9 / 5.9 (max 6.6-27.9) | 0.7 % (run 4: 11 %) |
| ORB-SLAM3 RGB-D | B | 100 % | 0.2 | 398 | 3.9 (max 7.1-7.2) | 5 % |

Map A: the robot walks 25 m but stops about 4 m sideways of the goal. The SLAM error is 0.25 m sideways per metre
driven, so the policy believes it is near the goal. Map B: all 5 runs reach the goal, although the pose error is
3.9 m, and GT pose fails on the same start (stuck in the rack). So the wrong pose changes the route and, by chance,
avoids the rack: success on Map B with ORB-SLAM3 is not evidence of good tracking. The five runs are almost identical
(deterministic policy, fixed start), so they are one sample, not five. Paper (ORB-SLAM3): A 80 %, B 100 %.

### Random start (10 runs per cell, 2026-10-20)

Start +-0.5 m in x and y, heading +-0.5 rad around the goal direction, no push; every run has its own start. Same
starts for GT and cuVSLAM (seed 1, 10 envs); ORB-SLAM3 one run per seed 1-10, restarted each time. 3000 steps,
`scripts/tools/eval_nav_protocol.py --jitter_xy 0.5 --jitter_yaw 0.5`. Policy `nav_hall_racks`.

| Pose source | Map | Success | Collisions (mean) | Steps of successful runs | SLAM error, mean per run (m) |
| --- | --- | --- | --- | --- | --- |
| GT | A | 10/10 | 0 | 365 | |
| cuVSLAM stereo | A | 10/10 | 0.2 | 364 | 0.17-0.47 |
| ORB-SLAM3 RGB-D | A | 5/10 | 3.8 | 1382 (623-2940) | 7-22 |
| GT | B | 4/10 | 41 | 785 (346-2091) | |
| cuVSLAM stereo | B | 5/10 | 48 | 389 | 0.08-0.75 (successes), 1.7-15 (3 of 5 failures) |
| ORB-SLAM3 RGB-D | B | 8/10 | 22.6 | 523 (416-1181) | 6.5-12.8 |

Reading:

- **cuVSLAM matches GT** on Map A (10/10, error under 0.5 m) and is equal or slightly better on Map B (5/10 vs 4/10;
  with 10 runs that difference is noise). All 5 successful cuVSLAM runs on Map B have error under 0.8 m. Of the 5 failures,
  3 have drift of 1.7-15 m and 2 have low error (0.08, 0.75 m): those fail the same way GT does, stuck in the
  rack. So Map B failures have two causes, the rack and SLAM drift; only the second is a SLAM problem.
- **ORB-SLAM3 RGB-D** has a large error everywhere (7-22 m on Map A: it opens new maps after losing track, so the pose
  jumps back to the origin). Its successes on Map A (5/10) take 600-2900 steps and only follow by wandering with a wrong
  pose, the same order as the paper's 1228-1540 steps. Its Map B success (8/10) is also luck: GT pose fails 6/10 on the
  same starts. A success rate alone does not show good SLAM; the pose error must be reported with it.
- The fixed-start ORB-SLAM3 result (A 0 %, B 100 %) was one sample. With random starts it is A 50 %, B 80 %, a different
  picture from the 5 near-identical runs, so fixed-start numbers must not be used alone.
- For the A / B / C experiment: the cuVSLAM status flag was 1 in every step, also in the drifting runs, so a quality
  signal has to come from more than `status`.

### Quality signals of cuVSLAM (E6, measured 2026-10-20)

16 random-start runs per map with `nav_hall_racks`, cuVSLAM stereo, per-step trace
(`eval_nav_protocol.py --trace_npz`, `scripts/tools/analyze_slam_quality.py`). Map B: 29 894 steps, mean error 5.1 m,
54 % of the steps above 1.5 m; Map A: 5 886 steps, error 0.31 m, no step above 1.5 m. Score = AUROC over the Map A + B
steps (0.5 = no information):

| Signal | Error above 1.5 m now | Error grows by 0.5 m in the next 25 steps |
| --- | --- | --- |
| `status` (pose returned) | none: 1 in every step | none |
| `num_features` | 0.61 | 0.78 |
| `num_features`, mean of last 10 steps | 0.69 | 0.91 |
| gap between odometry and loop-closure pose | 0.88 | 0.74 |
| `track_time_ms` | 0.64 | 0.80 |

Reading: the gap is the best sign that the pose is wrong now (healthy runs: gap about 0; wrong runs: median 0.85 m,
90th percentile 10 m); the averaged feature count is the best early warning of drift. Limits: steps of one run are not
independent, the scores pool all runs (a run that sits in a feature-poor corner counts many times), and the growth
label is rare (3 % of the steps). So these are signs that the inputs carry information, not a proof that a policy can
use them. Condition C gets three inputs: features, averaged features, gap (`mdp.slam_quality`).

### A / B / C, first test (300 epochs, 2026-10-21)

All three start from `nav_hall_racks` (epoch 7500); B and C train 300 more epochs in textured HallRacks worlds with one
cuVSLAM stereo tracker per env (`Spot-Nav-HallRacks-cuVSLAM-v0`, `-Quality-v0`); A is `nav_hall_racks` itself (GT pose,
no SLAM in training). Evaluation: cuVSLAM stereo pose, 16 random starts per map (same starts for all, seed 2), 3000
steps, one seed per policy. "Error" = SLAM position error per run (m).

| Policy | Map | Success | Collisions (mean) | Steps of successful runs | Error mean / median | Runs with error above 1.5 m |
| --- | --- | --- | --- | --- | --- | --- |
| A: GT pose | A | 16/16 | 0 | 369 | 0.31 / 0.28 | 0/16 |
| B: SLAM pose | A | 14/16 | 46 | 444 | 0.56 / 0.47 | 0/16 |
| C: SLAM pose + quality | A | 14/16 | 70 | 562 | 0.45 / 0.38 | 0/16 |
| A | B | 7/16 | 23 | 414 | 3.48 / 1.00 | 7/16 |
| B | B | 3/16 | 117 | 386 | 5.35 / 2.58 | 10/16 |
| C | B | 7/16 | 167 | 440 | 0.67 / 0.28 | 3/16 |

Reading (a first test, not a result):

- Training with SLAM in the loop (B) did not help in 300 epochs: slightly worse than A on both maps. The difference
  between 3/16, 7/16 and 7/16 is within what 16 runs and one seed can give.
- C keeps the tracker healthier on Map B: 3 of 16 runs drift above 1.5 m (A: 7, B: 10) and the mean error is 0.67 m
  (A 3.5, B 5.4). Success does not rise because most Map B failures are the rack (C has 167 collisions there).
  This fits "quality inputs change how the robot moves so that tracking stays good", but it is one seed and one
  300-epoch run; it needs the longer run and more seeds before it is a claim.
- The first B attempt (plain grey worlds) failed because cuVSLAM could not track there: median pose error 8.8 m, 83 %
  of the steps above 1.5 m. With the marble texture: 0.23 m, 5 %. A condition-B experiment needs a world the SLAM can
  track in; check this first for any new training world.

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
2. Hall (3300 epochs): A 81 %, B 31 %. HallRacks (2200 epochs): A 94 %, B 62 % (random start); fixed start on B
   still 0 of 5. Next: racks with Map B's real opening (0.6-1.1 m between 0.2 m posts, new task id), fine-tune again,
   or keep Map B as an out-of-distribution test like the paper's Map C. Condition A is done when Map B is close to
   Map A.
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
