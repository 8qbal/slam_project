# Design decisions: why it is built this way

One row per choice: what we picked, what else was possible, why, and where the evidence is. Choices that are only a
guess are listed again at the end. Outline of the paper: `docs/PAPER_OUTLINE.md`; run history: `docs/ROADMAP.md`.

## Low-level locomotion

| Decision | Alternatives | Why | Evidence / where |
| --- | --- | --- | --- |
| Table 3 weights on **Isaac Lab Spot reward functions** | Implement Eqs. 2-6 literally | The literal equations made the robot trot in place: 0.00 m/s at every commanded speed, although the trot pattern share was 0.64-0.80. The Isaac Lab functions give 0.54-0.69 m/s | `docs/low_level_reward.md`, "Measurements" |
| Eq. 5 (wall term) kept as ours | Drop it | The paper names it and it is the only depth-based term; `wall_min_depth` follows the equation | `mdp/rewards.py` |
| **PPO settings of Isaac Lab's Spot** (no action clipping, separate actor / critic, no obs or value normalization, lr 1e-3 adaptive) | Paper-like clipped actions, shared trunk | 81-92 % of the old policies' outputs sat on the [-1, 1] clip; removing it matched the reference Spot setup | same doc, "PPO settings" |
| `base_linear_velocity` sigma 0.25, gait / air-time thresholds off, 10 % standing envs | Keep Isaac Lab defaults (sigma 1) | Sigma 1 charged only 1.5 of 5 for a 0.35 m/s error, so the policy lagged the command; the 0.5 m/s thresholds paid a standing robot for walking | same doc, "Speed-tracking fix" |
| **Walls terrain** (long, 2.5-6 m tall walls, plus pillars and rough ground) | Paper terrain (flat + rough) only, obstacles only | The obstacle-trained policy stopped turning on Map A, where a wall near the start fills the whole depth image; wall tiles imitate the maps' walls and shelves | same doc, "Wall terrain", "Training-terrain study" |
| **Soft contacts**: a body / leg contact costs reward instead of ending the episode | Terminate on contact | Table 9 of the paper counts collisions during a run and does not stop it; the policy must be able to bump and recover | `paper_env_cfg.py` `WallsSoft`, `SpotPaperHighLevelEnvCfg_PLAY` |
| **Mirror loss** in PPO (coefficient 1.0, signed permutation of the 353-dim obs and the action) | Penalty terms on foot duty, symmetric data augmentation, a different reward | Spot in Isaac Sim is exactly mirror-symmetric (masses, centers of mass), so the limp is a learned symmetry break. The gait reward only matches air and contact times between foot pairs, so a 75 / 25 diagonal split earns it in full. Penalties only moved the limp (duty 74 / 28 became 65 / 36 and about 60 / 35) and cost turning; the mirror loss gave 49-54 % per foot | `docs/low_level_reward.md`, "Limp"; `rl_games_mirror.py`; method from Mittal et al., ICRA 2024 |
| Mirror as a **loss**, not augmented rollouts | Augment the batch with mirrored samples | Smallest change to rl_games (one extra term in `calc_losses`); checkpoints keep the standard `a2c_continuous` format, so every script loads them | `rl_games_mirror.py` |
| **20 % turn-in-place commands** (vx = 0, |wz| 0.1-0.4) | Wider command ranges, more standing envs | Commands draw vx from U(0, 1), so vx near 0 with wz != 0 almost never occurs; turning in place stayed at 0.00-0.17 rad/s. With the extra commands it reaches 0.60 / -0.30 at +-0.4 | same doc, "MirrorTurn" |
| Fine-tune in steps from the previous policy (14000 -> 21000 -> 24000) | Retrain from scratch each time | A retrain takes about 5.5 h per 8000 epochs; each fix changes one thing and keeps the earlier gait | `policies/README.md` |
| **Resume at the checkpoint's learning rate** | rl_games default | rl_games restarts the adaptive lr at 1e-3 instead of the checkpoint's ~3e-5; a resumed run fell in 80 % of episodes until a control run showed the lr was the cause | `scripts/rl_games/train.py` |
| **NaN guard** (clamp rewards and observations, end episodes on non-finite or absurd velocities) | None | A scratch run's weights turned to NaN near epoch 8750; the guard made resuming from epoch 8700 safe | `envs/rl_env.py`, `mdp/terminations.py` |
| One low level for every high-level experiment | A low level per experiment | The roadmap rule: change one thing at a time, so comparisons are about the high level only | `docs/ROADMAP.md` |

## High-level navigation

| Decision | Alternatives | Why | Evidence / where |
| --- | --- | --- | --- |
| **9-dim observation in the start frame** (pose, goal, goal heading, base velocity) | World frame, add depth stats (our old 11-dim) | Table 5 of the paper; ORB-SLAM3 reports poses relative to its first frame, so GT and SLAM poses go through the same transform | `mdp/commands.py`, `mdp/observations.py` |
| Action vx 0-1 m/s, wz +-0.4 rad/s, EMA alpha 0.25 | Wider ranges, no smoothing | Taken from the paper (Table 6, Eq. 7); alpha 0.25 prevents step changes from upsetting the gait | `mdp/actions.py` |
| Reward: **+5 per metre towards the goal, +10 on success, -0.01 per step, -5 on contact** | Sparse success only, paper reward | The paper gives no high-level reward. Progress shaping is the simplest signal that scales to any map; weights are per-step values divided by the step time because Isaac Lab multiplies rewards by it | `navigation_env_cfg.py` |
| **PPO of Table 6 mapped to rl_games**: 16 envs x horizon 64 = 1024 per update, batch 128, 4 epochs, 500k steps (490 epochs) | SB3 directly | The paper used SB3 (256 steps x 4 envs). The rest of our stack is rl_games; both give 1024 samples per update | `agents/rl_games_ppo_cfg.yaml` |
| 400 steps per training episode, 3000 per evaluation run | Same length for both | Both from the paper (Tables 6, 9). Successful paper runs took 1230-1540 steps. Our policies reach Map A in 360-430 steps with GT pose, but a 400-step test cut off slow runs and counted a contact as the end, which is not Table 9 | `docs/PAPER_OUTLINE.md`, E5 |
| **Train with ground-truth pose** (+ noise 0.01), evaluate with SLAM | Train with SLAM in the loop | The paper does the same. Parallel training needs one tracker per env; ORB-SLAM3 runs as one ROS 2 process and keeps its origin after a reset. This also makes conditions A / B / C a clean comparison | `docs/ROADMAP.md` sections 4, 5 |
| **Random mazes** for training (64 tiles, random start, heading and goal each reset) | Train on Map A or B, rebuild the lost `flat_maze.usd` | A fixed start and goal lets the policy memorize the route: the Map A model reached the goal in 100 % of runs on Map A and 0 % on Map B (E5). The original training maze is lost, and the paper itself calls its training variety limited. Maps A and B stay unseen for evaluation | `tasks/navigation/mdp/maze.py` |
| Maze made from **terrain tiles + flat patches** | USD mazes, one per env | No USD asset exists; terrain tiles are generated in code, each reset can move an env to another tile, and flat patches give start and goal spots that are 0.7 m clear of walls | `maze.py` (`reset_in_maze`) |
| **Hall tiles** (30 % mazes + 70 % halls: 3-5 wall rows with 1.3-2.4 m doorways, 0-4 free-standing shelves with 1 m clearance, 22 x 32 m, goals 8-22 m) fine-tuned from the maze policy | Retrain from scratch, copy Map B's exact layout | Measured from the USD: Maps A and B are 20 x 30 m halls with five wall rows 6 m apart, each with a 1.74 m doorway; Map B adds long shelves. Result: Map A random-start success 69 % -> 81 %, Map B 50 % -> 31 % (E5), so it helped A and not B | `tasks/navigation/mdp/maze.py` (`hall_terrain`), `Spot-Nav-Hall-v0` |
| Maze shape: 7 x 7 cells of 3 m, spanning tree plus 30 % extra openings, 2.5 m walls, goal 6-14 m | Narrower lanes, shelves, longer goals | A first guess at "indoor, walls fill the depth image". **Not tuned**; Map B's narrow shelf lanes and Map A's 26 m goal are outside it | open, see below |
| Start at the map start **facing the goal**, no push, in evaluation | Training-style random heading | A start facing the wall gives SLAM nothing to track; a push made the robot slide before the first command | `docs/cuvslam.md`, `paper_env_cfg.py` |

## SLAM

| Decision | Alternatives | Why | Evidence / where |
| --- | --- | --- | --- |
| **cuVSLAM stereo, in-process**, one tracker per env | ORB-SLAM3 for everything, cuVSLAM RGB-D | RGB-D odometry showed jumps whose cause was not found; stereo gave 0.27 m mean error and no jumps. In-process trackers can be rebuilt on reset and give every env its own SLAM | `docs/cuvslam.md`, `slam/cuvslam_manager.py` |
| ORB-SLAM3 also run in **stereo** (`-Stereo-Play`) and **RGB-D** (`-Play`) | RGB-D only | RGB-D reproduces the paper; stereo gives both systems the same input for the comparison | `navigation_env_cfg.py` |
| Pose error with **evo** against ground truth | Own metric | Standard tool, comparable to other SLAM papers | `docs/cuvslam.md` |
| Quality signals from the tracker: `status` (pose returned) and `num_features` | Covariance, reprojection error | They are what cuVSLAM exposes; each tells only part of the story (a returned pose is not always healthy tracking, feature count is not a confidence), so the paper must say what each does and does not show | `docs/ROADMAP.md` section 5 |

## Project structure

| Decision | Why |
| --- | --- |
| Task ids of trained policies never change | A checkpoint must keep loading under its task id |
| `rl_vslam_v0` frozen (tag `baseline/rl-vslam-v0`) | The old results (245-dim low level, 11-dim high level) stay reproducible; new work never breaks them |
| New variants are new classes and task ids, not edits | Same reason; every kept task's resolved config was compared before and after refactors |
| Final checkpoints copied into `policies/` with SHA-256 | A fresh clone can evaluate without the git-ignored `logs/` |
| Paths from `$ISAACSIM_ASSET_ROOT` and `$SPOT_VSLAM_LOW_LEVEL_CKPT`, never hardcoded | The repo is used on more than one machine |

## Weakly justified: revisit before submitting

| Item | Problem | What would settle it |
| --- | --- | --- |
| Rack opening (0.86-1.26 m between the posts) | Guess. Map B's posts are 0.2 m wide with 0.6 m between them, so ours are looser; the racks policy sticks inside Map B's rack (9 of 9 failed runs) | A tight-rack version (0.6-1.1 m, 0.2 m posts) fine-tuned from `nav_hall_racks` |
| Maze and hall parameters (cell size, openings, doorway width, shelves, goal distance) | Picked by guess. Both transfer to Map A (69-81 % random start) and not to Map B (31-50 %); the hall version, built to look like the maps, did not help B | Done: 9 of 11 failed Map B runs end stuck in the center shelf at the y = 3 doorway. `Shelf_1` is an open rack (boards from z = 0.96 m, upright frames every 4 m), so the route goes under it. Next world version needs racks like that, or Map B becomes an out-of-distribution test |
| High-level reward weights | Ours, not from the paper; no sweep | A small sweep, or state it as a design choice |
| No observation normalization in the high level | SB3's default, but pose and goal values reach 26 m; an input far outside the training range can hurt | Test with normalization on, same budget |
| One seed per policy | Differences within a seed spread are not claims | 2-3 seeds for the final high-level policies |
| Fixed start, deterministic policy | Repeated trials are nearly identical | Report random-start runs next to the fixed-start ones |
| cuVSLAM with 16 parallel trackers | Tested 2026-10-19: works (55 env-steps/s, 6.5 of 7.6 GiB GPU, all trackers valid) but GPU memory is at 85 %; 32 envs and a full training run are untested | A short training run with SLAM in the loop |
| cuVSLAM vs ORB-SLAM3 on one drive | One run is an anecdote | More drives, Map B, several starts |
