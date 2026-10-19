# Trained policies

Final checkpoints of the paper-version policies, copied from the git-ignored `logs/rl_games/spot_paper_ppo/<run>/`
(and `output/checkpoints/`) so a fresh clone can play and evaluate them. Intermediate checkpoints stay in `logs/`.
The `rl_vslam_v0` policies are in `baselines/rl_vslam_v0/policies/`. Policy roles: `docs/ROADMAP.md` section 1.

| Folder | Policy | Task | Source run, epoch |
| --- | --- | --- | --- |
| `paper_isaaclab_reward_reproduce/` | paper reproduction (Table 3 weights on Isaac Lab's Spot rewards) | `Spot-Paper-v0` | `2026-10-08_11-08-43`, ep 20000 |
| `walls_soft/` | WallsSoft, the main low level | `Spot-Paper-WallsSoft-v0` | `2026-10-12_12-50-26`, ep 14000 |
| `rough/` | Rough terrain | `Spot-Paper-Rough-v0` | `2026-10-12_18-25-42`, ep 12000 |
| `walls_soft_noise/` | WallsSoft + noise on every sensor | `Spot-Paper-WallsSoftNoise-v0` | `2026-10-13_11-29-20`, ep 14000 |
| `walls_soft_mirror/` | WallsSoft + left-right mirror loss: no limp (each foot down ~50 %) | `Spot-Paper-WallsSoftMirror-v0` | `2026-10-16_19-10-05`, ep 21000 (WallsSoft ep 14000 → `2026-10-16_15-41-52` ep 18000 → here) |
| `walls_soft_mirror_turn/` | WallsSoftMirror + 20 % turn-in-place commands: no limp, turns in place | `Spot-Paper-WallsSoftMirrorTurn-v0` | `2026-10-16_23-38-11`, ep 24000 (from `walls_soft_mirror` ep 21000) |
| `nav_map_a/` | rl_games high level trained on Map A (fixed start and goal), GT pose; low level `walls_soft_mirror_turn` | `Spot-Nav-MapA-v0` | `2026-10-18_01-23-40` ep 340 → `2026-10-18_11-48-30` ep 2000 |
| `nav_maze/` | high level trained on 64 random mazes, random start and goal 6-14 m | `Spot-Nav-Maze-v0` | `2026-10-18_16-44-48`, ep 2000 |
| `nav_hall/` | `nav_maze` fine-tuned on mazes (30 %) + halls with wall rows and shelves (70 %), goals 8-22 m | `Spot-Nav-Hall-v0` | `2026-10-19_00-56-16`, ep 2000 → 5300 |
| `nav_hall_racks/` | `nav_hall` fine-tuned with open racks in 40 % of the hall doorways (walk under them) | `Spot-Nav-HallRacks-v0` | `2026-10-19_11-52-33`, ep 5300 → 7500 |
| `high_level_paper_sb3/` | SB3 high level (9-dim obs with goal), trained on an older paper low level, before 2026-10-07 | `Spot-Paper-high-level-Map{A,B}-*` | `output/checkpoints/high_level_paper/high_level_policy.zip` |

Each low-level folder has `model.pth` (rl_games) and the run's `agent.yaml` / `env.yaml`. Use them as
`--checkpoint policies/walls_soft/model.pth` or `SPOT_VSLAM_LOW_LEVEL_CKPT=policies/walls_soft/model.pth`.

SHA-256:

```
b04dbbab13d83d6436e2540ec6a02cef58a592f3d3cc0d01eb2a33567815f028  paper_isaaclab_reward_reproduce/model.pth
d7b6cf4c6215ee2a31c8eaa24351774a0e3a32fa06eba8be4a86ebb2daf26db5  walls_soft/model.pth
4e9ea49b5d6f047e9a81c4b461e510f0b17a1468212ad2b3691ce896b4b9c2d7  rough/model.pth
156aec6f34827b21de7c61ba66fa15185f5f800fd55f32a8cc57a3c90ffd6fd8  walls_soft_noise/model.pth
bd7f3f6afef99cbd026926f465e0b1515d359523d09ce88262c062f9ae8c927d  walls_soft_mirror/model.pth
f652c193098072ce8a2a918ff95e5a8409815c2d3ef1818df8ae24bd7195afa6  walls_soft_mirror_turn/model.pth
246ccfd5fb3b90e278b2e706f57d3ca4a4cb9b89420321c846af72b5a854d4b9  high_level_paper_sb3/model.zip
1157b901816aa93a91c502f77ab08f1820e84a3519930b0ad1aed375fb1949ef  nav_map_a/model.pth
a3943747d03b7949cbc49a63a41ac74d9cfcf21cfd8ec6ea3871b5c66b0cbcf5  nav_maze/model.pth
e0ff24a78f1236fd3f1d56529d232912f1c02d2289c81dc0de21cb566e341626  nav_hall/model.pth
87394652ebe14c66595b1373d9e422ca5524188fb6000ed35dc5d4ca05ff1274  nav_hall_racks/model.pth
```
