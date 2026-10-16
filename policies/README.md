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
| `walls_soft_mirror/` | WallsSoft + left-right mirror loss: no limp (each foot down ~50 %) | `Spot-Paper-WallsSoftMirror-v0` | `2026-10-16_15-41-52`, ep 18000 (fine-tuned from WallsSoft ep 14000) |
| `high_level_paper_sb3/` | SB3 high level (9-dim obs with goal), trained on an older paper low level, before 2026-10-07 | `Spot-Paper-high-level-Map{A,B}-*` | `output/checkpoints/high_level_paper/high_level_policy.zip` |

Each low-level folder has `model.pth` (rl_games) and the run's `agent.yaml` / `env.yaml`. Use them as
`--checkpoint policies/walls_soft/model.pth` or `SPOT_VSLAM_LOW_LEVEL_CKPT=policies/walls_soft/model.pth`.

SHA-256:

```
b04dbbab13d83d6436e2540ec6a02cef58a592f3d3cc0d01eb2a33567815f028  paper_isaaclab_reward_reproduce/model.pth
d7b6cf4c6215ee2a31c8eaa24351774a0e3a32fa06eba8be4a86ebb2daf26db5  walls_soft/model.pth
4e9ea49b5d6f047e9a81c4b461e510f0b17a1468212ad2b3691ce896b4b9c2d7  rough/model.pth
156aec6f34827b21de7c61ba66fa15185f5f800fd55f32a8cc57a3c90ffd6fd8  walls_soft_noise/model.pth
662da68f340961b1afaa3f5b583f04c2fb152efc9448b16bab49176d5e9d281e  walls_soft_mirror/model.pth
246ccfd5fb3b90e278b2e706f57d3ca4a4cb9b89420321c846af72b5a854d4b9  high_level_paper_sb3/model.zip
```
