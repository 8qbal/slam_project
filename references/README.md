# References

- `main.pdf` — Y.-H. Li and J. Jeong, *System-Level Performance Analysis of Visual SLAM and Reinforcement
  Learning Integration for Quadruped Navigation*, IJCAS (preprint, 2026). This is the paper this project
  implements.
- `references.bib` — the paper's own entry plus its 24 references, keeping the paper's numbering [1]–[24].

## Paper settings and where they live in the code

| Paper | Value | Code |
| --- | --- | --- |
| Low-level sim dt / decimation / control rate (Table 1) | 0.002 s / 20 / 25 Hz | `tasks/locomotion/locomotion_env_cfg.py` |
| Low-level obs (Table 2) | 353-dim, depth 15×20 + 5 depth stats | `tasks/locomotion/locomotion_env_cfg.py`, `tasks/locomotion/mdp/observations.py` |
| Low-level reward (Eqs. 2-6, Table 3) | | `tasks/locomotion/config/spot/paper_env_cfg.py`, `tasks/locomotion/mdp/rewards.py` |
| Low-level MLP (Table 4) | [512, 256, 128], ELU | `tasks/locomotion/config/spot/agents/rl_games_paper_ppo_cfg.yaml` |
| High-level obs (Table 5) | 9-dim: pose, goal, base velocity | `tasks/navigation/config/spot/navigation_env_cfg.py` (`Spot-Nav-*`); SB3 version `legacy/high_level/paper_vec_env.py` |
| High-level PPO (Table 6) | lr 3e-4, n_steps 256, batch 128, 4 epochs, 500k steps | `tasks/navigation/config/spot/agents/rl_games_ppo_cfg.yaml`; SB3 version `scripts/high_level/train_high_level_ppo.py` |
| High-level action | vx ∈ [0, 1] m/s, ωz ∈ [−0.4, 0.4] rad/s, EMA α = 0.25, HL decimation 2 | `tasks/navigation/mdp/actions.py` (`LowLevelPolicyActionCfg`) |
| Max episode length | 400 HL steps | `tasks/navigation/config/spot/navigation_env_cfg.py` |
| Maps A / B (start, goal) | | `assets/maps.py` (`PAPER_MAPS`) |

Paths are under `source/spot_vslam/spot_vslam/`. The notes below describe the rl_vslam_v0 baseline code.

The paper and the code don't fully agree:

- **High-level observations.** Table 5 lists 9 dims (SLAM x, y, ψ; goal Δx, Δy, Δψ; vx, vy, ωz). The code
  (`HighLevelIsaacVecEnv._get_obs_tensor`) uses 11 dims: SLAM pose (3), ORB tracking status (1),
  depth stats (5) and the previous HL action (2). It has no goal term.
- **High-level reward.** The current reward rewards forward progress and penalizes getting stuck, body
  contact and turning. Nothing in it says "reach a goal", so the goal-reaching success rates in Table 9
  can't be reproduced with this reward as-is.
- **Hidden layers.** The paper says the high-level policy uses [256, 128]. The script uses SB3's default
  `MlpPolicy` ([64, 64]).
- **Play ranges.** `scripts/high_level/load_high_level.py` plays with vx ≤ 0.5 and |ωz| ≤ 0.25, which is
  narrower than the training ranges (vx ≤ 1.0, |ωz| ≤ 0.4).
