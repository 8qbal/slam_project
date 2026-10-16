"""Left-right mirror of the low-level policy's observation and action (docs/low_level_reward.md, "Limp").

Like Isaac Lab's ANYmal symmetry (``isaaclab_tasks/core/velocity/mdp/symmetry/anymal.py``, Mittal et al., ICRA 2024),
but for rl_games and our 353-dim observation. Every part of the mirror is a signed permutation, so it is stored as
``(index, sign)`` with ``mirrored = x[:, index] * sign``:

- body velocities / gravity / command: y, roll and yaw rates, and the yaw command change sign,
- joints (joint_pos, joint_vel, last actions; the action itself): left and right legs swap, ``*_hx`` (abduction)
  changes sign. Spot's default pose is mirror-symmetric (hx +-0.1), so relative joint positions mirror the same way,
- depth image (pooled, row-major): columns reverse (the camera sits on the body's center line),
- depth stats ``[left, center, right, front_min, left - right]``: left and right swap, the balance changes sign.

Spot in Isaac Sim is mirror-symmetric (link masses and centers of mass mirror exactly), so a mirrored state is as
valid as the original one.
"""

import torch

from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["MirrorSpec", "mirror_spec"]

_VEC_SIGNS = {
    "base_lin_vel": [1, -1, 1],
    "base_ang_vel": [-1, 1, -1],
    "projected_gravity": [1, -1, 1],
    "velocity_commands": [1, -1, -1],
}
_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")
_STATS_INDEX, _STATS_SIGN = [2, 1, 0, 3, 4], [1, 1, 1, 1, -1]


class MirrorSpec:
    """Signed permutations of the policy observation and the action."""

    def __init__(self, obs_index, obs_sign, act_index, act_sign, device):
        self.obs_index = torch.tensor(obs_index, device=device)
        self.obs_sign = torch.tensor(obs_sign, dtype=torch.float32, device=device)
        self.act_index = torch.tensor(act_index, device=device)
        self.act_sign = torch.tensor(act_sign, dtype=torch.float32, device=device)

    def obs(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.obs_index] * self.obs_sign

    def act(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.act_index] * self.act_sign


def _mirror_name(name: str) -> str:
    """fl_hx -> fr_hx, hr_kn -> hl_kn."""
    swap = {"fl": "fr", "fr": "fl", "hl": "hr", "hr": "hl"}
    leg, rest = name.split("_", 1)
    return f"{swap[leg]}_{rest}"


def _joint_mirror(names: list[str]) -> tuple[list[int], list[int]]:
    index = [names.index(_mirror_name(n)) for n in names]
    sign = [-1 if n.endswith("_hx") else 1 for n in names]
    return index, sign


def mirror_spec(env: ManagerBasedRLEnv, group: str = "policy") -> MirrorSpec:
    """Build the mirror from the env's observation terms (raises on a term it does not know)."""
    om = env.observation_manager
    joint_names = env.scene["robot"].joint_names
    j_index, j_sign = _joint_mirror(joint_names)
    act_term = env.action_manager.get_term("joint_pos")
    act_names = (
        [joint_names[i] for i in act_term._joint_ids] if not isinstance(act_term._joint_ids, slice) else joint_names
    )
    a_index, a_sign = _joint_mirror(act_names)

    camera = env.scene.sensors["train_camera"].cfg
    pooled_h, pooled_w = camera.height // 4, camera.width // 4
    flip = torch.arange(pooled_h * pooled_w).view(pooled_h, pooled_w).flip(1).flatten().tolist()

    index, sign, start = [], [], 0
    for name, (dim,) in zip(om.active_terms[group], om.group_obs_term_dim[group]):
        if name in _VEC_SIGNS:
            local, s = [0, 1, 2], _VEC_SIGNS[name]
        elif name in _JOINT_TERMS:
            assert dim == len(joint_names), name
            local, s = j_index, j_sign
        elif name == "camera_depth" and dim == pooled_h * pooled_w:
            local, s = flip, [1] * dim
        elif name == "camera_depth" and dim == pooled_h * pooled_w + 5:  # noisy_camera_depth_obs: image + stats
            local = flip + [pooled_h * pooled_w + i for i in _STATS_INDEX]
            s = [1] * (pooled_h * pooled_w) + _STATS_SIGN
        elif name == "camera_depth_stats":
            local, s = _STATS_INDEX, _STATS_SIGN
        else:
            raise ValueError(f"no mirror for observation term '{name}' ({dim})")
        assert len(local) == dim, name
        index += [start + i for i in local]
        sign += s
        start += dim
    return MirrorSpec(index, sign, a_index, a_sign, env.device)
