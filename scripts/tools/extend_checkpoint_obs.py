"""Add observation inputs to an rl_games checkpoint so a policy trained on N inputs can be fine-tuned on N + k.

The new columns of the first layer (and of its Adam moments) are zero, so the extended policy first behaves exactly like
the original one and learns to use the new inputs during fine-tuning. Epoch, frame and the rest are kept.

    uv run --no-sync python scripts/tools/extend_checkpoint_obs.py policies/nav_hall_racks/model.pth <out.pth> --extra 3
"""

import argparse

import torch

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--extra", type=int, required=True, help="number of new observation inputs (appended at the end)")
a = p.parse_args()

ckpt = torch.load(a.src, map_location="cpu", weights_only=False)
key = next(k for k in ckpt["model"] if k.endswith("actor_mlp.0.weight"))
old = ckpt["model"][key]


def widen(t: torch.Tensor) -> torch.Tensor:
    return torch.cat([t, torch.zeros(t.shape[0], a.extra, dtype=t.dtype)], dim=1)


ckpt["model"][key] = widen(old)
changed = 0
for state in ckpt["optimizer"]["state"].values():  # Adam moments of the same weight
    for name, t in state.items():
        if isinstance(t, torch.Tensor) and t.shape == old.shape:
            state[name] = widen(t)
            changed += 1
torch.save(ckpt, a.dst)
print(f"{key}: {tuple(old.shape)} -> {tuple(ckpt['model'][key].shape)}, {changed} optimizer tensors widened -> {a.dst}")
