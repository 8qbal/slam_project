"""rl_games PPO with a mirror loss (Mittal et al., "Symmetry Considerations for Learning Task Symmetric Robot
Policies", ICRA 2024; the mirror-loss variant of Isaac Lab's ``RslRlSymmetryCfg``).

Adds ``mirror_loss_coef * (|mu(M s) - M mu(s)|^2 + |V(M s) - V(s)|^2)`` to the PPO loss, with ``M`` the left-right
mirror of ``spot_vslam.tasks.locomotion.mdp.symmetry``. A mirror-symmetric policy acts in a state with FR + HL on
the ground as it does in the mirrored state with FL + HR down, which the limp breaks. Checkpoints keep the
``a2c_continuous`` format, so play / gait_check / the navigation low level load them unchanged.

Used by ``scripts/rl_games/train.py`` when the agent cfg sets ``params.config.mirror_loss_coef``.
"""

import torch
from rl_games.algos_torch.a2c_continuous import A2CAgent

from spot_vslam.tasks.locomotion.mdp.symmetry import MirrorSpec


class MirrorA2CAgent(A2CAgent):
    def __init__(self, base_name, params, spec: MirrorSpec):
        super().__init__(base_name, params)
        self.mirror = spec
        self.mirror_coef = float(self.config["mirror_loss_coef"])
        self._mirror_obs = None
        self._mirror_losses = []

    def calc_gradients(self, input_dict):
        # obs as the base class feeds it to the model (before the model's own input normalization)
        self._mirror_obs = self._preproc_obs(input_dict["obs"])
        super().calc_gradients(input_dict)

    def calc_losses(
        self, actor_loss_func, old_logp, logp, adv, e_clip, old_values, values, returns, mu, entropy, masks
    ):
        loss, a_loss, c_loss, entropy, b_loss, sum_mask = super().calc_losses(
            actor_loss_func, old_logp, logp, adv, e_clip, old_values, values, returns, mu, entropy, masks
        )
        obs_m = self.mirror.obs(self._mirror_obs)
        mu_m, _, value_m, _ = self.model.a2c_network({"obs": self.model.norm_obs(obs_m)})
        action_loss = (mu_m - self.mirror.act(mu)).pow(2).mean()
        value_loss = (value_m - values).pow(2).mean()
        self._mirror_losses.append(torch.stack([action_loss.detach(), value_loss.detach()]))
        return loss + self.mirror_coef * (action_loss + value_loss), a_loss, c_loss, entropy, b_loss, sum_mask

    def train_epoch(self):
        result = super().train_epoch()
        if self._mirror_losses and self.writer is not None:
            action_loss, value_loss = torch.stack(self._mirror_losses).mean(dim=0).tolist()
            self.writer.add_scalar("losses/mirror_action", action_loss, self.frame)
            self.writer.add_scalar("losses/mirror_value", value_loss, self.frame)
        self._mirror_losses = []
        return result


def register(runner, spec: MirrorSpec):
    """Make ``runner`` build ``MirrorA2CAgent`` for ``a2c_continuous``."""
    runner.algo_factory.register_builder(
        "a2c_continuous", lambda **kwargs: MirrorA2CAgent(kwargs["base_name"], kwargs["params"], spec)
    )
