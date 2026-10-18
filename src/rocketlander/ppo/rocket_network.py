"""Our neural network, written in PyTorch, and the policy that lets SB3's PPO train it.

    8 scaled observations ─┬─► actor:  Linear → Tanh → Linear → Tanh ─► throttle features
                           └─► critic: Linear → Tanh → Linear → Tanh ─► "how good is this?"

The actor decides what to do; the critic estimates how much reward is still to come, which
PPO uses during training to tell better-than-expected decisions from worse ones. After
training, only the actor flies the rocket.

SB3 adds the last layers: the actor's features → 3 throttle means (plus a learned spread for
the exploration noise), and the critic's features → 1 value. Those are tied to how PPO
computes probabilities, so they stay SB3's.
"""

from collections.abc import Sequence

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class RocketNetwork(nn.Module):
    """Two separate networks reading the same observations: the actor and the critic."""

    def __init__(self, observation_size: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.actor = _layers(observation_size, hidden_sizes)
        self.critic = _layers(observation_size, hidden_sizes)
        # SB3 reads these to size the final layers it adds.
        self.latent_dim_pi = hidden_sizes[-1]
        self.latent_dim_vf = hidden_sizes[-1]

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(observations), self.forward_critic(observations)

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)

    def forward_critic(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations)


def _layers(inputs: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    """Fully connected layers, each followed by Tanh (outputs between -1 and 1)."""
    layers: list[nn.Module] = []
    for size in hidden_sizes:
        layers += [nn.Linear(inputs, size), nn.Tanh()]
        inputs = size
    return nn.Sequential(*layers)


class SB3PolicyAdapter(ActorCriticPolicy):
    """SB3's actor-critic policy with RocketNetwork inside, for ``PPO(SB3PolicyAdapter, ...)``.

    Choose the layer sizes with ``policy_kwargs={"hidden_sizes": [64, 64]}``.
    """

    def __init__(self, *args, hidden_sizes: Sequence[int] = (64, 64), **kwargs) -> None:
        self.hidden_sizes = tuple(hidden_sizes)  # set first: the parent builds the network
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = RocketNetwork(self.features_dim, self.hidden_sizes)
