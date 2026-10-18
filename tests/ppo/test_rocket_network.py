import torch
from stable_baselines3 import PPO

from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.ppo.network_controller import ScaledObservation
from rocketlander.ppo.rocket_network import RocketNetwork, SB3PolicyAdapter


def test_network_has_separate_actor_and_critic_of_the_chosen_size() -> None:
    network = RocketNetwork(observation_size=8, hidden_sizes=[32, 16])

    actor, critic = network(torch.zeros(5, 8))  # a batch of 5 observations

    assert actor.shape == (5, 16) and critic.shape == (5, 16)
    assert (network.latent_dim_pi, network.latent_dim_vf) == (16, 16)
    shared = {id(p) for p in network.actor.parameters()} & {
        id(p) for p in network.critic.parameters()
    }
    assert not shared


def test_ppo_trains_our_network() -> None:
    ppo = PPO(
        SB3PolicyAdapter,
        ScaledObservation(RocketLandingEnv()),
        policy_kwargs={"hidden_sizes": [64, 64]},
        n_steps=64,
        batch_size=64,
        n_epochs=1,
    )
    before = [p.detach().clone() for p in ppo.policy.mlp_extractor.actor.parameters()]

    ppo.learn(64)

    assert isinstance(ppo.policy.mlp_extractor, RocketNetwork)
    after = list(ppo.policy.mlp_extractor.actor.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))
