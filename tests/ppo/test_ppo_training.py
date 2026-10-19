import dataclasses

from rocketlander.config_files import TRAIN_CONFIG_FILE
from rocketlander.environment.env_config import EnvConfig
from rocketlander.ppo.ppo_training import build_train_config, train
from rocketlander.rocket.rocket_spec import RocketSpec


def test_default_config_loads_with_ppo_settings() -> None:
    train_config = build_train_config(TRAIN_CONFIG_FILE)
    assert train_config.level == 0
    assert train_config.parallel_rockets == 8
    assert isinstance(train_config.ppo["n_steps"], int)
    assert train_config.ppo["policy_kwargs"]["hidden_sizes"] == [64, 64]


def test_a_tiny_training_run_saves_everything(
    rocket_spec: RocketSpec, env_config: EnvConfig, tmp_path
) -> None:
    train_config = dataclasses.replace(
        build_train_config(TRAIN_CONFIG_FILE),
        total_steps=256,
        parallel_rockets=2,
        validate_every=128,
        validation_episodes=1,
        ppo={
            "n_steps": 64,
            "batch_size": 64,
            "n_epochs": 1,
            "policy_kwargs": {"hidden_sizes": [64, 64]},
        },
    )

    train(train_config, rocket_spec, env_config, tmp_path, None)

    for name in (
        "best_network.zip",
        "final_network.zip",
        "train_config.json",
        "env_config.json",
        "rocket_spec.json",
        "training.log",
    ):
        assert (tmp_path / name).exists(), name
    assert list(tmp_path.glob("tensorboard_*"))
