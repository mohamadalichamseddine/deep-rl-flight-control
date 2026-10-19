import dataclasses
from pathlib import Path

from rocketlander.ppo.ppo_training import load_train_config, train

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "train" / "ppo.yaml"


def test_default_config_loads_with_ppo_settings() -> None:
    config = load_train_config(CONFIG)
    assert config.level == 0
    assert config.parallel_rockets == 8
    assert isinstance(config.ppo["n_steps"], int)
    assert config.ppo["policy_kwargs"]["hidden_sizes"] == [64, 64]


def test_a_tiny_training_run_saves_everything(tmp_path) -> None:
    config = dataclasses.replace(
        load_train_config(CONFIG),
        total_steps=256,
        parallel_rockets=2,
        validate_every=128,
        validation_episodes=1,
        ppo={"n_steps": 64, "batch_size": 64, "n_epochs": 1},
    )

    train(config, tmp_path)

    for name in (
        "best_network.zip",
        "final_network.zip",
        "config.json",
        "env_config.yaml",
        "training.log",
    ):
        assert (tmp_path / name).exists(), name
    assert list(tmp_path.glob("tensorboard_*"))
